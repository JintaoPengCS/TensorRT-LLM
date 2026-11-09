# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB replica copies on the copy engines through a symmetric staging buffer (two-hop prototype; selected by copy=ce).

Shipped path (balancer.py ``_copy_layers``): every per-slot plane of every replica goes through NCCL
``batch_isend_irecv`` on the forward stream between two replays. With NCCL_MAX_P2P_NCHANNELS=4 (the pre-connect of
every peer costs ~51 MB per peer at 4 channels, ~400 MB per peer at NCCL's default) a rank moves ~20-25 GB/s, a layer
takes ~2 ms (DEP16 median 2.05 ms, DEP32 2.89 ms), and a hot expert is sent once per destination by 4 NCCL CTAs.

This module (``copy=ce``):
  * every rank owns a small SYMMETRIC staging buffer: ``torch.distributed._symmetric_memory`` on the GenEPLB copy group,
    i.e. CUDA VMM memory exported as fabric handles and mapped by every peer of the NVL72 domain -- the provider MegaMoE
    already uses for its cross-node token buffers (cute_dsl_megamoe_custom_op.py);
  * an install chunk is cut into ROUNDS (double-buffered: round k uses half k % 2 of the staging buffer):
      1. stage: each source rank copies every (layer, expert) it must send ONCE (however many destinations) from its
         home slot into a staging slot -- a local D2D cudaMemcpyAsync (copy engine);
      2. sync: ``symm``  -> the symmetric-memory barrier kernel (1 CTA of max(32, R) threads, microseconds; default),
               ``memop`` -> SM-free point-to-point stream memory ops (cuStreamWriteValue32 / cuStreamWaitValue32 on
                            flags in the staging tail; ready + ack per (source, destination) pair; experimental),
               ``nccl``  -> a 1-element NCCL all-reduce (fallback);
      3. pull: each destination copies its replicas from the source's staging slot (a peer-mapped VA) into its spare
         slot -- cudaMemcpyAsync D2D over NVLink, executed by the copy engines (no SM);
  * everything is enqueued on the CURRENT (forward) stream between two replays, exactly where the shipped NCCL copies
    run: nothing overlaps MegaMoE's cross-rank barriers, and the next replay starts after the last pull.

Correctness (every rank makes the same calls: ``copy()`` runs the same agreed chunk at the same forward on every rank,
balancer.py, and the round schedule is a pure function of the plan):
  * RAW: a destination pulls round k only after the source's round-k staging finished (barrier, or the source's
    fenced READY write that the destination waits for);
  * WAR: a source re-stages half h (round k+2) only after every reader of its round-k data finished: with a barrier
    every rank's round-k pulls precede its round-(k+1) barrier; with memops the source waits for the readers' ACKs;
  * publish: a rank's ``done`` event is recorded after its own last pull, it votes only once that event completed and
    every rank publishes only after every rank voted (balancer.py, unchanged) -> every replica landed on every
    rank before any rank routes to it.

Data moves by raw device pointers (cuda-python runtime), never through torch views: ``begin_step`` runs under
``torch.inference_mode`` (runners/decoder/runner.py), so no view-mode / version-counter rules apply.
The pure-Python parts (``schedule``, ``program``) import nothing and are checked on a CPU by test_fastcopy.py.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable, Optional

COPY_NCCL, COPY_CE = 0, 1
MODE_NAMES = ("nccl", "ce")
SYNCS = ("symm", "memop", "nccl")
ALIGN = 256  # staging segment alignment, bytes
READY, ACK, BAR = 0, 1, 2  # memop flag planes in the staging tail: int32 flags[plane][peer]


def parse_copy_field(text: str) -> Optional[str]:
    """The copy mode a CTL text asks for: 'copy=ce', 'copy: nccl' or JSON '"copy": "ce"' anywhere -> 'ce' | 'nccl' |
    None (no field)."""
    m = re.search(r"""\bcopy["']?\s*[=:]\s*["']?(nccl|ce)\b""", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------------------------- pure-Python planning
def schedule(plans, home_slot: Callable[[int], int], M: int, R: int, me: int, per_half: int):
    """Round schedule of one copy chunk, seen from rank ``me`` (identical round count on every rank).

    plans: [{"layer": li, "copies": [(expert, destination global slot), ...]}, ...] (the balancer's install plans).
    -> (local, rounds)
       local:  [(layer, src local slot, dst local slot)] -- copies whose home and destination are both on ``me``;
       rounds: [{"stage":   [(staging index j, layer, src local slot)]          -- what ``me`` stages,
                 "pulls":   [(src rank, j, layer, dst local slot, expert)]      -- what ``me`` pulls,
                 "readers": [ranks pulling ``me``'s staged items this round],
                 "sources": [ranks ``me`` pulls from this round]}]
    Each source stages a (layer, expert) once, whatever its number of destinations; round k carries items
    k*per_half .. (k+1)*per_half-1 of every source (first-appearance order).
    """
    items, dests, local = {}, {}, []
    for p in plans:
        li = int(p["layer"])
        for expert, dst in p["copies"]:
            expert, dst = int(expert), int(dst)
            src = home_slot(expert)
            sr, dr = src // M, dst // M
            if sr == dr:
                if sr == me:
                    local.append((li, src % M, dst % M))
                continue
            key = (li, expert)
            if key not in dests:
                dests[key] = []
                items.setdefault(sr, []).append(key)
            dests[key].append(dst)
    n = max((-(-len(v) // per_half) for v in items.values()), default=0)
    rounds = []
    for k in range(n):
        stage, pulls, readers, sources = [], [], set(), set()
        for sr in sorted(items):
            for j, (li, e) in enumerate(items[sr][k * per_half : (k + 1) * per_half]):
                if sr == me:
                    stage.append((j, li, home_slot(e) % M))
                for dst in dests[(li, e)]:
                    dr = dst // M
                    if sr == me:
                        readers.add(dr)
                    if dr == me:
                        pulls.append((sr, j, li, dst % M, e))
                        sources.add(sr)
        # Rotate the sources (as torch's low-contention all-gather): destinations do not all start on one peer.
        pulls.sort(key=lambda x: ((x[0] - me) % R, x[1], x[2], x[3]))
        rounds.append(dict(stage=stage, pulls=pulls, readers=sorted(readers), sources=sorted(sources)))
    return local, rounds


def program(plans, *, home_slot, M, R, me, per_half, sync, round0, pending):
    """The op list ``me`` executes for one chunk (CeCopier.copy runs it; test_fastcopy.py simulates every rank's).

    round0: global rounds done so far (the same on every rank); pending: [dict, dict] per staging half, reader rank ->
    last round it read ``me``'s half (memop WAR state). -> (ops, new round count, new pending).
    ops: ("local", [(li, s, d)]) | ("stage", half, stage) | ("barrier",) | ("pull", half, pulls)
         | ("signal", target rank, plane, value)   -- target.flags[plane][me] = value (fenced stream write)
         | ("wait", plane, peer, value)            -- wait until my flags[plane][peer] >= value (stream wait)
    """
    if sync not in SYNCS:
        raise ValueError(f"sync {sync!r} not in {SYNCS}")
    local, rounds = schedule(plans, home_slot, M, R, me, per_half)
    ops = [("local", local)] if local else []
    pending = [dict(pending[0]), dict(pending[1])]
    rnd = round0
    for rd in rounds:
        rnd += 1
        h = rnd & 1
        if rd["stage"]:
            if sync == "memop":  # WAR: every earlier reader of half h is done with it
                for d, r in sorted(pending[h].items()):
                    ops.append(("wait", ACK, d, r))
                pending[h] = {}
            ops.append(("stage", h, rd["stage"]))
        if sync == "memop":
            for d in rd["readers"]:  # RAW: my staging of this round is complete (the write is fenced)
                ops.append(("signal", d, READY, rnd))
                pending[h][d] = rnd
            for s in rd["sources"]:
                ops.append(("wait", READY, s, rnd))
        else:
            ops.append(("barrier",))  # every rank, every round (RAW and WAR at once)
        if rd["pulls"]:
            ops.append(("pull", h, rd["pulls"]))
            if sync == "memop":
                for s in rd["sources"]:  # my pulls of this round are complete: s may re-stage the half
                    ops.append(("signal", s, ACK, rnd))
    return ops, rnd, pending


def work(plans, home_slot, M, R):
    """Per-rank traffic of a chunk: (in replicas, out replicas, staged items, local copies) lists indexed by rank."""
    n_in, n_out, staged, local = [0] * R, [0] * R, [0] * R, [0] * R
    seen = set()
    for p in plans:
        for expert, dst in p["copies"]:
            sr, dr = home_slot(int(expert)) // M, int(dst) // M
            if sr == dr:
                local[sr] += 1
                continue
            n_in[dr] += 1
            n_out[sr] += 1
            if (p["layer"], expert) not in seen:
                seen.add((p["layer"], expert))
                staged[sr] += 1
    return n_in, n_out, staged, local


def _enum_value(mod, cls, member, default):
    """An enum member of cuda-python as int, or its documented value if this binding names it differently."""
    try:
        return int(getattr(getattr(mod, cls), member))
    except AttributeError:
        return default


def _slices(ops, size):
    """Split (dst, src, n) copies into <= size pieces, interleaved across copies (DWDP's contention_opt order,
    weight_manager.py): consecutive batch entries touch different sources."""
    per = []
    for d, s, n in ops:
        per.append([(d + o, s + o, min(size, n - o)) for o in range(0, n, size)])
    out, i = [], 0
    while any(i < len(x) for x in per):
        out += [x[i] for x in per if i < len(x)]
        i += 1
    return out


# ---------------------------------------------------------------------------------------------- the copier
class CeCopier:
    """Copy-engine replica copies through a symmetric staging buffer (see the module docstring).

    Two phases so that a failing rank never leaves its peers in a collective: ``CeCopier(...)`` is local (checks the
    slot planes, allocates this rank's staging buffer); ``connect()`` is the collective rendezvous on ``group``.
    Knobs (constructor argument, else env): per_half TLLM_GENEPLB_COPY_CE_SLOTS (1: 2 staging slots, ~71 MiB/rank for
    DSV4-Pro), sync TLLM_GENEPLB_COPY_CE_SYNC (symm|memop|nccl), batch TLLM_GENEPLB_COPY_CE_BATCH (0|1:
    cudaMemcpyBatchAsync per op group), slice TLLM_GENEPLB_COPY_CE_SLICE (bytes; batch only; 0 = whole planes),
    timeout TLLM_GENEPLB_COPY_CE_TIMEOUT_MS (symm barrier; 0 = wait forever).
    """

    def __init__(self, slot_tensors, *, home_slot, local_slots, ranks, rank, group, device, per_half=None,
                 sync=None, batch=None, slice_bytes=None, timeout_ms=None):
        import torch
        import torch.distributed._symmetric_memory as symm
        from cuda.bindings import driver as cu
        from cuda.bindings import runtime as rt

        env = os.environ.get
        self.per_half = int(per_half if per_half is not None else env("TLLM_GENEPLB_COPY_CE_SLOTS", "1"))
        self.sync = str(sync if sync is not None else env("TLLM_GENEPLB_COPY_CE_SYNC", "symm")).lower()
        self.batch = bool(int(batch if batch is not None else env("TLLM_GENEPLB_COPY_CE_BATCH", "0")))
        self.slice_bytes = int(slice_bytes if slice_bytes is not None else env("TLLM_GENEPLB_COPY_CE_SLICE", "0"))
        self.timeout_ms = int(timeout_ms if timeout_ms is not None else env("TLLM_GENEPLB_COPY_CE_TIMEOUT_MS", "600000"))
        if self.sync not in SYNCS or self.per_half < 1:
            raise ValueError(f"GenEPLB CE copy: sync {self.sync!r} (one of {SYNCS}), slots per half {self.per_half}")
        self.home_slot, self.M, self.R, self.rank = home_slot, int(local_slots), int(ranks), int(rank)
        self.group, self.device = group, torch.device(device)
        self._cu, self._rt = cu, rt
        self._ok_cu, self._ok_rt = cu.CUresult.CUDA_SUCCESS, rt.cudaError_t.cudaSuccess
        self._kind = rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        # CUDA driver API values (cuda.h): CU_STREAM_WRITE_VALUE_DEFAULT = 0 (the write is preceded by a memory fence
        # over the stream's prior work), CU_STREAM_WAIT_VALUE_GEQ = 0 (cyclic (int32)(*addr - value) >= 0).
        self._wflag = _enum_value(cu, "CUstreamWriteValue_flags", "CU_STREAM_WRITE_VALUE_DEFAULT", 0)
        self._wait_geq = _enum_value(cu, "CUstreamWaitValue_flags", "CU_STREAM_WAIT_VALUE_GEQ", 0)
        # Per-slot planes (balancer._slot_tensors, sorted by name): one geometry on every layer, slot-major contiguous.
        ref, self.base = None, []
        for li, planes in enumerate(slot_tensors):
            sig, ptrs = [], []
            for name, t in planes:
                if t.shape[0] != self.M or not t.is_contiguous() or t.device != self.device:
                    raise RuntimeError(
                        f"GenEPLB CE copy: layer {li} plane {name} {tuple(t.shape)} on {t.device} is not a contiguous "
                        f"slot-major tensor with {self.M} slots on {self.device}"
                    )
                sig.append((name, t.numel() // self.M * t.element_size()))
                ptrs.append(int(t.data_ptr()))
            if ref is None:
                ref = sig
            elif sig != ref:
                raise RuntimeError(f"GenEPLB CE copy: layer {li} planes {sig} differ from layer 0 {ref}")
            self.base.append(ptrs)
        self.planes = ref
        self.seg, off = [], 0
        for _, n in ref:
            self.seg.append((off, n))
            off += -(-n // ALIGN) * ALIGN
        self.item = off  # one staged (layer, expert): every plane of its slot, 256-B aligned segments
        self.flags_off = 2 * self.per_half * self.item
        self.nbytes = self.flags_off + -(-(3 * self.R * 4) // ALIGN) * ALIGN
        self._symm = symm
        self.gname = str(group.group_name)
        symm.enable_symm_mem_for_group(self.gname)  # idempotent (MegaMoE does the same on its EP group)
        self.buf = symm.empty(self.nbytes, dtype=torch.uint8, device=self.device)
        self.buf.zero_()  # memop flags start at 0 (rounds count from 1)
        torch.cuda.synchronize(self.device)
        self.hdl, self.ptrs, self.connected = None, None, False
        self.round, self.epoch, self.pending = 0, 0, [{}, {}]
        self.last_host_ms = 0.0
        self._one = torch.zeros(1, dtype=torch.int32, device=self.device) if self.sync == "nccl" else None
        self._attr = None

    def connect(self) -> "CeCopier":
        """Collective on the copy group: map every peer's staging buffer (fabric handles across the NVL72 nodes)."""
        self.hdl = self._symm.rendezvous(self.buf, self.gname)
        base = [int(p) for p in self.hdl.buffer_ptrs]  # block bases (CUDASymmetricMemory.cu), 0 = unmapped
        if len(base) != self.R or not all(base):
            raise RuntimeError(f"GenEPLB CE copy: peer staging pointers {base} (need {self.R}, all mapped)")
        off = int(self.buf.data_ptr()) - base[self.rank]  # the handle's offset into its block (same on every rank)
        if off < 0 or off != int(getattr(self.hdl, "offset", off)):
            raise RuntimeError(f"GenEPLB CE copy: staging offset {off} vs handle offset {getattr(self.hdl, 'offset', None)}")
        self.ptrs = [p + off for p in base]
        self.connected = True
        return self

    def describe(self) -> str:
        return (
            f"staging {2 * self.per_half} x {self.item / 2**20:.1f} MiB = {self.nbytes / 2**20:.1f} MiB symmetric/rank, "
            f"sync {self.sync}, batch {int(self.batch)}" + (f", slice {self.slice_bytes >> 10} KiB" if self.slice_bytes else "")
        )

    # -------------------------------------------------------------------------------------- copy
    def copy(self, plans):
        """Copy the plans' replicas on the current stream (every rank, same plans): (start, done) timing events."""
        import torch

        if not self.connected:
            raise RuntimeError("GenEPLB CE copy: connect() was not called")
        stream = torch.cuda.current_stream(self.device)
        sh = stream.cuda_stream
        t0 = time.perf_counter()
        ops, self.round, self.pending = program(
            plans, home_slot=self.home_slot, M=self.M, R=self.R, me=self.rank, per_half=self.per_half,
            sync=self.sync, round0=self.round, pending=self.pending,
        )
        start, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for op in ops:
            self._exec(op, sh)
        done.record(stream)
        self.last_host_ms = (time.perf_counter() - t0) * 1e3
        return start, done

    def _slot(self, li, k, s):
        return self.base[li][k] + s * self.planes[k][1]

    def _stage_off(self, h, j):
        return (h * self.per_half + j) * self.item

    def _flag(self, owner, plane, idx):
        return self.ptrs[owner] + self.flags_off + 4 * (plane * self.R + idx)

    def _exec(self, op, sh):
        kind = op[0]
        if kind == "local":
            self._memcpy([(self._slot(li, k, d), self._slot(li, k, s), n)
                          for li, s, d in op[1] for k, (_, n) in enumerate(self.planes)], sh)
        elif kind == "stage":
            h, me = op[1], self.ptrs[self.rank]
            self._memcpy([(me + self._stage_off(h, j) + o, self._slot(li, k, s), n)
                          for j, li, s in op[2] for k, (o, n) in enumerate(self.seg)], sh)
        elif kind == "pull":
            h = op[1]
            self._memcpy([(self._slot(li, k, d), self.ptrs[sr] + self._stage_off(h, j) + o, n)
                          for sr, j, li, d, _ in op[2] for k, (o, n) in enumerate(self.seg)], sh)
        elif kind == "barrier":
            self._barrier()
        elif kind == "signal":
            self._write(self._flag(op[1], op[2], self.rank), op[3], sh)
        elif kind == "wait":
            self._wait(self._flag(self.rank, op[1], op[2]), op[3], sh)
        else:
            raise ValueError(f"GenEPLB CE copy: op {op!r}")

    def _memcpy(self, ops, sh):
        if not ops:
            return
        if self.batch and len(ops) > 1:
            if self.slice_bytes > 0:
                ops = _slices(ops, self.slice_bytes)
            if self._attr is None:
                attr = self._rt.cudaMemcpyAttributes()
                attr.srcAccessOrder = self._rt.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
                self._attr = attr
            # As DWDP's weight_manager.py: CUDA 13 bindings return (err,).
            err = self._rt.cudaMemcpyBatchAsync(
                [d for d, _, _ in ops], [s for _, s, _ in ops], [n for _, _, n in ops], len(ops),
                [self._attr], [0], 1, sh,
            )[0]
            if err != self._ok_rt:
                raise RuntimeError(f"GenEPLB CE copy: cudaMemcpyBatchAsync ({len(ops)} copies): {err}")
            return
        for d, s, n in ops:
            err = self._rt.cudaMemcpyAsync(d, s, n, self._kind, sh)[0]
            if err != self._ok_rt:
                raise RuntimeError(f"GenEPLB CE copy: cudaMemcpyAsync({d:#x} <- {s:#x}, {n}): {err}")

    def _barrier(self):
        if self.sync == "symm":
            self.hdl.barrier(channel=0, timeout_ms=self.timeout_ms)  # 1 CTA on the current stream
        elif self.sync == "nccl":
            import torch.distributed as dist

            dist.all_reduce(self._one, group=self.group)  # the current stream waits for it
        else:
            raise RuntimeError("GenEPLB CE copy: memop sync has no barrier op")

    def _write(self, addr, value, sh):
        err = self._cu.cuStreamWriteValue32(sh, addr, value & 0xFFFFFFFF, self._wflag)[0]
        if err != self._ok_cu:
            raise RuntimeError(f"GenEPLB CE copy: cuStreamWriteValue32({addr:#x}, {value}): {err}")

    def _wait(self, addr, value, sh):
        err = self._cu.cuStreamWaitValue32(sh, addr, value & 0xFFFFFFFF, self._wait_geq)[0]
        if err != self._ok_cu:
            raise RuntimeError(f"GenEPLB CE copy: cuStreamWaitValue32({addr:#x}, {value}): {err}")

    def barrier_all(self):
        """A full barrier of every rank with this copier's sync (microbenchmark; memop uses the BAR flag plane)."""
        import torch

        if self.sync != "memop":
            self._barrier()
            return
        sh = torch.cuda.current_stream(self.device).cuda_stream
        self.epoch += 1
        for p in range(self.R):
            if p != self.rank:
                self._write(self._flag(p, BAR, self.rank), self.epoch, sh)
        for p in range(self.R):
            if p != self.rank:
                self._wait(self._flag(self.rank, BAR, p), self.epoch, sh)


# ---------------------------------------------------------------------------------------------- shipped NCCL reference
def nccl_copy(slot_tensors, home_slot, M, rank, group, plans):
    """The shipped copy (balancer.py without its first-use connection probes): microbenchmark reference."""
    import torch
    import torch.distributed as dist

    ops, local = [], []
    for p in plans:
        for expert, dst in p["copies"]:
            src = home_slot(int(expert))
            sr, dr = src // M, int(dst) // M
            for _, t in slot_tensors[p["layer"]]:
                if sr == dr == rank:
                    local.append((t[int(dst) % M], t[src % M]))
                elif sr == rank:
                    ops.append(dist.P2POp(dist.isend, t[src % M], dr, group=group))
                elif dr == rank:
                    ops.append(dist.P2POp(dist.irecv, t[int(dst) % M], sr, group=group))
    start, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for dst_t, src_t in local:
        dst_t.copy_(src_t, non_blocking=True)
    for w in dist.batch_isend_irecv(ops) if ops else []:
        w.wait()
    done.record()
    return start, done


def nccl_connect(group, R, rank, device, nbytes=4 << 20):
    """The shipped pre-connect (balancer.py): -> (seconds, MB of device memory NCCL took)."""
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    send = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    recv = torch.empty((R, nbytes), dtype=torch.uint8, device=device)
    free0 = torch.cuda.mem_get_info(device)[0]
    peers = [p for p in range(R) if p != rank]
    ops = [dist.P2POp(dist.isend, send, p, group=group) for p in peers]
    ops += [dist.P2POp(dist.irecv, recv[p], p, group=group) for p in peers]
    for w in dist.batch_isend_irecv(ops) if ops else []:
        w.wait()
    torch.cuda.synchronize()
    mb = (free0 - torch.cuda.mem_get_info(device)[0]) / 2**20
    return time.perf_counter() - t0, mb
