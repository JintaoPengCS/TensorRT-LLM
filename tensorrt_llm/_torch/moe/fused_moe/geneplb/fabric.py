# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB's MegaMoE weight planes in fabric memory and single-hop
copy-engine replica installs.

Allocation (TLLM_GENEPLB_FABRIC=1, read at import; default off = the shipped torch allocation). At the first GenEPLB
layer's create_weights (quantization.py NVFP4MegaMoECuteDslMethod.create_weights, the HALO-Q arena hook point) every rank
creates ONE fabric allocation per MegaMoE plane spanning every GenEPLB layer (sami/fabric.py FabricAllocation), laid out
[L][M][row] with the layer stride padded to LAYER_ALIGN, exports the 64-B handles, all-gathers them over MPI (never
torch.distributed, which is NCCL here) and imports + maps every peer's seven allocations IN FULL: a device VA for every
(rank, plane, layer, slot), 7 x (EP - 1) imports per rank (the GB NVLink fabric routing table holds ~928 entries per GPU,
one per mapped import: dwdp/weight_buffer.py; a GB-class fabric import is mapped whole, so a peer's spare rows
are readable too and a copy may read ANY current holder of an expert). Each layer's seven Parameters become views of its
[M] rows before any weight is loaded (quantization.py _bind_live_weight_planes), so loading writes straight
into the arena and no torch storage of the planes is ever allocated (MetaInitMode leaves them meta until then);
cache_derived_state asserts the aliases after the transform (assert_live_weight_aliases) and GenEPLB's
finalize_model checks every layer's slot tensors against the arena on every rank before the first forward. KV-cache
sizing counts the VMM bytes as memory outside torch (pyexecutor/_util.py).

Copies (TLLM_GENEPLB_COPY=fabric | both, balancer.py): one copy-engine row per (replica, plane) from the chosen holder's
resident row straight into the destination's FREE slot (no staging, no NCCL), one cudaMemcpyBatchAsync per <= 64 rows
(the CUDA 13 8-argument form of dwdp/weight_manager.py; srcAccessOrder STREAM, PREFER_OVERLAP_WITH_COMPUTE) on a
dedicated stream created with cudaStreamCreateWithFlags(non-blocking) and wrapped as torch.cuda.ExternalStream (a
torch.cuda.Stream() is a pooled round-robin stream that can alias other aux streams; a batch on the legacy NULL stream is
cudaErrorInvalidValue). Pre-created events only (a small ring): the copy stream waits the WAR event recorded on the
forward stream after the last table write, then records start / done (E).

Pure helpers (geometry, rows, batches) import nothing; torch and cuda-python are imported where they are used, so the
CPU-only tests run them without a GPU.
"""

from __future__ import annotations

import os
import time

FABRIC = os.environ.get("TLLM_GENEPLB_FABRIC", "0") == "1"
# Debug: after each install (and the startup probe) compare every copied row with the expert's HOME row (the canonical
# copy) through peer views on a verify stream; the mismatched rows are counted in the install / renewal lines.
VERIFY = os.environ.get("TLLM_GENEPLB_VERIFY_COPY", "0") == "1"
# nsys: cudaProfilerStart right before the copies of the k-th fabric install (1-based; 0 = off), cudaProfilerStop
# NSYS_STEPS forwards after its publish (run under nsys --capture-range=cudaProfilerApi --capture-range-end=stop).
NSYS_INSTALL = int(os.environ.get("TLLM_GENEPLB_NSYS_INSTALL", "0") or 0)
NSYS_STEPS = int(os.environ.get("TLLM_GENEPLB_NSYS_STEPS", "8") or 8)
MAX_ROWS = 64  # rows per cudaMemcpyBatchAsync (the reference install path's bound; one install = ceil(rows / 64) calls)
# Fallback knob, default 0 = ONE allocation per plane spanning every layer. k > 0: a plane's layers are split into
# allocations of k layers each (imports 7 x ceil(L / k) x (EP - 1)), only if a single allocation of a plane's whole span
# (e.g. ~35 GiB for mega_fc1_weight at DSV4-Pro DEP16) is refused by the driver / fabric.
CHUNK_LAYERS = int(os.environ.get("TLLM_GENEPLB_FABRIC_CHUNK_LAYERS", "0") or 0)
LAYER_ALIGN = 4096  # every layer's [M] rows of a plane start 4 KiB-aligned (the planes' own rows are 512-B multiples)
EVENT_RING = 4  # pre-created (start, done) pairs: one install in flight + the startup probe


def align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


# ------------------------------------------------------------------------------------------------------- pure helpers
def install_rows(plans, *, rank: int, M: int, home_slot, addr, rows):
    """This rank's copy rows of an install: every copy (expert, destination slot) whose destination is on ``rank``, one row
    per plane, read from the copy's source slot (plan["sources"][j], a current holder chosen by the planner; the expert's
    home without one). Replicas are interleaved round-robin over their source ranks (every source link busy at once).
    -> dict(dst, src, size, ref, replicas, layers): ``ref`` is the expert's HOME row (the canonical copy) for VERIFY."""
    groups = {}
    layers = set()
    for p in plans:
        li, srcs = int(p["layer"]), p.get("sources")
        for j, (e, d) in enumerate(p["copies"]):
            e, d = int(e), int(d)
            if d // M != rank:
                continue
            s = int(srcs[j]) if srcs is not None else int(home_slot(e))
            groups.setdefault(s // M, []).append((li, s, d, e))
            layers.add(li)
    queues = [groups[r] for r in sorted(groups)]
    order = []
    for k in range(max((len(q) for q in queues), default=0)):
        order.extend(q[k] for q in queues if k < len(q))
    out = dict(dst=[], src=[], size=[], ref=[], replicas=len(order), layers=len(layers))
    for li, s, d, e in order:
        h = int(home_slot(e))
        for k, nb in enumerate(rows):
            out["dst"].append(addr(d // M, k, li, d % M))
            out["src"].append(addr(s // M, k, li, s % M))
            out["size"].append(int(nb))
            out["ref"].append(addr(h // M, k, li, h % M))
    return out


def batches(n: int, k: int = MAX_ROWS):
    """[lo, hi) row ranges of the batch calls of n rows."""
    return [(lo, min(n, lo + k)) for lo in range(0, n, k)]


def profiler(on: bool) -> None:
    """cudaProfilerStart / Stop: the range an nsys --capture-range=cudaProfilerApi run records (a no-op without one)."""
    import torch

    rt = torch.cuda.cudart()
    (rt.cudaProfilerStart if on else rt.cudaProfilerStop)()


# ------------------------------------------------------------------------------------------------------- the arena
class _SamiAlloc:
    """FabricAllocation (sami/fabric.py): cuMemCreate (PINNED, FABRIC
    handle type, recommended granularity), local map + RW access, zeroed, 64-B export; import_peer maps a peer's whole
    allocation (same size on every rank)."""

    def __init__(self, device: int):
        self.device = int(device)

    def create(self, nbytes: int):
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.fabric import FabricAllocation

        return FabricAllocation.create(int(nbytes), self.device)


class FabricLayerArena:
    """One layer's seven [M]-row plane views in the duck type quantization.py's _bind_live_weight_planes /
    assert_live_weight_aliases take (bundle, plane_names, local_plane_views, tekit_alias_views)."""

    def __init__(self, bundle, views):
        self.bundle = bundle
        self.plane_names = tuple(p.name for p in bundle.planes)
        self.local_plane_views = tuple(views)
        self.tekit_alias_views = tuple(views)


class FabricArena:
    """Every GenEPLB layer's seven MegaMoE planes: ONE fabric allocation per plane, [L][M][row]."""

    def __init__(self, *, names, rows, layers: int, slots: int, ranks: int, rank: int, device: int, bundle=None,
                 chunk_layers: int = None):
        self.names = tuple(names)
        self.rows = tuple(int(r) for r in rows)
        self.L, self.M, self.R, self.rank, self.device = int(layers), int(slots), int(ranks), int(rank), int(device)
        self.bundle = bundle
        self.stride = tuple(align_up(self.M * r, LAYER_ALIGN) for r in self.rows)  # bytes of one layer's [M] rows
        per = CHUNK_LAYERS if chunk_layers is None else int(chunk_layers)
        self.per = per if 0 < per < self.L else self.L  # layers per allocation (default: all of them, one allocation)
        self.chunks = -(-self.L // self.per)
        self.nbytes = tuple(self.per * s for s in self.stride)  # bytes of one allocation of each plane
        self.local = []  # this rank's allocation objects (owners), plane-major: [plane * chunks + chunk]
        self.sizes = ()  # their allocated (granularity-rounded) sizes
        self.base = None  # [rank][plane * chunks + chunk] -> VA of that rank's allocation, mapped here
        self.views = {}  # layer -> FabricLayerArena
        self.imports = 0
        self.seconds = 0.0

    @classmethod
    def create(cls, bundle, *, layers: int, slots: int, ranks: int, rank: int, device: int, comm, alloc=None,
               chunk_layers: int = None):
        """Collective over the EP ranks (MPI ``comm``): every rank creates its seven allocations, the handles are
        all-gathered and every peer's allocations imported. A failure on any rank raises on every rank."""
        arena = cls(names=[p.name for p in bundle.planes], rows=[p.nbytes for p in bundle.planes], layers=layers,
                    slots=slots, ranks=ranks, rank=rank, device=device, bundle=bundle, chunk_layers=chunk_layers)
        arena.allocate(comm, alloc if alloc is not None else _SamiAlloc(device))
        return arena

    def allocate(self, comm, alloc) -> None:
        t0 = time.perf_counter()
        err, handles = None, []
        try:
            for n in self.nbytes:
                for _ in range(self.chunks):
                    self.local.append(alloc.create(n))
            handles = [bytes(a.shareable) for a in self.local]
            self.sizes = tuple(int(a.size) for a in self.local)
            if any(len(h) != 64 for h in handles):
                raise RuntimeError("a fabric handle is not 64 bytes")
        except Exception as exc:  # every rank learns it below and raises
            err = f"{type(exc).__name__}: {exc}"
        got = comm.allgather((self.rank, err, handles, self.sizes))
        errors = [f"rank {r}: {e}" for r, e, _, _ in got if e]
        if errors:
            raise RuntimeError("GenEPLB fabric arena: allocation failed: " + "; ".join(errors))
        peers = {int(r): (h, s) for r, _, h, s in got}
        if sorted(peers) != list(range(self.R)):
            raise RuntimeError(f"GenEPLB fabric arena: handle exchange returned ranks {sorted(peers)}, EP {self.R}")
        if any(tuple(s) != self.sizes for _, s in peers.values()):
            raise RuntimeError("GenEPLB fabric arena: the ranks' allocation sizes differ")
        err, base = None, [[0] * len(self.local) for _ in range(self.R)]
        try:
            for peer in range(self.R):
                if peer == self.rank:
                    base[peer] = [int(a.local_ptr) for a in self.local]
                    continue
                for k, a in enumerate(self.local):  # maps the peer's WHOLE allocation (same size as ours)
                    base[peer][k] = int(a.import_peer(peers[peer][0][k]))
                    self.imports += 1
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        errors = [f"rank {r}: {e}" for r, e in enumerate(comm.allgather(err)) if e]
        if errors:
            raise RuntimeError("GenEPLB fabric arena: peer import failed: " + "; ".join(errors))
        self.base = base
        self.seconds = time.perf_counter() - t0

    def addr(self, rank: int, plane: int, layer: int, slot: int) -> int:
        chunk, li = divmod(layer, self.per)
        return self.base[rank][plane * self.chunks + chunk] + li * self.stride[plane] + slot * self.rows[plane]

    def layer_arena(self, layer: int, bundle, slots: int) -> FabricLayerArena:
        """Layer ``layer``'s [M]-row storage views of the seven planes (memoized: create_weights runs once per module)."""
        if layer in self.views:
            return self.views[layer]
        if not 0 <= int(layer) < self.L:
            raise RuntimeError(f"GenEPLB fabric arena holds {self.L} layers (the model's body); layer {layer} is past it")
        if int(slots) != self.M:
            raise RuntimeError(f"GenEPLB fabric arena: the module has {slots} compute slots, the arena {self.M}")
        if self.bundle is not None and bundle != self.bundle:
            raise RuntimeError("GenEPLB fabric arena: every GenEPLB layer must share one seven-plane geometry")
        views = [plane_view(self.addr(self.rank, k, layer, 0), plane, self.M, self.device)
                 for k, plane in enumerate(bundle.planes)]
        self.views[layer] = FabricLayerArena(bundle, views)
        return self.views[layer]

    def check_layers(self, slot_tensors) -> None:
        """Every GenEPLB layer's slot tensors are exactly its arena rows (raises otherwise): a Parameter replaced after
        create_weights would leave the copies writing a stale buffer."""
        if len(slot_tensors) > self.L:
            raise RuntimeError(f"{len(slot_tensors)} GenEPLB layers, the fabric arena holds {self.L}")
        for li, named in enumerate(slot_tensors):
            if li not in self.views:
                raise RuntimeError(f"layer {li} was never bound to the fabric arena (create_weights hook)")
            got = dict(named)
            if set(got) != set(self.names):
                raise RuntimeError(f"layer {li}: slot tensors {sorted(got)} != the arena planes {sorted(self.names)}")
            for k, name in enumerate(self.names):
                t = got[name]
                ptr, nbytes = int(t.data_ptr()), int(t.numel()) * int(t.element_size())
                if ptr != self.addr(self.rank, k, li, 0) or nbytes != self.M * self.rows[k]:
                    raise RuntimeError(f"layer {li} plane {name}: slot tensor 0x{ptr:x} ({nbytes} B) is not the arena "
                                       f"rows 0x{self.addr(self.rank, k, li, 0):x} ({self.M * self.rows[k]} B)")

    def describe(self) -> str:
        gib = sum(self.sizes) / 2**30
        alloc = ("one fabric allocation per plane" if self.chunks == 1 else
                 f"{self.chunks} fabric allocations per plane ({self.per} layers each, TLLM_GENEPLB_FABRIC_CHUNK_LAYERS)")
        return (f"{len(self.names)} planes x {self.L} layers x {self.M} slots, {alloc} ({gib:.2f} GiB/rank, row bytes "
                f"{list(self.rows)}), {self.imports} peer imports ({len(self.local)} x {self.R - 1}, each mapped whole), "
                f"create + exchange + import {self.seconds:.2f} s")


# ------------------------------------------------------------------------------------------------------- torch views
class _RawPointer:
    """A CUDA address through __cuda_array_interface__ (per_layer_dynamic_eplb/arena.py); torch keeps the
    holder alive with the tensor. The arena owns the memory (process lifetime)."""

    def __init__(self, pointer: int, nbytes: int):
        self.__cuda_array_interface__ = {"data": (int(pointer), False), "shape": (int(nbytes),), "typestr": "|u1",
                                         "strides": None, "version": 3}


def raw_u8(pointer: int, nbytes: int, device):
    """A uint8 tensor over [pointer, pointer + nbytes) on ``device`` (an ordinal or a torch.device)."""
    import torch

    return torch.as_tensor(_RawPointer(pointer, nbytes), device=torch.device("cuda", int(getattr(device, "index", device))))


def plane_view(pointer: int, plane, slots: int, device: int):
    """The framework storage view of one plane's [slots] rows at ``pointer`` (per_layer_dynamic_eplb/arena.py,
    kernel=False): the shape / stride / dtype quantization.py registers (plane.storage_metadata)."""
    import torch

    shape, stride, dtype_name, element_size, alignment = plane.storage_metadata(int(slots))
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise RuntimeError(f"torch has no dtype {dtype_name!r} for plane {plane.name}")
    typed = raw_u8(pointer, int(slots) * int(plane.nbytes), device).view(dtype)
    if typed.element_size() != element_size:
        raise RuntimeError(f"{plane.name}: element size {typed.element_size()} != {element_size}")
    if int(typed.data_ptr()) % alignment:
        raise RuntimeError(f"{plane.name}: pointer 0x{int(typed.data_ptr()):x} is not {alignment}-B aligned")
    return typed.as_strided(shape, stride)


# ------------------------------------------------------------------------------------------------------- copy engine
def _enum(mod, cls: str, member: str, default):
    try:
        return getattr(getattr(mod, cls), member)
    except AttributeError:
        return default


class FabricCopier:
    """The copy engine side of an install: a dedicated non-blocking stream, pre-created events, batched CE rows."""

    def __init__(self, arena: FabricArena, device):
        import torch
        from cuda.bindings import runtime as cudart

        self.arena, self.cudart = arena, cudart
        self.device = torch.device("cuda", int(getattr(device, "index", device) or 0))
        self._ok = cudart.cudaError_t.cudaSuccess
        self.stream, self._raw = self._create_stream()
        # VERIFY: its own stream (compare kernels never run on the copy stream) and a local scratch row
        self.vstream, self._vraw = self._create_stream() if VERIFY else (None, None)
        self._scratch = None
        attr = cudart.cudaMemcpyAttributes()
        attr.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
        flag = _enum(cudart, "cudaMemcpyFlags", "cudaMemcpyFlagPreferOverlapWithCompute", 1)
        try:
            attr.flags = int(flag)
        except TypeError:
            attr.flags = flag
        self._attrs = [attr]
        with torch.cuda.device(self.device):
            self.ring = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                         for _ in range(EVENT_RING)]
            self.war = torch.cuda.Event()  # recorded on the forward stream after every table write
            for start, done in self.ring:  # torch creates the driver event at its first record: do it now
                start.record(self.stream)
                done.record(self.stream)
            self.war.record(torch.cuda.current_stream())
            self.stream.synchronize()
        self._next = 0
        self.calls = 0

    def _create_stream(self):
        import torch

        cudart = self.cudart
        with torch.cuda.device(self.device):
            err, raw = cudart.cudaStreamCreateWithFlags(int(getattr(cudart, "cudaStreamNonBlocking", 1)))
        if err != self._ok:
            raise RuntimeError(f"cudaStreamCreateWithFlags(non-blocking) failed: {err}")
        return torch.cuda.ExternalStream(int(raw), device=self.device), raw

    def _acquire(self):
        start, done = self.ring[self._next % len(self.ring)]
        if not done.query():
            raise RuntimeError("GenEPLB fabric: event ring exhausted (an earlier install's copies are still in flight)")
        self._next += 1
        return start, done

    def issue(self, rows: dict, war=None):
        """Enqueue the rows on the copy stream after ``war`` (the forward stream's last table write): -> (start, done)."""
        import torch

        start, done = self._acquire()
        dst, src, size = rows["dst"], rows["src"], rows["size"]
        with torch.cuda.device(self.device):
            if war is not None:
                self.stream.wait_event(war)
            start.record(self.stream)
            for lo, hi in batches(len(dst)):
                self._batch(dst[lo:hi], src[lo:hi], size[lo:hi], self._raw)
            done.record(self.stream)
        return start, done

    def _batch(self, dst, src, size, stream) -> None:
        """One cudaMemcpyBatchAsync (the CUDA 13 8-argument form of dwdp/weight_manager.py) on ``stream``."""
        res = self.cudart.cudaMemcpyBatchAsync(dst, src, size, len(dst), self._attrs, [0], 1, int(stream))
        err = res[0] if isinstance(res, tuple) else res
        if err != self._ok:
            raise RuntimeError(f"cudaMemcpyBatchAsync ({len(dst)} rows) failed: {err}")
        self.calls += 1

    def verify(self, rows: dict, after) -> int:
        """(TLLM_GENEPLB_VERIFY_COPY=1, debug) Rows whose destination bytes differ from the expert's home row. On the
        verify stream after ``after`` (the copies' done event): each home row (on a peer, mostly) is pulled into a
        local scratch row by the copy engine, then compared with the destination row (both local: no torch view of
        peer memory); one host wait at the end."""
        import torch

        if not rows["dst"]:
            return 0
        vs, vraw = (self.vstream, self._vraw) if self.vstream is not None else (self.stream, self._raw)
        with torch.cuda.device(self.device), torch.cuda.stream(vs):
            vs.wait_event(after)
            if self._scratch is None or self._scratch.numel() < max(rows["size"]):
                self._scratch = torch.empty((max(rows["size"]),), dtype=torch.uint8, device=self.device)
            scratch = self._scratch
            bad = torch.zeros((), dtype=torch.int64, device=self.device)
            for d, r, n in zip(rows["dst"], rows["ref"], rows["size"]):
                self._batch([int(scratch.data_ptr())], [int(r)], [int(n)], vraw)  # stream-ordered after the last compare
                bad += (raw_u8(d, n, self.device) != scratch[:n]).any()
            n_bad = int(bad.item())
        return n_bad

    def describe(self) -> str:
        return (f"copy stream 0x{int(self._raw):x} (cudaStreamCreateWithFlags non-blocking, ExternalStream), "
                f"cudaMemcpyBatchAsync <= {MAX_ROWS} rows/call, srcAccessOrder STREAM, PREFER_OVERLAP_WITH_COMPUTE, "
                f"event ring {len(self.ring)}{', verify on' if VERIFY else ''}")
