# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-T stream modes for cost model B's staged MegaMoE layers, chosen per
captured CUDA-graph bucket.

T is the rows per rank of the captured forward (layer 0's route() ids: batch x (1 + draft) of the graph, padding and
attention-DP dummies included; a graph replays only with the same rows on every rank). Two launch orders:
  inline   one stream, today's order: gate -> quantize into the bucket's staging -> route+stage (balancer._stage,
           l3b_mbs_k) -> MegaMoE.
  overlap  run_moe forks the route+stage onto a dedicated side stream right before the quantize (fork event recorded on
           the forward stream, side stream waits it), and the forward stream waits the join event right before the
           MegaMoE launch. The MegaMoE kernel never leaves the forward stream (EP launch lockstep).
stats_side (optional): the fused statistics kernel (route() of the last layer, first invocation) on its own dedicated
stream; its join is consumed UNCONDITIONALLY in the last layer's run_moe before the MegaMoE launch (inline or overlap
bucket), and every layer's id tensors stay referenced until that wait is recorded.
Forks happen only while capturing: eager forwards (warm-up runs, prefill, non-graph decode) run the inline order. Streams come
from cudaStreamCreateWithFlags(cudaStreamNonBlocking) (cuda-python), wrapped as torch.cuda.ExternalStream (torch.cuda.Stream()
hands out pooled streams that can alias the MoeShared / DSV4 aux streams); one (fork, join) event pair per layer and one for
the statistics, all created and recorded once at setup (finalize_model), before any capture. Nothing is allocated on a side
stream (the NaN probe runs on the forward stream before the fork).

Env (read at finalize_model; the launch order is captured into each graph and main never re-captures, so one value per GEN
lifetime -- the control file cannot switch it):
  TLLM_GENEPLB_LEAN          inline (default: OFF) | auto | overlap
                             inline (with STATS_SIDE 0 / 1): no hook installed, today's code path exactly (the default, so
                             a build carrying this code runs the single-stream order unless a mode is asked for);
                             auto: overlap iff INLINE_MAX_T < T <= OVERLAP_MAX_T, inline elsewhere; overlap: every bucket.
  TLLM_GENEPLB_INLINE_MAX_T  127 (default: overlap from T 128; the reference rule is 64, never measured in 64 < T < 128)
  TLLM_GENEPLB_OVERLAP_MAX_T 256 (default; T > 256 runs bucket 512 and was never measured)
  TLLM_GENEPLB_STATS_SIDE    0 (default, the shipped launch on the forward stream) | 1 (in overlap buckets: the reference
                             overlap preset) | all (in every captured bucket)
  TLLM_GENEPLB_LEAN_CHECK    0 (default) | 1: debug only -- after each overlap join the forward stream recomputes the
                             route+stage inline into scratch rows and counts mismatching words (captured: every replay of
                             every overlap layer); read back at eager forwards into the status file. Costs a kernel per layer.
  TLLM_GENEPLB_LEAN_STATUS   path prefix of the per-rank status files <prefix>.rank<r> (JSON: config, fork / join counts,
                             per-bucket captures, check counters); default <TLLM_GENEPLB_CTL_FILE>.lean when the control
                             file is set, else none.
"""

from __future__ import annotations

import functools
import json
import os

import torch

from tensorrt_llm.logger import logger

LEAN_CHOICES = ("auto", "inline", "overlap")
STATS_CHOICES = {"0": "off", "off": "off", "1": "overlap", "on": "overlap", "overlap": "overlap", "all": "all"}
STATS_CODES = {"off": 0, "overlap": 1, "all": 2}
MARK = "GENEPLB-STREAMS"
_STREAM_HANDLES = []  # cuda-python handle objects of the dedicated streams (process lifetime; never destroyed)


def _int_env(env, name: str, default: int) -> int:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: an integer >= 0") from None
    if v < 0:
        raise ValueError(f"{name}={raw!r}: an integer >= 0")
    return v


def config_from_env(env=None) -> dict:
    """The lean config of an environment (os.environ by default); ValueError on a malformed value."""
    env = os.environ if env is None else env
    lean = str(env.get("TLLM_GENEPLB_LEAN", "")).strip().lower() or "inline"  # default OFF: the single-stream order
    if lean not in LEAN_CHOICES:
        raise ValueError(f"TLLM_GENEPLB_LEAN={lean!r}: one of {', '.join(LEAN_CHOICES)}")
    raw_stats = str(env.get("TLLM_GENEPLB_STATS_SIDE", "")).strip().lower() or "0"
    if raw_stats not in STATS_CHOICES:
        raise ValueError(f"TLLM_GENEPLB_STATS_SIDE={raw_stats!r}: one of 0, 1, all")
    raw_check = str(env.get("TLLM_GENEPLB_LEAN_CHECK", "")).strip() or "0"
    if raw_check not in ("0", "1"):
        raise ValueError(f"TLLM_GENEPLB_LEAN_CHECK={raw_check!r}: 0 or 1")
    status = str(env.get("TLLM_GENEPLB_LEAN_STATUS", "")).strip()
    if not status and str(env.get("TLLM_GENEPLB_CTL_FILE", "")).strip():
        status = str(env.get("TLLM_GENEPLB_CTL_FILE")).strip() + ".lean"
    return dict(
        lean=lean,
        inline_max_t=_int_env(env, "TLLM_GENEPLB_INLINE_MAX_T", 127),
        overlap_max_t=_int_env(env, "TLLM_GENEPLB_OVERLAP_MAX_T", 256),
        stats_side=STATS_CHOICES[raw_stats],
        check=raw_check == "1",
        status=status,
    )


def lean_mode(T: int, cfg: dict) -> str:
    """The launch order of a captured forward with T rows per rank: 'inline' or 'overlap'."""
    if cfg["lean"] == "overlap":
        return "overlap"
    if cfg["lean"] == "inline":
        return "inline"
    return "overlap" if cfg["inline_max_t"] < int(T) <= cfg["overlap_max_t"] else "inline"


def stats_side_for(mode: str, cfg: dict) -> bool:
    """Whether the statistics kernel of a bucket in ``mode`` goes to its own stream."""
    return cfg["stats_side"] == "all" or (cfg["stats_side"] == "overlap" and mode == "overlap")


def needs_hooks(cfg: dict) -> bool:
    """False: no bucket can fork (LEAN=inline with STATS_SIDE 0 or 1): today's code path, nothing installed."""
    return cfg["lean"] != "inline" or cfg["stats_side"] == "all"


def cfg_text(cfg: dict) -> str:
    stats = {"off": "0", "overlap": "1", "all": "all"}[cfg["stats_side"]]
    return (
        f"LEAN={cfg['lean']} INLINE_MAX_T={cfg['inline_max_t']} OVERLAP_MAX_T={cfg['overlap_max_t']} "
        f"STATS_SIDE={stats} CHECK={int(cfg['check'])}"
    )


def new_stream(device) -> "torch.cuda.ExternalStream":
    """A dedicated CUDA stream of ``device``: cudaStreamCreateWithFlags(cudaStreamNonBlocking) through cuda-python, default
    priority, wrapped as a torch ExternalStream (never destroyed: the handle lives as long as the process). cuda-python's
    runtime may be another runtime instance than torch's, whose current device would default to 0: it is set explicitly,
    the stream's device is checked (cudaStreamGetDevice when present) and the stream runs a kernel on ``device`` once."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:  # older cuda-python layout
        from cuda import cudart
    ok = cudart.cudaError_t.cudaSuccess
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    dev = torch.device("cuda", idx)
    with torch.cuda.device(dev):
        (err,) = cudart.cudaSetDevice(idx)
        if err != ok:
            raise RuntimeError(f"cuda-python cudaSetDevice({idx}) failed: {err}")
        err, handle = cudart.cudaStreamCreateWithFlags(getattr(cudart, "cudaStreamNonBlocking", 1))
        if err != ok:
            raise RuntimeError(f"cudaStreamCreateWithFlags(cudaStreamNonBlocking) failed on {dev}: {err}")
        _STREAM_HANDLES.append(handle)
        get_device = getattr(cudart, "cudaStreamGetDevice", None)
        if get_device is not None:
            err, sdev = get_device(handle)
            if err == ok and int(sdev) != idx:
                raise RuntimeError(f"the new stream is on device {int(sdev)}, not {idx}")
        stream = torch.cuda.ExternalStream(int(handle), device=dev)
        cur = torch.cuda.current_stream(dev)
        probe = torch.zeros(1, dtype=torch.int32, device=dev)  # allocated on the forward stream
        stream.wait_stream(cur)
        with torch.cuda.stream(stream):
            probe.add_(1)
        cur.wait_stream(stream)
        if int(probe.item()) != 1:
            raise RuntimeError(f"the new stream 0x{int(handle):x} did not run a kernel on {dev}")
    return stream


class Lean:
    """The balancer's per-T stream modes (one per process; collective setup through ``setup``)."""

    def __init__(self, b, cfg: dict):
        self.b, self.cfg = b, cfg
        self.L = int(b.num_layers)
        self.last = self.L - 1
        self.rank = int(b.ep_rank)
        dev = b.device
        self.staged_all = len(b.staged) == self.L
        # statistics side stream only when every layer is a staged MegaMoE layer: its join needs the last layer's run_moe
        self.stats_ok = cfg["stats_side"] != "off" and self.staged_all
        self.side = new_stream(dev) if cfg["lean"] != "inline" else None
        self.stats_stream = new_stream(dev) if self.stats_ok else None
        self.events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(self.L)] if self.side is not None else []
        self.stats_events = (torch.cuda.Event(), torch.cuda.Event()) if self.stats_ok else None
        cur = torch.cuda.current_stream(dev)
        for f, j in self.events + ([self.stats_events] if self.stats_events else []):
            f.record(cur)  # created now (cudaEventCreate at the first record): no driver object inside a capture
            j.record(cur)
        cur.synchronize()
        self.chk = (
            torch.zeros(2, dtype=torch.int32, device=dev) if cfg["check"] and self.side is not None else None
        )  # [mismatching words, checked layer-launches]
        self.chk_host = (0, 0)
        self.capturing, self.mode, self.stats_on = False, "inline", False
        self.stats_pending = None  # (join event, held id tensors) between the statistics fork and its join
        self.cap = None  # the capture in progress: its T, mode and tallies
        self.caps = {}  # T -> the latest capture's record
        self.counts = dict(
            captures=0, overlap_captures=0, forks=0, joins=0, stats_forks=0, stats_joins=0, stats_main=0, stale=0
        )
        self.logged = set()

    # ---------------------------------------------------------------------------------------------- per forward
    def begin_forward(self, rows: int) -> None:
        """route() of layer 0: this forward's T and launch order (inline unless capturing)."""
        if self.stats_pending is not None:  # a forward that failed between the statistics fork and its join
            self.stats_pending = None  # never waited for outside its capture
            self.counts["stale"] += 1
        self.capturing = bool(torch.cuda.is_current_stream_capturing())
        if not self.capturing:
            self.mode, self.stats_on, self.cap = "inline", False, None
            if self.chk is not None:
                self._read_check()
            return
        T = int(rows)
        self.mode = lean_mode(T, self.cfg) if self.side is not None else "inline"
        self.stats_on = self.stats_ok and stats_side_for(self.mode, self.cfg)
        self.cap = dict(T=T, mode=self.mode, stats_side=self.stats_on, layers=0, forks=0, joins=0, stats_forks=0,
                        stats_joins=0)
        self.counts["captures"] += 1
        self.counts["overlap_captures"] += self.mode == "overlap"

    def fork(self, layer, ids, weights, idx_out, w_out):
        """MegaMoE run_moe (backend._route_stager_fork), before its input quantize: None = inline (run_moe runs the stager
        after the quantize, as before); else the join run_moe calls right before the MegaMoE launch instead of the stager."""
        if not self.capturing or self.cap is None or not torch.cuda.is_current_stream_capturing():
            return None  # eager forwards never fork
        b, cap = self.b, self.cap
        cap["layers"] += 1
        last = layer.index == self.last
        if self.mode != "overlap":
            if not last:
                return None
            if self.stats_pending is None:
                self._end_capture()
                return None

            def join_inline():
                b._stage(layer, ids, weights, idx_out, w_out)  # today's order: after the quantize
                self._join_stats()
                self._end_capture()

            return join_inline
        main = torch.cuda.current_stream()
        ev_fork, ev_join = self.events[layer.index]
        ev_fork.record(main)  # after the gate, the stats fork and the staging buffers' last readers on this stream
        self.side.wait_event(ev_fork)
        with torch.cuda.stream(self.side):
            b._stage(layer, ids, weights, idx_out, w_out, nan_probe=False)
            ev_join.record(self.side)
        # the NaN probe allocates temporaries: on the forward stream, beside the forked stage (both only read
        # the gate's weights), before the quantize
        b._stage_probe(weights)
        self.counts["forks"] += 1
        cap["forks"] += 1

        def join():
            torch.cuda.current_stream().wait_event(ev_join)  # the stream that launches MegaMoE next
            self.counts["joins"] += 1
            cap["joins"] += 1
            if self.chk is not None:
                self._shadow(layer, ids, weights, idx_out, w_out)
            if last:
                if self.stats_pending is not None:
                    self._join_stats()
                self._end_capture()

        return join

    def stats_fork(self, launch, held) -> bool:
        """route() of the last layer, capturing: True = the fused statistics launch went to the statistics stream (fork on
        the forward stream after every layer's ids, kernel + join on the side), the ids held until the join; False = the
        caller launches it on the forward stream (as shipped)."""
        if not (self.capturing and self.cap is not None and torch.cuda.is_current_stream_capturing()):
            return False
        if not self.stats_on or self.stats_pending is not None:
            self.counts["stats_main"] += 1
            return False
        main = torch.cuda.current_stream()
        ev_fork, ev_join = self.stats_events
        ev_fork.record(main)
        self.stats_stream.wait_event(ev_fork)
        with torch.cuda.stream(self.stats_stream):
            launch()
            ev_join.record(self.stats_stream)
        self.stats_pending = (ev_join, list(held))
        self.counts["stats_forks"] += 1
        self.cap["stats_forks"] += 1
        return True

    def _join_stats(self) -> None:
        ev_join, _held = self.stats_pending
        torch.cuda.current_stream().wait_event(ev_join)
        self.stats_pending = None  # the wait is recorded: the held id blocks may be reused only after this point
        self.counts["stats_joins"] += 1
        if self.cap is not None:
            self.cap["stats_joins"] += 1

    def _end_capture(self) -> None:
        cap, self.cap = self.cap, None
        if cap is None:
            return
        self.caps[cap["T"]] = cap
        key = (cap["T"], cap["mode"], cap["stats_side"])
        if self.rank == 0 and key not in self.logged:
            self.logged.add(key)
            logger.info(
                f"GenEPLB lean bucket T {cap['T']}: {cap['mode']}"
                f"{' (route+stage forked to the side stream)' if cap['mode'] == 'overlap' else ''}"
                f"{', stats kernel on its own stream' if cap['stats_side'] else ''}; staged layers {cap['layers']}, "
                f"forks {cap['forks']} joins {cap['joins']}, stats forks {cap['stats_forks']} joins {cap['stats_joins']}"
            )
        self.write_status()

    # ---------------------------------------------------------------------------------------------- the debug check
    def _shadow(self, layer, ids, weights, idx_out, w_out) -> None:
        """(LEAN_CHECK) after the join: the inline route+stage into scratch rows on the forward stream, compared word for
        word with the forked one (captured, so every replay checks): chk[0] += mismatches, chk[1] += 1."""
        k = int(idx_out.shape[1])
        s_idx = torch.empty_like(idx_out)
        s_w = torch.empty_like(w_out)
        self.b._stage(layer, ids, weights, s_idx, s_w, nan_probe=False)
        bad = (s_idx != idx_out).sum(dtype=torch.int32) + (
            s_w[:, :k].view(torch.int32) != w_out[:, :k].view(torch.int32)
        ).sum(dtype=torch.int32)
        self.chk[0:1].add_(bad)  # in place on the views (``t[i] += x`` would add a setitem copy)
        self.chk[1:2].add_(1)

    def _read_check(self) -> None:
        bad, n = (int(v) for v in self.chk.tolist())  # one D2H (check mode only)
        if (bad, n) != self.chk_host:
            if bad and not self.chk_host[0]:
                logger.warning(f"GenEPLB lean rank {self.rank}: LEAN_CHECK found {bad} mismatching staging words")
            self.chk_host = (bad, n)
            self.write_status()

    # ---------------------------------------------------------------------------------------------- report
    def summary(self) -> dict:
        return dict(
            armed=True,
            cfg=self.cfg,
            layers=self.L,
            staged=len(self.b.staged),
            stats_side_ok=self.stats_ok,
            counts=dict(self.counts),
            buckets={str(t): c["mode"] for t, c in sorted(self.caps.items())},
            caps={str(t): c for t, c in sorted(self.caps.items())},
            check=None if self.chk is None else dict(bad=self.chk_host[0], layer_launches=self.chk_host[1]),
        )

    def write_status(self) -> None:
        _write_status(self.cfg, self.rank, self.summary())


def _write_status(cfg, rank: int, doc: dict) -> None:
    if not cfg or not cfg.get("status"):
        return
    path = f"{cfg['status']}.rank{rank}"
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(dict(rank=rank, mark=MARK, **doc), sort_keys=True) + "\n")
        os.replace(tmp, path)
    except OSError:
        pass


def setup(b, comm=None):
    """GenEplbMoeLoadBalancer.finalize_model, after _init_pipeline (collective on every rank, before any capture or
    MegaMoE launch): parse the env, check the MegaMoE hooks, make the dedicated streams and events when a bucket can fork,
    agree on every rank (one MPI allgather: the same config on every rank and every rank's setup done, else EVERY rank
    raises the same error -- the MegaMoE barrier has no timeout, so no rank may fail alone later), then install the fork
    hook on every staged MegaMoE backend. Returns the Lean, or None (today's code path: nothing installed)."""
    cfg, err, lean, reason = None, "", None, ""
    try:
        cfg = config_from_env()
    except ValueError as e:
        err = f"bad env: {e}"
    want = cfg is not None and needs_hooks(cfg)
    if want and (b.mode != "B" or not b.staged):
        want, reason = False, "inert: needs cost model B's staged MegaMoE layers"
    if want:
        missing = [li.index for li in b.layers if li.index in b.staged
                   and not hasattr(getattr(li.moe, "backend", None), "_route_stager_fork")]
        if missing:
            err = (f"the MegaMoE backend of layers {missing[:4]} has no _route_stager_fork "
                   "(mega_moe_cute_dsl.py lacks the GENEPLB-STREAMS patch)")
        else:
            try:
                lean = Lean(b, cfg)
            except Exception as e:  # every rank raises below, together
                err = f"stream / event setup failed: {e!r}"
    codes = (
        (cfg["lean"], cfg["inline_max_t"], cfg["overlap_max_t"], cfg["stats_side"], cfg["check"], lean is not None)
        if cfg is not None
        else None
    )
    own = comm is None
    if own:
        from tensorrt_llm._utils import mpi_comm

        comm = mpi_comm().Dup()
    try:
        every = comm.allgather((int(b.ep_rank), not err, codes, err))
    finally:
        if own:
            comm.Free()
    failed = [(r, e) for r, ok, _, e in every if not ok]
    if failed:
        raise RuntimeError(
            f"GenEPLB lean ({MARK}) setup failed on {len(failed)} rank(s), every rank stops before any MegaMoE launch: "
            + "; ".join(f"rank {r}: {e}" for r, e in failed[:4])
        )
    if len({c for _, _, c, _ in every}) != 1:
        raise RuntimeError(
            f"GenEPLB lean ({MARK}): the ranks' TLLM_GENEPLB_LEAN* settings differ: "
            + "; ".join(f"rank {r}: {c}" for r, _, c, _ in every[:8])
        )
    if lean is not None:
        for layer in b.layers:
            if layer.index in b.staged:
                layer.moe.backend._route_stager_fork = functools.partial(lean.fork, layer)
        lean.write_status()
    else:
        _write_status(cfg, int(b.ep_rank), dict(armed=False, cfg=cfg, reason=reason or "no hook needed"))
    if int(b.ep_rank) == 0:
        if lean is None:
            logger.info(
                f"GenEPLB lean: {cfg_text(cfg)}: no hook installed, today's single-stream order"
                f"{' (' + reason + ')' if reason else ''}"
            )
        else:
            logger.info(
                f"GenEPLB lean: {cfg_text(cfg)}; fork hooks on {len(b.staged)} staged layers, side stream "
                f"{'0x%x' % lean.side.cuda_stream if lean.side is not None else 'none'}, stats stream "
                f"{'0x%x' % lean.stats_stream.cuda_stream if lean.stats_stream is not None else 'none'}"
                f"{'' if lean.stats_ok or cfg['stats_side'] == 'off' else ' (stats side inert: not every layer is staged)'}"
            )
    return lean
