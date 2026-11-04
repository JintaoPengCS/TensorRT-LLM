# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB's CUDA kernels: JIT builds and CUDA-graph-capturable launch wrappers.

Sources (``csrc/``, the shipped kernels with the geometry as -D macros):
  - ``l3_pad_route_kernels.cu`` entry ``route_pad`` (product C): expert id -> physical slot by the native round-robin
    rule (wmode 0, TRT-LLM's moeComputeRouteKernel bit for bit) or the weighted rule (wmode 1: copy k of expert e takes
    the Q32 share cumw[start_e + k], systematic sampling with a Weyl start phase); the padding rows of a padded replay
    (rows >= spread_limit(valid_rows, rows, rows_ref)) take rule-based experts (j + ep_rank * E / ranks) % E;
  - ``l3b_route_kernels.cu`` entry ``route_b`` (product B): the same slot for every element under the same tables,
    cumw, wmode and padding word, from 256-thread blocks, int32 or int64 output;
  - ``stats_kernels.cu`` entry ``record``: per launch C[l][e] = rows of expert e among the first valid_rows rows,
    F[l][e] = fl32(fl32(0.95 F) + C), steps[l] += 1 (enable / valid_rows read from device words).

One extension per (source, geometry, flags, device SM, torch / CUDA version), compiled for the current device's SM
only (``TORCH_CUDA_ARCH_LIST`` does not apply). ``$TRTLLM_GENEPLB_CACHE_DIR`` (default
``~/.cache/tensorrt_llm/geneplb``, may be shared by many nodes) holds ``ext/<name>/<name>.so``; it is only read or
added to (``csolvers.fetch`` / ``csolvers.publish``): a process imports its node-local copy, and a missing
extension is built node-locally (torch's build lock serializes the ranks of a node) and then published.

Every launch reads fixed device addresses and words only: no host sync, no allocation but the output (through the
caching allocator), so it can be captured in a CUDA graph; the host changes tables, cumw, wmode and valid_rows
between replays on the replay stream.
"""

from __future__ import annotations

import hashlib
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

try:
    from . import csolvers
except ImportError:  # this directory on sys.path
    import csolvers

CSRC = Path(__file__).resolve().parent / "csrc"
SOURCES = {"C": "l3_pad_route_kernels.cu", "B": "l3b_route_kernels.cu", "stats": "stats_kernels.cu"}
PREFIX = {"C": "geneplb_route_c", "B": "geneplb_route_b", "stats": "geneplb_stats"}
VALID_ALL = (
    0x7FFFFFFF  # valid_rows word of a launch without padding rows: count every row, spread none
)
MAX_ROWS = 1 << 16  # rows per launch (the kernels' host cap)
BLOCK = 1024  # route rule block: flat (row, k) elements
MB_TPB = 256  # B kernel threads per block
WEYL = dict(
    WR_PHI_G=0x9E3779B9, WR_PHI_E=0x632BE5AB, WR_GROUP_STRIDE=64
)  # the weighted rule's frozen constants

_LOCK = threading.Lock()
_EXT = {}  # extension name -> module
_STAGE = {}  # (device, top_k) -> int32 staging the C kernel never writes (verify_enable 0)

DeviceLike = Union[torch.device, int, str]


def check_geometry(variant: str, experts: int, slots: int, ranks: int, top_k: int) -> None:
    """The kernels' compile-time geometry assumptions, as a ValueError before a build (variant 'C', 'B' or 'stats')."""
    if not (ranks >= 1 and experts >= 1 and experts % ranks == 0 and slots % ranks == 0):
        raise ValueError(
            f"GenEPLB kernels: ranks {ranks} must divide experts {experts} and slots {slots}"
        )
    if not 1 <= top_k <= experts:
        raise ValueError(f"GenEPLB kernels: top_k {top_k} must be in [1, experts {experts}]")
    if variant == "stats":
        return
    if not experts <= slots:
        raise ValueError(f"GenEPLB route kernels: experts {experts} > slots {slots}")
    if variant == "C" and not (
        slots <= BLOCK
        and 4 * BLOCK + 2 * (BLOCK // 32) * experts + 12 * experts + 8 * slots + 4 <= 48 * 1024
    ):
        raise ValueError(
            f"GenEPLB route kernel C: slots {slots} <= {BLOCK} and the static shared memory "
            f"(76 E + 8 S + 4100 bytes) <= 48 KB required (E {experts}, S {slots})"
        )
    if variant == "B" and slots > 2 * MB_TPB:
        raise ValueError(
            f"GenEPLB route kernel B: slots {slots} > {2 * MB_TPB} (two table words per thread)"
        )


def _device(device: DeviceLike) -> torch.device:
    d = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if d.type != "cuda":
        raise ValueError("GenEPLB kernels need a CUDA device")
    return torch.device("cuda", d.index if d.index is not None else torch.cuda.current_device())


def _arch(device: torch.device):
    """(key, nvcc flags) of the device's SM: explicit -gencode flags make torch skip TORCH_CUDA_ARCH_LIST."""
    major, minor = torch.cuda.get_device_capability(device)
    cc = f"{major}{minor}"
    return "sm" + cc, [f"-gencode=arch=compute_{cc},code=sm_{cc}"]


def load_extension(
    variant: str,
    *,
    experts: int,
    slots: Optional[int],
    ranks: int,
    top_k: int,
    layers: Optional[int] = None,
    device: DeviceLike = None,
    cache_dir=None,
    verbose: bool = False,
):
    """Build (once per key) and load one kernel source for a geometry; checks the module's constants."""
    src = CSRC / SOURCES[variant]
    defines = dict(GENEPLB_E=int(experts), GENEPLB_RANKS=int(ranks), GENEPLB_TOPK=int(top_k))
    if slots is not None:
        defines["GENEPLB_SLOTS"] = int(slots)
    if layers is not None:
        defines["GENEPLB_LAYERS"] = int(layers)
    dev = _device(device if device is not None else torch.cuda.current_device())
    arch_key, arch_flags = _arch(dev)
    # no -std: torch.utils.cpp_extension adds the standard its headers need (C++20 for torch >= 2.14)
    flags = ["-O3"] + [f"-D{k}={v}" for k, v in sorted(defines.items())]
    key = hashlib.sha256(
        src.read_bytes()
        + repr((variant, flags, arch_key, torch.__version__, torch.version.cuda)).encode()
    )
    name = f"{PREFIX[variant]}_{key.hexdigest()[:16]}"
    with _LOCK:
        ext = _EXT.get(name)
        if ext is None:
            root = Path(cache_dir) if cache_dir is not None else csolvers.default_cache_dir()
            shared = root.expanduser().resolve() / "ext" / name / f"{name}.so"
            local = csolvers.local_dir() / "ext" / f"{name}.so"
            if csolvers.fetch(shared, local):
                spec = importlib.util.spec_from_file_location(name, str(local))
                ext = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ext)
            else:
                from torch.utils.cpp_extension import load

                build = csolvers.local_dir() / "build" / name
                build.mkdir(parents=True, exist_ok=True)
                ext = load(
                    name=name,
                    sources=[str(src)],
                    extra_cflags=flags,
                    extra_cuda_cflags=flags + arch_flags,
                    build_directory=str(build),
                    with_cuda=True,
                    verbose=verbose,
                )
                csolvers.publish(build / f"{name}.so", shared)
            want = dict(EXPERTS=experts, RANKS=ranks, TOP_K=top_k)
            if slots is not None:
                want["SLOTS"] = slots
            if variant == "stats":
                if layers is not None:
                    want["LAYERS"] = layers
            else:
                want.update(PAD_K=top_k, PAD_PHASE=experts // ranks, BLOCK=BLOCK, **WEYL)
            for k, v in want.items():
                if getattr(ext, k, None) != v:
                    raise RuntimeError(
                        f"GenEPLB {SOURCES[variant]}: constant {k} = {getattr(ext, k, None)}, expected {v}"
                    )
            _EXT[name] = ext
    return ext


def _staging(device: torch.device, top_k: int) -> torch.Tensor:
    """The C kernel's staging operand: never written (verify_enable is 0), sized for MAX_ROWS rows; one per device."""
    with _LOCK:
        t = _STAGE.get((device, top_k))
        if t is None:
            t = _STAGE[(device, top_k)] = torch.zeros(
                MAX_ROWS * top_k, dtype=torch.int32, device=device
            )
        return t


# ================================================================================================ tables


@dataclass
class LayerTables:
    """One layer's device tables read by the route kernels (fixed addresses: views into stacked buffers are fine).

    expert_count / expert_start int32 [E], slot_ids int32 [SLOTS] (native MoePlacementInfo layout), cumw int32 [SLOTS]
    (Q32 cumulative shares in slot_ids order, uint32 bits; shares.cumw_from_weights), wmode int32 [1] (0 native rule,
    1 weighted). All contiguous on the kernel's device.
    """

    expert_count: torch.Tensor
    expert_start: torch.Tensor
    slot_ids: torch.Tensor
    cumw: torch.Tensor
    wmode: torch.Tensor


# ================================================================================================ kernels
class RouteKernel:
    """Expert ids -> physical slot ids of one EP rank (variant 'C': product C's kernel, 'B': product B's)."""

    def __init__(
        self,
        variant: str,
        experts: int,
        slots: int,
        ranks: int,
        ep_rank: int,
        device: DeviceLike,
        top_k: int = 6,
        *,
        cache_dir=None,
        verbose: bool = False,
    ):
        if variant not in ("C", "B"):
            raise ValueError(f"GenEPLB route kernel variant must be 'C' or 'B', got {variant!r}")
        check_geometry(variant, experts, slots, ranks, top_k)
        if not 0 <= ep_rank < ranks:
            raise ValueError(f"ep_rank {ep_rank} outside [0, {ranks})")
        self.variant, self.experts, self.slots, self.ranks, self.top_k = (
            variant,
            experts,
            slots,
            ranks,
            top_k,
        )
        self.ep_rank = int(ep_rank)
        self.device = _device(device)
        self.ext = load_extension(
            variant,
            experts=experts,
            slots=slots,
            ranks=ranks,
            top_k=top_k,
            device=self.device,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        self._zero = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )  # verify_enable 0 / the B kernel's unused word
        if variant == "C":
            self.path_counters = torch.zeros(
                3, dtype=torch.int32, device=self.device
            )  # {fast, full native, full weighted}
            self._stage = _staging(self.device, top_k)
        self._shapes = ((experts,), (experts,), (slots,), (slots,), (1,))

    def _check(self, ids: torch.Tensor, tables: LayerTables) -> None:
        if (
            ids.dtype != torch.int32
            or ids.dim() != 2
            or ids.shape[1] != self.top_k
            or not ids.is_contiguous()
        ):
            raise ValueError(f"GenEPLB route: ids must be contiguous int32 [rows, {self.top_k}]")
        for t, shape in zip(
            (tables.expert_count, tables.expert_start, tables.slot_ids, tables.cumw, tables.wmode),
            self._shapes,
        ):
            if (
                t.dtype != torch.int32
                or tuple(t.shape) != shape
                or not t.is_contiguous()
                or t.device != ids.device
            ):
                raise ValueError(
                    f"GenEPLB route: table tensors must be contiguous int32 {self._shapes} on {ids.device}"
                )

    def __call__(
        self,
        ids: torch.Tensor,
        tables: LayerTables,
        rows_ref: int,
        valid_rows: Optional[torch.Tensor],
        out_dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Slot ids [rows, top_k] (out_dtype int32 | int64) of ``ids`` (int32 [rows, top_k], this rank's rows).

        rows_ref: rows of the main pass of this forward (layer 0's first launch; >= rows: the MTP layer's re-entries
        have fewer rows and their padding limit scales by rows / rows_ref). valid_rows: int32 [1] device word, the real
        rows of the main pass (>= rows_ref, e.g. VALID_ALL: no padding rows); None = no padding rule. An expert id
        outside [0, experts) maps to the sentinel slot id ``slots``.
        """
        self._check(ids, tables)
        if out_dtype not in (torch.int32, torch.int64):
            raise ValueError("GenEPLB route: out_dtype must be torch.int32 or torch.int64")
        c, s, sl = (
            tables.expert_count.data_ptr(),
            tables.expert_start.data_ptr(),
            tables.slot_ids.data_ptr(),
        )
        if self.variant == "C":
            if valid_rows is None:
                out = self.ext.route(
                    ids,
                    self.ep_rank,
                    c,
                    s,
                    sl,
                    tables.cumw,
                    tables.wmode,
                    self._zero,
                    self._stage,
                    self._stage,
                    self.path_counters,
                )
            else:
                out = self.ext.route_pad(
                    ids,
                    self.ep_rank,
                    c,
                    s,
                    sl,
                    tables.cumw,
                    tables.wmode,
                    self._zero,
                    self._stage,
                    self._stage,
                    self.path_counters,
                    int(rows_ref),
                    valid_rows,
                )
            return out if out_dtype == torch.int32 else out.to(out_dtype)
        out = torch.empty(tuple(ids.shape), dtype=out_dtype, device=ids.device)
        pad = valid_rows is not None
        self.ext.route_b(
            ids,
            self.ep_rank,
            c,
            s,
            sl,
            tables.cumw,
            tables.wmode,
            int(rows_ref) if pad else int(ids.shape[0]),
            valid_rows if pad else self._zero,
            pad,
            out,
        )
        return out

    def stage(
        self,
        ids: torch.Tensor,
        tables: LayerTables,
        rows_ref: int,
        valid_rows: Optional[torch.Tensor],
        idx_out: torch.Tensor,
        w_in: torch.Tensor,
        w_out: torch.Tensor,
    ) -> None:
        """B's route written straight into a MegaMoE bucket's staging (the shipped stage cut): idx_out [bucket,
        top_k] gets the slots of ``ids``' rows and -1 past them, w_out [bucket, >= top_k] (unit column stride)
        gets ``w_in``'s fp32 weights and 0 past them. Arguments otherwise as ``__call__``."""
        if self.variant != "B":
            raise ValueError("GenEPLB route stage is cost model B's")
        self._check(ids, tables)
        pad = valid_rows is not None and ids.shape[0] > 0
        self.ext.route_b_stage(
            ids,
            self.ep_rank,
            tables.expert_count.data_ptr(),
            tables.expert_start.data_ptr(),
            tables.slot_ids.data_ptr(),
            tables.cumw,
            tables.wmode,
            int(rows_ref) if pad else int(ids.shape[0]),
            valid_rows if pad else self._zero,
            pad,
            idx_out,
            w_in,
            w_out,
        )


class StatsKernel:
    """Per-layer expert counts and their EMA over CUDA-graph replays (the shipped statistics kernel).

    Buffers (fixed addresses): counts int32 [L, E] (C of the latest launch), load float32 [L, E] (F), steps int32 [L]
    (launches since reset), status int32 [L + 1] (ST_* flags). Record each layer once per forward (the shipped hook
    skips the MTP layer's re-entries), then read load / steps after a window: per-launch rows = F x ema_scale(steps).
    """

    def __init__(
        self,
        layers: int,
        experts: int,
        ranks: int,
        device: DeviceLike,
        top_k: int = 6,
        *,
        slots: Optional[int] = None,
        cache_dir=None,
        verbose: bool = False,
    ):
        check_geometry("stats", experts, slots if slots is not None else ranks, ranks, top_k)
        self.layers, self.experts, self.ranks, self.top_k = (
            int(layers),
            int(experts),
            int(ranks),
            int(top_k),
        )
        self.device = _device(device)
        self.ext = load_extension(
            "stats",
            experts=experts,
            slots=slots,
            ranks=ranks,
            top_k=top_k,
            layers=layers,
            device=self.device,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        L, E, dev = self.layers, self.experts, self.device
        self.counts = torch.zeros((L, E), dtype=torch.int32, device=dev)
        self.load = torch.zeros((L, E), dtype=torch.float32, device=dev)
        self.steps = torch.zeros((L,), dtype=torch.int32, device=dev)
        self.status = torch.zeros((L + 1,), dtype=torch.int32, device=dev)

    def record(
        self, layer: int, ids: torch.Tensor, valid_rows: torch.Tensor, enable: torch.Tensor
    ) -> None:
        """Count layer ``layer``'s ids (int32 [rows, top_k]) among the first valid_rows rows when enable != 0 (both
        int32 [1] device words), then F = fl32(fl32(0.95 F) + C) and steps += 1. Capturable."""
        self.ext.record(
            int(layer), ids, valid_rows, enable, self.counts, self.load, self.steps, self.status
        )

    def record_fused(
        self,
        ptrs: Sequence[int],
        rows: Sequence[int],
        valid_rows: torch.Tensor,
        enable: torch.Tensor,
    ) -> None:
        """One launch for all L layers (block l counts layer l; stats_device.record_fused): ptrs[l] = data_ptr() of
        layer
        l's int32 [rows[l], top_k] ids, kernel arguments fixed at capture -- keep those tensors alive until this
        launch."""
        if len(ptrs) != self.layers or len(rows) != self.layers:
            raise ValueError(f"GenEPLB fused stats: {self.layers} pointers and row counts required")
        self.ext.record_fused(
            [int(p) for p in ptrs],
            [int(r) for r in rows],
            valid_rows,
            enable,
            self.counts,
            self.load,
            self.steps,
            self.status,
        )

    def reset(self) -> None:
        """Zero C, F, steps and status on the current stream (between replays)."""
        self.ext.reset(self.counts, self.load, self.steps, self.status)
