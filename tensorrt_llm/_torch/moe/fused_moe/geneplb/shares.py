# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB's routing tables and traffic shares (numpy + the standard library: the planner worker imports it standalone).

A layer's layout ``phys`` is one int per physical slot (expert id, or -1 for a FREE slot), rank-major: slot
s = r * local_slots + j. Its route tables are the native MoePlacementInfo layout the route kernels read:
``expert_count[E]`` (copies of e), ``expert_start[E]`` (offset of e's copies in ``slot_ids``) and
``slot_ids[SLOTS]`` (global slot ids grouped by expert, unused entries -1). A replicated expert's tokens split
over its copies by per-slot weights; the weighted rule reads them as ``cumw[SLOTS]``, Q32 cumulative shares in
``slot_ids`` order (uint32 bits in int32 storage, M32 = -1 where unused).

The shares come from the shipped solvers (csolvers): ``solve_rows`` (product C: the rows split, frozen
l3_core.c, mode guided, band 1e-9, 64 Newton steps, no tile alignment) and ``solve_b`` (product B: cost model
B's split, l3b_core.c). Every function returns the shipped bits for the same inputs.
"""

import ctypes
from typing import NamedTuple, Optional, Sequence

import numpy as np

try:
    from . import costmodel, csolvers
except ImportError:  # imported standalone (the planner worker): this directory is on sys.path
    import costmodel
    import csolvers

LAMBDA = 0.95  # the statistics kernel's EMA weight (stats_kernels.cu: F = fl32(fl32(0.95 F) + C))
M32 = (1 << 32) - 1
MODES = {"exact": 0, "guided": 1}
# l3_route.json (the shipped C split): solver mode / band_rel / max_newton, lever alignment, tile_rows
C_MODE, C_BAND_REL, C_MAX_NEWTON = "guided", 1e-09, 64
ALIGN_TILE, TILE_ROWS = False, 256
# l3b_solver.py: the B solver's parameter vector and diagnostics
PAR_NAMES = (
    "cf",
    "cl",
    "cc",
    "ch",
    "tau",
    "tile",
    "margin",
    "max_act_steps",
    "polish_iters",
    "c_mode",
    "c_band_rel",
    "c_max_newton",
    "bisect_tol",
    "lse_top",
    "pricing",
    "disp",
)
PAR_DEFAULTS = dict(
    margin=0.0,
    max_act_steps=6.0,
    polish_iters=64.0,
    bisect_tol=0.05,
    lse_top=8.0,
    pricing=0.0,
    disp=0.0,
)
DIAG_NAMES = (
    "J",
    "J_C_raw",
    "J_cand_C",
    "J_cand_B",
    "J_cand_A",
    "pick",
    "pick_stage",
    "active_copies",
    "replicated_copies",
    "flows_b",
    "flows_c",
    "c_fallback",
    "act_accepted",
    "act_tried",
    "polish_iters",
    "T_b",
    "replicated_experts",
    "J_B_raw",
    "J_A_raw",
    "c_improved",
    "deactivated_copies",
    "b_fallback",
    "reserved22",
    "reserved23",
)
RC = {
    -1: "allocation failed",
    -2: "phys out of range or an expert with rows but no routed slot",
    -3: "bad load or parameters",
}

_CACHE_DIR = None  # configure(); None = csolvers.default_cache_dir()


class RouteTables(NamedTuple):
    """One layer's route tables as int32 numpy arrays (unpacks as (expert_count, expert_start, slot_ids))."""

    expert_count: np.ndarray
    expert_start: np.ndarray
    slot_ids: np.ndarray


def configure(cache_dir=None) -> None:
    """Where the solver libraries are built / found (default: $TRTLLM_GENEPLB_CACHE_DIR or
    ~/.cache/tensorrt_llm/geneplb)."""
    global _CACHE_DIR
    _CACHE_DIR = None if cache_dir is None else str(cache_dir)


def _rows_lib():
    return csolvers.build_rows_solver(_CACHE_DIR)


def _b_lib():
    return csolvers.build_b_solver(_CACHE_DIR)


def _dp(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ip(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# ================================================================================================ tables


def tables_from_phys(phys: Sequence[int], experts: int) -> RouteTables:
    """Route tables of a layout: each expert's copies in ascending slot order (joint_tables.h makeTables(layout,
    original(layout)) = lean_route_device.tables_from_layout). An expert without a copy gets count 0."""
    ph = np.asarray(phys, dtype=np.int64)
    if ph.ndim != 1 or (ph.size and (ph.min() < -1 or ph.max() >= experts)):
        raise ValueError("layout must be one int in -1..%d per slot" % (experts - 1))
    act = np.nonzero(ph >= 0)[0]
    order = act[np.argsort(ph[act], kind="stable")]  # by expert, ascending slot within an expert
    count = np.bincount(ph[act], minlength=experts).astype(np.int32)
    start = np.zeros(experts, dtype=np.int32)
    np.cumsum(count[:-1], out=start[1:])
    slot_ids = np.full(len(ph), -1, dtype=np.int32)
    slot_ids[: len(order)] = order
    return RouteTables(count, start, slot_ids)


def even_weights(phys: Sequence[int], experts: Optional[int] = None) -> np.ndarray:
    """The even split over each expert's copies (FREE slots 0): float64 per slot (eplb_gate.gt1_even)."""
    return costmodel.even_split(phys, experts)


# ================================================================================================ the solvers
def solve_rows(
    load, phys, experts: int, ranks: int, local_slots: int, *, info: Optional[dict] = None
) -> np.ndarray:
    """Product C's per-slot weights of one layer (the rows split; l3_route_solver.weights): float64 [ranks *
    local_slots].

    load[e] = expert e's rows (converted to double), phys[slot] = expert or -1. w[s] = planned rows / N_e, even
    within a rank; 1 / copies for an expert on one rank or without rows. ``info`` (a dict) receives T / improved /
    max_flows / fallback.
    """
    lib = _rows_lib()
    c_load = np.ascontiguousarray(load, dtype=np.float64)
    c_phys = np.ascontiguousarray(phys, dtype=np.int32)
    n = len(c_phys)
    if c_load.shape != (experts,) or n != ranks * local_slots:
        raise ValueError(
            "rows solve: %d loads / %d slots for the %d x %d geometry"
            % (c_load.size, n, ranks, local_slots)
        )
    w = np.zeros(n, dtype=np.float64)
    T, improved, st = ctypes.c_double(), ctypes.c_int(), (ctypes.c_int * 2)()
    rc = lib.l3_weights_full(
        experts,
        ranks,
        local_slots,
        _dp(c_load),
        n,
        _ip(c_phys),
        int(ALIGN_TILE),
        float(TILE_ROWS),
        _dp(w),
        ctypes.byref(T),
        ctypes.byref(improved),
        MODES[C_MODE],
        C_BAND_REL,
        C_MAX_NEWTON,
        st,
    )
    if rc == -2:
        raise ValueError("rows solve: an expert without a routed slot")
    if rc != 0:
        raise MemoryError("l3_weights_full: allocation failed")
    if info is not None:
        info.update(
            T=T.value, improved=bool(improved.value), max_flows=int(st[0]), fallback=int(st[1])
        )
    return w


def par_vector(params) -> np.ndarray:
    """The B solver's double[16] from a full sequence, a solver dict (cf, cl, cc, ch, tau required;
    l3b_solver.par_vector)
    or a cost-model parameter dict (costmodel.prior_params: its tier's settings fill tile / margin / pricing /
    polish)."""
    if isinstance(params, dict):
        if (
            "tile_rows" in params and "T" in params
        ):  # costmodel.prior_params -> the shipped l3b_params(params, settings)
            params = costmodel.l3b_params(params)
        unknown = sorted(set(params) - set(PAR_NAMES))
        if unknown:
            raise ValueError("unknown B solver parameters %s" % unknown)
        d = dict(
            PAR_DEFAULTS,
            tile=float(TILE_ROWS),
            c_mode=float(MODES[C_MODE]),
            c_band_rel=float(C_BAND_REL),
            c_max_newton=float(C_MAX_NEWTON),
        )
        d.update(params)
        missing = [k for k in ("cf", "cl", "cc", "ch", "tau") if k not in d]
        if missing:
            raise ValueError("missing cost-model parameters %s" % missing)
        vals = [float(d[k]) for k in PAR_NAMES]
    else:
        vals = [float(v) for v in params]
        if len(vals) != len(PAR_NAMES):
            raise ValueError(
                "B solver parameters: %d values, %d expected" % (len(vals), len(PAR_NAMES))
            )
    return np.ascontiguousarray(vals, dtype=np.float64)


def solve_b_raw(load_rows, phys, par, local_slots: int):
    """(rc, w, diag) of l3b_weights on the rows ``load_rows`` without raising (tests: error codes included)."""
    lib = _b_lib()
    c_load = np.ascontiguousarray(load_rows, dtype=np.float64)
    c_phys = np.ascontiguousarray(phys, dtype=np.int32)
    c_par = par_vector(par)
    n = len(c_phys)
    if c_load.ndim != 1 or c_phys.ndim != 1 or local_slots <= 0 or n % local_slots:
        raise ValueError(
            "B solve: 1-d load / phys and %d slots for %d slots per rank" % (n, local_slots)
        )
    w = np.zeros(n, dtype=np.float64)
    diag = np.zeros(csolvers.N_DIAG, dtype=np.float64)
    rc = lib.l3b_weights(
        len(c_load),
        n // local_slots,
        local_slots,
        _dp(c_load),
        _ip(c_phys),
        _dp(c_par),
        _dp(w),
        _dp(diag),
    )
    return rc, w, diag


def solve_b(
    load, phys, scale: float, local_slots: int, params, *, info: Optional[dict] = None
) -> Optional[np.ndarray]:
    """Product B's per-slot weights of one layer on the per-replay rows load[e] x scale (l3b_solver.solve_layer).

    params: costmodel.prior_params(T) (its tier's solver settings), costmodel.l3b_params(...) or the full vector.
    Returns float64 [len(phys)] (0 = a deactivated copy), or None when the solver refuses the inputs (rc -2 / -3:
    decided by the inputs alone, every rank refuses alike) -- the caller then takes solve_rows. ``info`` receives the
    solver's diagnostics (DIAG_NAMES; pick 'C' | 'B' | 'A').
    """
    if not (isinstance(scale, float) and scale > 0.0):
        raise ValueError("B solve: a positive float per-replay scale")
    rows = np.asarray(load, dtype=np.float64) * scale  # float(load[e]) * scale, element by element
    rc, w, diag = solve_b_raw(rows, phys, params, local_slots)
    if rc == -1:
        raise MemoryError("l3b_weights: allocation failed")
    if rc != 0:
        if info is not None:
            info.update(rc=rc, error=RC.get(rc, "?"))
        return None
    if info is not None:
        d = {k: float(v) for k, v in zip(DIAG_NAMES, diag) if not k.startswith("reserved")}
        d["pick"] = ("C", "B", "A")[int(d["pick"])]
        d["pick_stage"] = ("raw", "hot", "lse")[int(d["pick_stage"])]
        info.update(d, rc=0)
    return w


# ================================================================================================ cumw / EMA
def _pysum_rows(w: np.ndarray) -> np.ndarray:
    """Per row, CPython 3.12's builtin sum() of the row's floats (Neumaier-compensated, start 0; l3_core.c l3_pysum),
    vectorised over rows; trailing zero padding leaves it unchanged. The shipped conversion sums with sum()."""
    f = 0.0 + w[:, 0]
    c = np.zeros(w.shape[0])
    for k in range(1, w.shape[1]):
        x = w[:, k]
        t = f + x
        c = c + np.where(np.abs(f) >= np.abs(x), (f - t) + x, (x - t) + f)
        f = t
    fix = (c != 0.0) & np.isfinite(c)
    return np.where(fix, f + c, f)


def cumw_from_weights(weights, phys=None, tables=None) -> np.ndarray:
    """The cumw buffer of one layer: int32 [SLOTS] (wroute_ref.cumw_from_weights + l3_route_weights.as_int32).

    weights[s] = the share of physical slot s; tables = (expert_count, expert_start, slot_ids) of the routed layout
    (tables_from_phys(phys) when omitted; when both are given every routed slot must hold its expert in phys).
    For an expert with c > 1 copies: w_k = max(0, weights[slot_ids[start + k]]) in table order, total = their sum
    (equal shares when <= 0), cumw[start + k] = min(max(int(acc_k / total * 2^32 + 0.5), cumw[start + k - 1]), M32)
    for k < c - 1; every other entry M32. Stored as int32 (the uint32 bits). A non-finite weight of a replicated
    expert's copy raises ValueError (the shipped numpy path would store platform-dependent garbage).
    """
    if tables is None:
        if phys is None:
            raise ValueError("cumw: tables or phys required")
        tables = tables_from_phys(phys, int(np.max(phys)) + 1 if len(phys) else 0)
    count, start, slot_ids = (np.asarray(a, dtype=np.int64) for a in tables)
    n_slots = len(slot_ids)
    w_all = np.asarray(weights, dtype=np.float64)
    if phys is not None:
        ph = np.asarray(phys, dtype=np.int64)
        e_of = np.repeat(np.arange(len(count)), count)
        used = slot_ids[: len(e_of)]
        if len(e_of) > n_slots or np.any(used < 0) or np.any(ph[used] != e_of):
            raise ValueError("cumw: the tables do not match the layout")
    out = np.full(n_slots, M32, dtype=np.int64)
    rep = np.nonzero(count > 1)[0]
    if rep.size:
        c = count[rep]
        col = np.arange(int(c.max()))
        valid = col[None, :] < c[:, None]
        pos = np.where(valid, start[rep][:, None] + col[None, :], 0)
        w = np.where(valid, w_all[np.where(valid, slot_ids[pos], 0)], 0.0)
        if not np.isfinite(w).all():
            raise ValueError("cumw: non-finite weight of a replicated expert's copy")
        w = np.maximum(0.0, w)
        total = _pysum_rows(w)
        zero = total <= 0.0
        if zero.any():  # zero total weight -> equal shares
            w = np.where(zero[:, None] & valid, 1.0, w)
            total = np.where(zero, c.astype(np.float64), total)
        acc = np.cumsum(w, axis=1)  # the running acc += w[k]
        q = np.floor(acc / total[:, None] * 4294967296.0 + 0.5)
        q = np.minimum(np.maximum.accumulate(np.maximum(q, 0.0), axis=1), float(M32)).astype(
            np.int64
        )
        wr = col[None, :] < (c[:, None] - 1)
        out[pos[wr]] = q[wr]
    return np.where(out >= (1 << 31), out - (1 << 32), out).astype(np.int32)


def ema_scale(steps: int) -> float:
    """Per-replay expected rows from the EMA after ``steps`` replays since its reset: (1 - 0.95) / (1 - 0.95 **
    steps)."""
    if type(steps) is not int or steps < 1:
        raise ValueError("EMA steps >= 1")
    return (1.0 - LAMBDA) / (1.0 - LAMBDA**steps)
