# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB's time models: cost model B (settings, prior parameters, objective) and the install gate's forms.

numpy and the standard library only (no torch, no tensorrt_llm): the planner worker imports it standalone.
Every function equals the shipped one on the shipped geometry (costmodel_b.py, eplb_gate.py gt1_*): the same
operations in the same order, so the same floats; the geometry (ranks, slots per rank) is an argument here.

Cost model B: rank r's MoE time c_r = cf F + cl Lt + cc R + ch H (F active slots, Lt tiles after the first,
R rows, H = tiles of the hottest slot - 1; tile = tile_rows), a layer's time = tau log sum_r exp(c_r / tau),
the max for tau <= tau_hard. The parameters are the prior's row nearest to the decode tier T = tokens per rank
per step (local batch x (1 + MTP draft)) by log2 distance. Form T (the rows model C's gate uses): c_r =
slope_us sum_s ceil(x_s / m) m + beta_us sum_s x_s, the busiest rank's c_r.
"""

import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

PRIOR_FILE = Path(__file__).resolve().parent / "data" / "costmodel_b_prior.json"
PRIOR = "configs/costmodel_b_prior.json"  # the shipped label of the prior (part of the parameters' sha256)
PARAMS_SCHEMA = "geneplb_costmodel_b_params_v1"
TAU_HARD = 0.06  # tau <= this: the hard max (l3b_core.c TAU_MAX_RULE)
TILE_ROWS = 256  # the token tile of the static default MegaMoE tactic
MTP_DRAFT = 3  # the shipped decode: MTP with 3 draft tokens (T = local batch x 4)
LAYERS_PER_CALL = 16
INLINE_MAX_T = 64  # the shipped launch order: inline for T <= 64, overlap above
POLISH_ITERS = 64
REFRESH = dict(period=16, window=8, delay=2, mode="piggyback", min_replays=4)
PRESET = {
    32: dict(
        pricing="poisson", disp=8.0, margin=3.0
    ),  # T -> the fast level's pricing (measured tiers 32 / 64 / 128)
    64: dict(pricing="mean", margin=1.5),
    128: dict(pricing="poisson", disp=4.0, margin=1.5),
}
# The shipped gate model of C (eplb_policy_defaults.json q9.gate.model: form T, the microbenchmark fit).
GATE_MODEL_T = dict(form="T", m=256, slope_us=0.0125, beta_us=0.009, tau_w_us=0.0)
T_FORMS = (
    "L",
    "T",
    "RT",
    "ET",
)  # the gate's rows-model forms (eplb_gate GT1_FORMS); every other dict with cf.. is B
B_KEYS = ("cf", "cl", "cc", "ch", "tau", "tau_hard", "tile_rows")


# ================================================================================================ settings / prior


def tier_of(T: int) -> int:
    """The preset row of T: the largest measured tier <= T, the smallest one below it (shipped settings())."""
    return max([t for t in PRESET if t <= T] or [min(PRESET)])


def extrapolated(T: int) -> bool:
    """T below the smallest measured tier (32): its preset and prior are the T = 32 rows (flagged, never measured)."""
    return T < min(PRESET)


def settings_for_T(T: int, local_batch: Optional[int] = None) -> dict:
    """B's resolved settings for T tokens per rank per step (shipped costmodel_b.settings without the q9 block).

    T < 32 (e.g. no MTP, or local batch < 8) takes the T = 32 preset and prior row and is flagged ``extrapolated``.
    """
    T = int(T)
    tier = tier_of(T)
    out = dict(
        source="prior",
        T=T,
        tier=tier,
        mode="inline" if T <= INLINE_MAX_T else "overlap",
        layers_per_call=LAYERS_PER_CALL,
        polish_iters=POLISH_ITERS,
        presolve=True,
        refresh=dict(REFRESH),
        prior=PRIOR,
        prior_key=prior_key(T),
        extrapolated=extrapolated(T),
        **PRESET[tier],
    )
    if local_batch is not None:
        out["local_batch"] = int(local_batch)
    return out


def canonical(value):
    """``value`` with every float as float.hex (dict keys sorted by json): identical bytes iff identical floats."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def params_sha(params: dict) -> str:
    """sha256 of the canonical JSON of every key but sha256."""
    body = {k: v for k, v in params.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(canonical(body), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sealed(params: dict) -> dict:
    out = dict(params)
    out["sha256"] = params_sha(out)
    return out


@functools.lru_cache(maxsize=1)
def _prior_raw() -> bytes:
    return PRIOR_FILE.read_bytes()


def prior_key(T: int) -> int:
    """The prior row of T: T itself when present, else the nearest key by log2 distance (the smaller on a tie)."""
    keys = sorted(int(k) for k in json.loads(_prior_raw()))
    return (
        T if T in keys else min(keys, key=lambda k: (abs(math.log2(k) - math.log2(max(1, T))), k))
    )


def prior_params(T: int, **identity) -> dict:
    """The prior's row for T tokens per rank as a sealed params dict (shipped costmodel_b.prior_params, the same
    sha256).

    ``identity`` keys (e.g. local_batch, layers) are recorded in the dict, ``tile_rows`` overrides the tile. For
    T < 32 the dict also carries ``extrapolated: True`` (never shipped: the T = 32 row).
    """
    raw = _prior_raw()
    table = json.loads(raw)
    key = prior_key(T)
    e = table[str(key)]
    out = dict(
        schema=PARAMS_SCHEMA,
        source="prior",
        prior=PRIOR,
        prior_sha256=hashlib.sha256(raw).hexdigest(),
        prior_key=key,
        model=e["model"],
        form="hot",
        cf=float(e["cf"]),
        cl=float(e["cl"]),
        cc=float(e["cc"]),
        ch=float(e["ch"]),
        tau=float(e["tau"]),
        tau_hard=TAU_HARD,
        a=float(e["a"]),
        rmse_us=None,
        cv_rmse_us=None,
        n=int(e["n"]),
        tile_rows=int(identity.get("tile_rows") or TILE_ROWS),
        T=int(T),
    )
    out.update({k: v for k, v in identity.items() if k != "tile_rows"})
    if extrapolated(int(T)):
        out["extrapolated"] = True
    return sealed(out)


def l3b_params(params: dict, s: Optional[dict] = None) -> dict:
    """The B share solver's parameters: the model, its tile, the tier's margin / pricing and the polish budget.

    ``s`` = settings_for_T(params['T']) when omitted (the shipped costmodel_b.l3b_params(params, settings)).
    """
    s = settings_for_T(int(params["T"])) if s is None else s
    out = dict(
        cf=float(params["cf"]),
        cl=float(params["cl"]),
        cc=float(params["cc"]),
        ch=float(params["ch"]),
        tau=float(params["tau"]),
        tile=float(params["tile_rows"]),
        margin=float(s["margin"]),
        polish_iters=float(s["polish_iters"]),
    )
    if s["pricing"] == "poisson":
        out.update(pricing=1.0, disp=float(s["disp"]))
    return out


def is_b(model: dict) -> bool:
    """Form B: ``form == 'B'``, or B's parameters without a rows-model form (prior_params' form 'hot')."""
    form = model.get("form")
    return form == "B" or (form not in T_FORMS and all(k in model for k in B_KEYS))


# ================================================================================================ cost model B
def rank_costs_B(x, params: dict, ranks: int) -> list:
    """B's us per rank of the slot rows x (one float per slot, rank-major; shipped costmodel_b.rank_costs)."""
    t, cf, cl, cc, ch = params["tile_rows"], params["cf"], params["cl"], params["cc"], params["ch"]
    local = len(x) // ranks
    out = []
    for r in range(ranks):
        F = Lt = hot = 0
        R = 0.0
        for n in x[r * local : (r + 1) * local]:
            if n > 0:
                q = math.ceil(n / t)
                F += 1
                Lt += q - 1
                R += n
                hot = max(hot, q)
        out.append(cf * F + cl * Lt + cc * R + ch * max(hot - 1, 0))
    return out


def lse_B(costs, params: dict) -> float:
    """tau log sum_r exp(c_r / tau) over the ranks' costs, or their max for tau <= tau_hard."""
    m = max(costs)
    tau = params["tau"]
    if tau <= params["tau_hard"]:
        return m
    return m + tau * math.log(sum(math.exp((c - m) / tau) for c in costs))


def gate_us_B(params: dict, x, ranks: int) -> np.ndarray:
    """(L,) B's us of each layer from its slot rows x (L, SLOTS) (shipped costmodel_b.gate_us, the gate's form B)."""
    x = np.asarray(x, dtype=np.float64)
    L = x.shape[0]
    t = float(params["tile_rows"])
    q, tmp, pos = np.ceil(x / t), np.zeros(x.shape), x > 0
    np.subtract(q, 1.0, out=tmp)
    np.maximum(tmp, 0.0, out=tmp)
    F = pos.reshape(L, ranks, -1).sum(2)
    Lt = tmp.reshape(L, ranks, -1).sum(2)
    R = x.reshape(L, ranks, -1).sum(2)
    H = np.maximum(q.reshape(L, ranks, -1).max(2) - 1.0, 0.0)
    c = params["cf"] * F + params["cl"] * Lt + params["cc"] * R + params["ch"] * H
    m = c.max(1)
    if params["tau"] <= params["tau_hard"]:
        return m
    return m + params["tau"] * np.log(np.exp((c - m[:, None]) / params["tau"]).sum(1))


# ================================================================================================ the gate's layer time
def even_split(phys, experts: Optional[int] = None) -> np.ndarray:
    """The even split over each expert's copies, FREE slots 0 (shipped eplb_gate.gt1_even)."""
    ph = np.asarray(phys)
    act = ph >= 0
    k = np.bincount(ph[act], minlength=int(experts or 0))
    w = np.zeros(len(ph))
    w[act] = 1.0 / k[ph[act]]
    return w


def slot_rows(cbar, phys, w) -> np.ndarray:
    """Slot rows of one layer: x_s = cbar[phys[s]] w[s], FREE slots 0.0 (the products of eplb_gate.gt1_layer_us)."""
    ph = np.asarray(phys)
    act = ph >= 0
    x = np.zeros(len(ph))
    x[act] = np.asarray(cbar, dtype=np.float64)[ph[act]] * np.asarray(w, dtype=np.float64)[act]
    return x


def slot_rows_2d(cbar, phys, w) -> np.ndarray:
    """(L, SLOTS) slot rows of L layers, the same elementwise products (eplb_gate.gt1_slot_rows)."""
    ph = np.asarray(phys)
    cb = np.asarray(cbar, dtype=np.float64)
    act = ph >= 0
    gath = np.take_along_axis(cb, np.maximum(ph, 0), axis=1)
    x = np.zeros(ph.shape)
    np.multiply(gath, np.asarray(w, dtype=np.float64), out=x, where=act)
    return x


def _rank_cost_T(model: dict, x3) -> np.ndarray:
    """Per-rank us of a rows-model form on slot rows (..., ranks, local) (eplb_gate.gt1_rank_cost)."""
    f, t, tw, mm = model["form"], model["slope_us"], model["tau_w_us"], model["m"]
    if f == "L":
        c = t * x3.sum(-1)
    else:
        kp = np.ceil(x3 / mm) * mm
        if f == "T":
            c = t * kp.sum(-1)
        elif f == "RT":
            c = np.maximum((x3 > 0).sum(-1) * tw, t * kp.sum(-1))
        else:
            c = np.where(x3 > 0, np.maximum(tw, t * kp), 0.0).sum(-1)
    if model["beta_us"]:
        c = c + model["beta_us"] * x3.sum(-1)
    return c


def layer_us_T(model: dict, cbar, phys, w, ranks: int) -> float:
    """The busiest rank's us of one layer under a rows-model form (eplb_gate.gt1_layer_us, forms L / T / RT / ET)."""
    x = slot_rows(cbar, phys, w).reshape(ranks, -1)
    return float(_rank_cost_T(model, x).max())


def layer_us(model: dict, cbar, phys, w, ranks: int) -> float:
    """The gate's us of one layer: form B (``is_b``) or a rows-model form (eplb_gate.gt1_layer_us)."""
    if is_b(model):
        return float(gate_us_B(model, slot_rows(cbar, phys, w).reshape(1, -1), ranks)[0])
    return layer_us_T(model, cbar, phys, w, ranks)
