# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB's time models: cost model B (settings, prior parameters, objective) and the install gate's forms.

numpy and the standard library only (no torch, no tensorrt_llm): the planner worker imports it standalone.
Every function equals the reference implementation's on its geometry: the same operations in the same order, so
the same floats; the geometry (ranks, slots per rank) is an argument here.

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
PRIOR = "configs/costmodel_b_prior.json"  # the prior's label (part of the parameters' sha256)
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
# The gate model of C (form T, the microbenchmark fit).
GATE_MODEL_T = dict(form="T", m=256, slope_us=0.0125, beta_us=0.009, tau_w_us=0.0)
T_FORMS = (
    "L",
    "T",
    "RT",
    "ET",
)  # the gate's rows-model forms; every other dict with cf.. is B
B_KEYS = ("cf", "cl", "cc", "ch", "tau", "tau_hard", "tile_rows")

# ================================================================================================ P5: prior v2
# Calibrated rows per EP (the slot geometry is part of the key: EP16 = 28 slots / rank, EP32 = 16) and per T tier, with ONE
# tier rule for the coefficients and the pricing (the largest tier <= T; below the smallest, the smallest, flagged
# extrapolated), plus the activity model of the eF objective (placement_ef.py).  Used only with TLLM_GENEPLB_OBJECTIVE=ef.
PRIOR_V2_FILE = Path(__file__).resolve().parent / "data" / "costmodel_b_prior_v2.json"
PARAMS_SCHEMA_EF = "geneplb_costmodel_ef_params_v1"
EF_DEFAULTS = dict(
    act_k=8.0,  # NB shape of a copy's rows per replay
    act_lam0=0.1,  # rows / replay added to every expert's window rate (regression to the mean; zero-load homes)
    tile_disp=4.0,  # normal variance / mean of a copy's rows (later-tile pricing, as the shipped poisson presets)
    gate_margin_us=0.5,  # a split / plan must beat the identity (home-only) routing by this (us per layer-step)
    per_rank=2.0,  # placement flex: options (replicas) per home rank
    keep_bonus=0.25,  # placement flex: hysteresis of a held replica's expert
    place_min_gain_us=0.05,  # placement greedy: the smallest move kept
    new_penalty_us=0.5,  # placement greedy: a new replica must beat a held one by this
)


@functools.lru_cache(maxsize=1)
def _prior_v2_raw() -> bytes:
    return PRIOR_V2_FILE.read_bytes()


def tier_v2(T: int, keys) -> int:
    """ONE tier rule for the coefficients AND the pricing: the largest measured tier <= T, else the smallest."""
    return max([k for k in keys if k <= T] or [min(keys)])


def params_ef(T: int, ranks: int, local_slots: int, kernel: Optional[dict] = None) -> dict:
    """The eF model of T tokens per rank (the balancer's T) at EP ``ranks`` with ``local_slots`` slots per rank: the prior v2
    row (sealed params dict).  Raises when the file has no row set for this EP / slot geometry (never a silent EP32 row at
    EP16, the shipped prior's flaw).  ``kernel``: a warm-up cost table row (b_params_for(..., 'warmup')) whose
    kernel terms cf / cl / cc / ch / a and tile replace the row's; act_k / act_lam0 / tile_disp / the placement knobs stay prior
    v2's (a kernel micro-benchmark cannot observe the routing's per-replay activity).  None: the prior v2 row, unchanged."""
    raw = _prior_v2_raw()
    table = json.loads(raw)
    sec = table.get("ep%d" % int(ranks))
    if sec is None:
        raise ValueError("costmodel_b_prior_v2.json: no rows for EP%d" % int(ranks))
    if int(sec["local_slots"]) != int(local_slots):
        raise ValueError(
            "costmodel_b_prior_v2.json: EP%d rows are for %s slots per rank, the layer has %d"
            % (int(ranks), sec["local_slots"], int(local_slots))
        )
    keys = sorted(int(k) for k in sec["rows"])
    tier = tier_v2(int(T), keys)
    e = sec["rows"][str(tier)]
    out = dict(
        schema=PARAMS_SCHEMA_EF,
        source="prior_v2",
        prior_sha256=hashlib.sha256(raw).hexdigest(),
        form="eF",
        ranks=int(ranks),
        local_slots=int(local_slots),
        T=int(T),
        tier=int(tier),
        tile_rows=TILE_ROWS,
        rmse_us=e.get("rmse_us"),
        extrapolated=int(T) < keys[0],
    )
    out.update({k: float(e[k]) for k in ("cf", "cl", "cc", "ch", "a")})
    dflt = table.get("defaults", {})
    out.update({k: float(e.get(k, sec.get(k, dflt.get(k, v)))) for k, v in EF_DEFAULTS.items()})
    if kernel is not None:  # the warm-up row's kernel terms over the prior v2 row
        out.update({k: float(kernel[k]) for k in ("cf", "cl", "cc", "ch", "a")})
        out.update(
            tile_rows=int(kernel["tile_rows"]),
            source="prior_v2+" + str(kernel.get("source", "warmup")),
            kernel_sha256=str(kernel.get("row_sha256") or kernel.get("sha256")),
            kernel_tier=int(kernel.get("tier", kernel["T"])),
        )
    return sealed(out)


def l3b_params_ef(p: dict, tau: float) -> dict:
    """The B solver's parameters for pricing 2 (l3b_core.c, P5): the calibrated row, the layer's tau (> TAU_HARD, so the LSE
    polish runs) and act_var = cf^2 / (2 tau) (the Gaussian bound's variance term)."""
    tau = max(float(tau), 2.0 * TAU_HARD)
    return dict(
        cf=float(p["cf"]),
        cl=float(p["cl"]),
        cc=float(p["cc"]),
        ch=float(p["ch"]),
        tau=tau,
        tile=float(p["tile_rows"]),
        margin=1.5,
        polish_iters=float(POLISH_ITERS),
        pricing=2.0,
        disp=float(p["tile_disp"]),
        act_k=float(p["act_k"]),
        act_var=float(p["cf"]) ** 2 / (2.0 * tau),
    )


# ================================================================================================ settings / prior


def tier_of(T: int) -> int:
    """The preset row of T: the largest measured tier <= T, the smallest one below it."""
    return max([t for t in PRESET if t <= T] or [min(PRESET)])


def extrapolated(T: int) -> bool:
    """T below the smallest measured tier (32): its preset and prior are the T = 32 rows (flagged, never measured)."""
    return T < min(PRESET)


def settings_for_T(T: int, local_batch: Optional[int] = None) -> dict:
    """B's resolved settings for T tokens per rank per step.

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
    """The prior's row for T tokens per rank as a sealed params dict (with its sha256).

    ``identity`` keys (e.g. local_batch, layers) are recorded in the dict, ``tile_rows`` overrides the tile. For
    T < 32 the dict also carries ``extrapolated: True`` (never measured: the T = 32 row).
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

    ``s`` = settings_for_T(params['T']) when omitted.
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
    """B's us per rank of the slot rows x (one float per slot, rank-major)."""
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
    """(L,) B's us of each layer from its slot rows x (L, SLOTS) (the gate's form B)."""
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
    """The even split over each expert's copies, FREE slots 0."""
    ph = np.asarray(phys)
    act = ph >= 0
    k = np.bincount(ph[act], minlength=int(experts or 0))
    w = np.zeros(len(ph))
    w[act] = 1.0 / k[ph[act]]
    return w


def slot_rows(cbar, phys, w) -> np.ndarray:
    """Slot rows of one layer: x_s = cbar[phys[s]] w[s], FREE slots 0.0 (the products the gate's layer time sums)."""
    ph = np.asarray(phys)
    act = ph >= 0
    x = np.zeros(len(ph))
    x[act] = np.asarray(cbar, dtype=np.float64)[ph[act]] * np.asarray(w, dtype=np.float64)[act]
    return x


def slot_rows_2d(cbar, phys, w) -> np.ndarray:
    """(L, SLOTS) slot rows of L layers, the same elementwise products."""
    ph = np.asarray(phys)
    cb = np.asarray(cbar, dtype=np.float64)
    act = ph >= 0
    gath = np.take_along_axis(cb, np.maximum(ph, 0), axis=1)
    x = np.zeros(ph.shape)
    np.multiply(gath, np.asarray(w, dtype=np.float64), out=x, where=act)
    return x


def _rank_cost_T(model: dict, x3) -> np.ndarray:
    """Per-rank us of a rows-model form on slot rows (..., ranks, local)."""
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
    """The busiest rank's us of one layer under a rows-model form (forms L / T / RT / ET)."""
    x = slot_rows(cbar, phys, w).reshape(ranks, -1)
    return float(_rank_cost_T(model, x).max())


def layer_us(model: dict, cbar, phys, w, ranks: int) -> float:
    """The gate's us of one layer: form B (``is_b``) or a rows-model form."""
    if is_b(model):
        return float(gate_us_B(model, slot_rows(cbar, phys, w).reshape(1, -1), ranks)[0])
    return layer_us_T(model, cbar, phys, w, ranks)


# ================================================================================================ cost table
# The warm-up micro-benchmark's sealed cost table (costmb.py via balancer.before_capture: rank 0 fits, broadcasts the pickled
# table, every rank re-seals it, the sha is voted) and every renewal's B parameters (balancer._b_params, control field params).
# One sealed row per T tier: the fitted kernel terms of the tier's MegaMoE launch bucket, or the tier's sealed fallback row (prior
# v2's kernel terms of the geometry, else the v1 prior's, flagged geometry_mismatch). ONE tier rule, P5's (tier_v2), applied within
# T's launch bucket: the largest tier <= T, below the bucket's smallest tier the smallest; a T whose bucket holds no tier takes its
# fallback row. A row has every key cost model B prices with (B_KEYS, T); b_params_for adds T (the balancer's), kernel (P5's
# overlay trigger) and row_sha256 (the renewal's plan digest).
TABLE_SCHEMA = "geneplb_costmodel_b_table_v1"
FALLBACK_TAU = 0.5  # a fallback row's LSE tau (the v1 prior rows' tau; prior v2 rows carry none; B polishes LSE for tau > 0.06)
PARAMS_SOURCES = ("prior", "prior_v2", "warmup")
_TABLE_KEYS = ("schema", "tiers", "rows", "outcome", "sha256")


def bucket_of(T: int, buckets) -> Optional[int]:
    """The MegaMoE launch bucket of T rows (mega_moe_cute_dsl._select_launch_max_tokens: the smallest rung >= T; the top one above
    it), None without a ladder."""
    b = sorted(int(x) for x in buckets or ())
    if not b:
        return None
    return next((x for x in b if int(T) <= x), b[-1])


def prior_files_sha() -> dict:
    """sha256 of the two prior files a fallback row reads (the micro-benchmark's identity vote)."""
    return dict(
        prior_sha256=hashlib.sha256(_prior_raw()).hexdigest(),
        prior_v2_sha256=hashlib.sha256(_prior_v2_raw()).hexdigest(),
    )


def fallback_row(T: int, ranks: int, local_slots: int, bucket=None, tile_rows: int = TILE_ROWS) -> dict:
    """The sealed fallback row of tier T at EP ``ranks`` x ``local_slots``: prior v2's kernel terms (cf, cl, cc, ch, a) of the
    geometry's row of T (tier_v2) with tau FALLBACK_TAU; without prior v2 rows for the geometry, the v1 prior's row of T flagged
    geometry_mismatch. A function of the files and the arguments only (rank-identical)."""
    raw = _prior_v2_raw()
    sec = json.loads(raw).get("ep%d" % int(ranks))
    if sec is not None and int(sec.get("local_slots", -1)) == int(local_slots):
        keys = sorted(int(k) for k in sec["rows"])
        tier = tier_v2(int(T), keys)
        e = sec["rows"][str(tier)]
        out = dict(
            schema=PARAMS_SCHEMA, source="fallback", kernel="prior_v2", model="prior_v2", form="hot",
            cf=float(e["cf"]), cl=float(e["cl"]), cc=float(e["cc"]), ch=float(e["ch"]), a=float(e["a"]), tau=FALLBACK_TAU,
            v2_tier=int(tier), prior_v2_sha256=hashlib.sha256(raw).hexdigest(),
        )
    else:
        p = prior_params(int(T))
        out = dict(
            schema=PARAMS_SCHEMA, source="fallback", kernel="prior", model=p["model"], form="hot", cf=p["cf"], cl=p["cl"],
            cc=p["cc"], ch=p["ch"], a=p["a"], tau=p["tau"], prior_key=p["prior_key"], prior_sha256=p["prior_sha256"],
            geometry_mismatch=True,
        )
    out.update(
        tau_hard=TAU_HARD, tile_rows=int(tile_rows), T=int(T), tier=int(T), bucket=None if bucket is None else int(bucket),
        ranks=int(ranks), slots_per_rank=int(local_slots),
    )
    return sealed(out)


def make_table(rows: dict, outcome: dict, **meta) -> dict:
    """A sealed cost table: rows {T: sealed row}, outcome {T: 'warmup' | 'fallback'} and the meta (geometry, ladder, protocol)."""
    keys = sorted(int(k) for k in rows)
    rows = {int(k): v for k, v in rows.items()}
    outcome = {int(k): v for k, v in outcome.items()}
    t = dict(
        schema=TABLE_SCHEMA, tiers=keys, rows={str(k): rows[k] for k in keys}, outcome={str(k): str(outcome[k]) for k in keys}
    )
    t.update(meta)
    return sealed(t)


def fallback_table(ranks: int, local_slots: int, tiers=(), buckets=(), tile_rows: int = TILE_ROWS, **meta) -> dict:
    """Every tier's fallback row: the table EVERY rank takes when the micro-benchmark fails anywhere."""
    buckets = sorted(int(b) for b in buckets or ())
    rows = {int(T): fallback_row(T, ranks, local_slots, bucket_of(T, buckets), tile_rows) for T in tiers}
    return make_table(
        rows, {T: "fallback" for T in rows}, ranks=int(ranks), local_slots=int(local_slots), buckets=buckets,
        tile_rows=int(tile_rows), **meta
    )


def table_meta(table: dict) -> dict:
    return {k: v for k, v in table.items() if k not in _TABLE_KEYS}


def fallback_like(table: dict) -> dict:
    """The fallback table of a table's tiers and meta (its sha is what a forced fallback of the same run produces)."""
    m = table_meta(table)
    ranks, slots, buckets = m.pop("ranks"), m.pop("local_slots"), m.pop("buckets")
    return fallback_table(ranks, slots, table.get("tiers") or (), buckets, m.pop("tile_rows", TILE_ROWS), **m)


def table_ok(table: dict) -> bool:
    """The table and every row hash to their recorded sha256 (a table that travelled through a pickle)."""
    rows = table.get("rows") or {}
    return params_sha(table) == table.get("sha256") and all(params_sha(r) == r.get("sha256") for r in rows.values())


def table_row(table: dict, T: int) -> Optional[dict]:
    """The table's row of T: tier_v2 over the tiers of T's launch bucket; None when no tier of the table is in that bucket."""
    rows = table.get("rows") or {}
    b = bucket_of(T, table.get("buckets"))
    keys = sorted(int(k) for k, r in rows.items() if r.get("bucket") == b)
    return rows[str(tier_v2(int(T), keys))] if keys else None


def b_params_for(source: str, T: int, table: Optional[dict], ranks: int, local_slots: int) -> dict:
    """B's parameters of a renewal at T (the balancer's T) from ``source``:
      prior     prior_params(T) itself: the shipped b_params (P5 then prices with the prior v2 row)
      prior_v2  prior v2's kernel terms of the geometry as a B row (fallback_row(T)); P5: the prior v2 row
      warmup    the table's row of T (table_row; a T whose bucket holds no tier: its fallback row); P5: the row's kernel terms
                over the prior v2 row (params_ef(..., kernel=), planner._ef_params)
    The result is sealed and carries row_sha256 (the renewal's plan digest) and kernel (P5's overlay trigger: 'warmup')."""
    T = max(1, int(T))
    if source == "prior":
        return prior_params(T)
    if source == "prior_v2":
        row = fallback_row(T, ranks, local_slots)
        extra = dict(kernel="prior_v2", source="prior_v2")
    elif source == "warmup":
        if table is None:
            raise ValueError("cost table: params source warmup without a table")
        row = table_row(table, T)
        if row is None:
            row = fallback_row(T, ranks, local_slots, bucket_of(T, table.get("buckets")), table.get("tile_rows", TILE_ROWS))
        extra = dict(kernel="warmup")
    else:
        raise ValueError("cost model B params source %r: one of %s" % (source, PARAMS_SOURCES))
    out = {k: v for k, v in row.items() if k != "sha256"}
    out.update(extra, T=T, row_sha256=row["sha256"])
    return sealed(out)
