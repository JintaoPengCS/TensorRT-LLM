# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB decisions: replica counts, sticky fixed-home placement, traffic shares and the install gate.

Pure functions over numpy (no torch, no tensorrt_llm import): the planner worker imports this module
standalone, every rank computes the same decision from the same gathered load. The replica count
allocation and the placement are the reference implementation's (native doReplication; fixed-home LPT
into FREE slots, ties to the smaller rank, never a rank that already holds the expert) with its constants;
shares come from the compiled solvers (C: rows objective, B: cost model B); the gate is the copy-basis
rule, gain per step x remaining steps > the measured copy time of one layer.
"""

import hashlib
import heapq
import math
import os
import sys
import struct

try:
    from . import costmodel, placement_ef, shares
    from .geometry import Geometry
except ImportError:  # imported standalone by the planner worker
    import costmodel
    import placement_ef
    import shares
    from geometry import Geometry

LAMBDA = 0.95  # the statistics kernel's EMA weight (stats_kernels.cu)
# The gate model of C (form T, the microbenchmark fit).
GATE_MODEL_T = dict(form="T", m=256, slope_us=0.0125, beta_us=0.009, tau_w_us=0.0)


# ================================================================================================ P5 / P6 controls
# Both default to the shipped behaviour.  Read once per process: the planner worker inherits the serving process's environment.
#   TLLM_GENEPLB_OBJECTIVE  mean (shipped: cost model B on the window means, LSE tau 0.5, F = [mean rows > 0])
#                           ef   (P5: placement_ef's expected-max objective on the prior v2 row of (EP, T): a copy is charged
#                                 cf x P(active per replay), zero-load homes by activity, LSE tau from the per-step dispersion;
#                                 shares by l3b_core.c pricing 2 (never a q=1 split), home-only unless the pick beats it by
#                                 gate_margin_us; layer_us -- hence decide()'s install gate -- prices layouts with it too)
#   TLLM_GENEPLB_PLACEMENT  lpt  (shipped rows-LPT)
#                           flex (P6: every rank's 2 most loaded homes get a replica = options for the fast level's
#                                 whole-expert moves; needs OBJECTIVE ef)
#                           greedy (experimental: placement_ef.place, greedy by the change of the eF objective; needs ef)
def _control(name: str, default: str, allowed) -> str:
    v = os.environ.get(name, default).strip().lower()
    if v not in allowed:
        print("GenEPLB planner: %s=%r is not one of %s; using %r" % (name, v, allowed, default), file=sys.stderr)
        v = default
    return v


OBJECTIVE = _control("TLLM_GENEPLB_OBJECTIVE", "mean", ("mean", "ef"))
PLACEMENT = _control("TLLM_GENEPLB_PLACEMENT", "lpt", ("lpt", "flex", "greedy"))
if PLACEMENT != "lpt" and OBJECTIVE != "ef":
    print("GenEPLB planner: TLLM_GENEPLB_PLACEMENT=%s needs TLLM_GENEPLB_OBJECTIVE=ef; using lpt" % PLACEMENT, file=sys.stderr)
    PLACEMENT = "lpt"
_EF = {}  # the layer under decision (decide() solves its CURRENT layout first): totals, rates, params, model, tau, cur, hint
_EF_PARAMS = {}  # (T, ranks, local_slots) -> the prior v2 params
# Run-time switching through the control file (balancer.py CTL_SPEC objective /
# placement): a decide() request whose ctl carries objective / placement (every rank sends the MAX over ranks of the codes,
# so every rank decides alike) runs on them instead of the env values above; a request without them (no control file) runs on
# the env values. placement=flex runs only with objective=ef and the request's pad=drop (P6 flex with spread
# padding is worse than shipped in simulation); otherwise it is inert (lpt placement), with a one-time
# warning per reason on stderr.
_RUN = None  # (objective, placement) of the decide() in progress, from its request; None: the env values
_WARNED = set()  # the one-time warnings already printed


def _objective() -> str:
    return OBJECTIVE if _RUN is None else _RUN[0]


def _placement() -> str:
    return PLACEMENT if _RUN is None else _RUN[1]


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print("GenEPLB planner: " + msg, file=sys.stderr)


def _run_controls(req: dict):
    """(objective, placement) of a decide() request whose ctl carries the run-time fields, else None (the env values)."""
    ctl = req.get("ctl") or {}
    if "objective" not in ctl and "placement" not in ctl:
        return None
    if OBJECTIVE != "mean" or PLACEMENT != "lpt":
        _warn_once(
            "env",
            "TLLM_GENEPLB_OBJECTIVE / TLLM_GENEPLB_PLACEMENT (%s / %s) are ignored: the control file's objective / "
            "placement fields rule" % (OBJECTIVE, PLACEMENT),
        )
    obj = "ef" if ctl.get("objective") == "ef" else "mean"
    plc = "flex" if ctl.get("placement") == "flex" else "lpt"
    if plc == "flex" and obj != "ef":
        _warn_once("flex-ef", "placement=flex needs objective=ef; inert (lpt placement) while objective=shipped")
        plc = "lpt"
    elif plc == "flex" and ctl.get("pad") != "drop":
        _warn_once("flex-pad", "placement=flex needs pad=drop; inert (lpt placement) while pad is not drop")
        plc = "lpt"
    return obj, plc


def _ef_params(b_params: dict, geo: Geometry) -> dict:
    key = (int(b_params["T"]), int(geo.ranks), int(geo.local_slots))
    p = _EF_PARAMS.get(key)
    if p is None:
        p = _EF_PARAMS[key] = costmodel.params_ef(*key)
    return p


def _ef_layer(totals, scale: float, geo: Geometry, b_params: dict) -> dict:
    """The eF context of the layer under decision, built at its first solve (the same ``totals`` object all along)."""
    if _EF.get("totals") is totals and _EF.get("scale") == scale:
        return _EF
    p = _ef_params(b_params, geo)
    m = placement_ef.Model.from_params(p, geo.ranks)
    rates = [float(t) * float(scale) for t in totals]
    _EF.clear()
    _EF.update(
        totals=totals, scale=scale, params=p, model=m, rates=rates, cur=None, hint=None,
        tau=placement_ef.identity_tau(m, rates, geo.ranks, geo.homes),
    )
    return _EF


def _home_weights(phys, geo: Geometry) -> list:
    M, H = geo.local_slots, geo.homes
    return [1.0 if (e >= 0 and s % M < H) else 0.0 for s, e in enumerate(phys)]


def _hint_weights(phys, hint: dict, geo: Geometry):
    """The placement's intended shares on ``phys`` (None when a hinted replica did not land: sticky_layout's wave)."""
    M, H = geo.local_slots, geo.homes
    w = _home_weights(phys, geo)
    for e, lst in hint.items():
        for r, sh in lst:
            if r == e // H:
                w[r * M + e % H] = float(sh)
                continue
            s = next((r * M + j for j in range(H, M) if phys[r * M + j] == e), None)
            if s is None:
                return None
            w[s] = float(sh)
    return w


def _solve_weights_ef(totals, scale, phys, geo: Geometry, b_params: dict) -> list:
    """P5 fast level: the lowest eF objective of {l3b_core.c pricing 2 on rates + lam0, the plan's hint}, or the home-only
    split (identity routing) unless that beats it by gate_margin_us."""
    ctx = _ef_layer(totals, scale, geo, b_params)
    if ctx["cur"] is None:
        ctx["cur"] = list(phys)
    p, m, tau, rates = ctx["params"], ctx["model"], ctx["tau"], ctx["rates"]
    R, M = geo.ranks, geo.local_slots
    w_home = _home_weights(phys, geo)
    J_home = placement_ef.layer_J(m, rates, phys, w_home, R, M, tau=tau)[0]
    best_w, best_J = w_home, J_home
    if float(scale) > 0.0:
        load = [float(t) + p["act_lam0"] / float(scale) for t in totals]
        w = shares.solve_b(load, phys, float(scale), M, costmodel.l3b_params_ef(p, tau))
        if w is not None:
            w = [float(x) for x in w]
            J = placement_ef.layer_J(m, rates, phys, w, R, M, tau=tau)[0]
            if J < best_J:
                best_w, best_J = w, J
    if ctx["hint"]:
        w2 = _hint_weights(phys, ctx["hint"], geo)
        if w2 is not None:
            J2 = placement_ef.layer_J(m, rates, phys, w2, R, M, tau=tau)[0]
            if J2 < best_J - 1e-9:
                best_w, best_J = w2, J2
    if J_home - best_J < p["gate_margin_us"]:
        return w_home
    return best_w


def _fixedhome_targets_ef(geo: Geometry, total_active: int, cap: int) -> dict:
    """P6 plan of the layer under decision ({expert: [ranks]}, as fixedhome_targets): flex options, or the greedy plan unless it
    beats identity by gate_margin_us."""
    ctx = _EF
    p = ctx["params"]
    M, H = geo.local_slots, geo.homes
    cur = ctx.get("cur") or []
    held = {e: s // M for s, e in enumerate(cur) if e >= 0 and s % M >= H}
    if _placement() == "flex":
        ctx["hint"] = None
        return placement_ef.flex(
            ctx["model"], ctx["rates"], geo.ranks, H, total_active - geo.experts, cap, per_rank=int(p["per_rank"]),
            held=held, keep_bonus=p["keep_bonus"],
        )
    res = placement_ef.place(
        ctx["model"], ctx["rates"], geo.ranks, H, total_active - geo.experts, cap, min_gain_us=p["place_min_gain_us"],
        keep=set(held.items()), new_penalty_us=p["new_penalty_us"], tau=ctx["tau"],
    )
    if res["J_identity"] - res["J"] < p["gate_margin_us"]:
        ctx["hint"] = None
        return {}
    ctx["hint"] = res["shares"]
    return {e: list(rs) for e, rs in res["plan"].items()}


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ema_scale(steps: int) -> float:
    """Per-launch expected rows from the EMA after ``steps`` launches since its reset."""
    return (1.0 - LAMBDA) / (1.0 - LAMBDA**steps)


def do_replication(load, total_slots: int, experts: int) -> list:
    """Copies per expert (native doReplication): float32 loads, max-heap of load / copies, ties to the larger id."""
    load = [_f32(float(x)) for x in load]
    if total_slots == experts:
        return [1] * experts
    if sum(load) == 0.0:
        base, rem = divmod(total_slots, experts)
        return [base + (i < rem) for i in range(experts)]
    counts = [1] * experts
    heap = [(-(load[e] / 1.0), -e) for e in range(experts)]
    heapq.heapify(heap)
    for _ in range(total_slots - experts):
        _, neg_e = heapq.heappop(heap)
        e = -neg_e
        counts[e] += 1
        heapq.heappush(heap, (-(load[e] / float(counts[e])), -e))
    return counts


def fixedhome_targets(load, geo: Geometry, total_active: int, cap: int) -> dict:
    """The fresh fixed-home plan of one layer: {expert: [ranks of its replicas]} (homes never move).

    Replica counts from do_replication; replicas in the order (-load / copies, expert), each to the
    least-loaded rank with room under ``cap`` that does not hold the expert (ties to the smaller rank),
    a second copy on one rank only when every rank with room holds it -- the shipped planner's order.
    """
    if _placement() != "lpt" and _EF.get("totals") is load:  # P6 (decide() solved this layer's current layout first)
        return _fixedhome_targets_ef(geo, total_active, cap)
    E, R, H = geo.experts, geo.ranks, geo.homes
    load = [_f32(float(x)) for x in load]
    counts = do_replication(load, total_active, E)
    rank_experts = [set(range(r * H, (r + 1) * H)) for r in range(R)]
    rank_load = [sum(load[e] / counts[e] for e in sorted(rank_experts[r])) for r in range(R)]
    extra = [0] * R
    out = {}
    replicas = sorted(
        ((load[e] / counts[e], e) for e in range(E) for _ in range(counts[e] - 1)),
        key=lambda t: (-t[0], t[1]),
    )
    heap = [(rank_load[r], r) for r in range(R) if extra[r] < cap]
    heapq.heapify(heap)
    for size, e in replicas:
        aside, r = [], -1
        while heap:
            key = heapq.heappop(heap)
            if e in rank_experts[key[1]]:
                aside.append(key)
                continue
            r = key[1]
            break
        if r < 0 and aside:
            r = aside.pop(0)[1]
        for key in aside:
            heapq.heappush(heap, key)
        if r < 0:
            break  # no rank has room: the remaining replicas are not placed
        rank_experts[r].add(e)
        rank_load[r] += size
        extra[r] += 1
        out.setdefault(e, []).append(r)
        if extra[r] < cap:
            heapq.heappush(heap, (rank_load[r], r))
    return out


def sticky_layout(cur: list, plan: dict, geo: Geometry, retired=()):
    """Map a fresh plan onto the current layout in one wave.

    A planned (expert, rank) replica that the current layout already holds keeps its slot; any other
    lands in the lowest FREE spare slot of that rank (a slot no rank routes to now and none routed to
    in the previous layout: ``retired`` slots wait one renewal, so no in-flight step of any rank still
    reads them); a replica that finds no such slot is left out of this wave. Current replicas the plan
    drops leave the new layout. -> (new layout, copies [(expert, destination slot)]).
    """
    M, H = geo.local_slots, geo.homes
    new = [-1] * len(cur)
    for r in range(geo.ranks):
        for j in range(H):
            new[r * M + j] = cur[r * M + j]
    held = {}  # (expert, rank) -> current replica slots
    for s, e in enumerate(cur):
        if e >= 0 and s % M >= H:
            held.setdefault((e, s // M), []).append(s)
    retired = set(retired)
    free = [
        [r * M + j for j in range(H, M) if cur[r * M + j] < 0 and r * M + j not in retired]
        for r in range(geo.ranks)
    ]
    copies = []
    for e in sorted(plan):
        for r in plan[e]:
            slots = held.get((e, r))
            if slots:
                new[slots.pop(0)] = e
            elif free[r]:
                s = free[r].pop(0)
                new[s] = e
                copies.append((e, s))
    return new, copies


def source_slot(layout: list, expert: int, geo: Geometry) -> int:
    """The slot an install copies ``expert`` from: its home (always resident)."""
    return geo.home_slot(expert)


def copy_sources(cur: list, copies, geo: Geometry, egress: list, load=None) -> list:
    """The slot each copy of an install reads (the fabric transport maps every holder's rows on every
    rank): among the expert's current holders -- its home and every spare slot holding it in the CURRENT layout, all
    resident (copies land only in FREE slots, and a slot a layout drops keeps its bytes until a later install overwrites
    it, which needs this install published first) -- the one on the least loaded rank: fewest reads assigned so far in
    this install (``egress`` per rank, updated in place across the install's layers), then the fewest modeled rows
    (``load`` per rank under the current layout and shares), then the lower slot. A pure function of the decision: the
    same on every rank."""
    M = geo.local_slots
    holders = {}
    for s, e in enumerate(cur):
        if e >= 0:
            holders.setdefault(int(e), []).append(s)
    out = []
    for e, _ in copies:
        cand = holders.get(int(e)) or [geo.home_slot(int(e))]
        s = min(cand, key=lambda s: (egress[s // M], float(load[s // M]) if load is not None else 0.0, s))
        egress[s // M] += 1
        out.append(int(s))
    return out


def copy_rows(copies, geo: Geometry, planes: int) -> int:
    """The fabric gate's rows of a layer's copies: planes x the most copies one destination rank pulls
    (each rank pulls its replicas on its own copy stream; the busiest one sets the layer's copy time)."""
    per = {}
    for _, d in copies:
        r = int(d) // geo.local_slots
        per[r] = per.get(r, 0) + 1
    return int(planes) * max(per.values()) if per else 0


def layer_us(model: dict, cbar, phys, weights, geo: Geometry) -> float:
    """Busiest GPU's MoE microseconds of one layer under ``phys`` / ``weights`` (form T or B)."""
    if _objective() == "ef" and costmodel.is_b(model) and "T" in model:  # P5: the eF objective prices the gate's layouts
        p = _ef_params(model, geo)
        m = placement_ef.Model.from_params(p, geo.ranks)
        rates = [float(x) for x in cbar]
        tau = placement_ef.identity_tau(m, rates, geo.ranks, geo.homes)
        return float(placement_ef.layer_J(m, rates, phys, weights, geo.ranks, geo.local_slots, tau=tau)[0])
    return float(costmodel.layer_us(model, cbar, phys, weights, geo.ranks))


def solve_weights(mode: str, totals, scale: float, phys, geo: Geometry, b_params):
    """Per-slot shares of ``phys`` (C: the rows split of the window totals; B: cost model B on the per-launch rows
    totals x scale; a refused B input falls back to the rows split, as shipped)."""
    if _objective() == "ef" and mode == "B" and b_params is not None:  # P5
        return _solve_weights_ef(totals, scale, phys, geo, b_params)
    if mode == "B" and b_params is not None:
        w = shares.solve_b(totals, phys, float(scale), geo.local_slots, b_params)
        if w is not None:
            return w
    return shares.solve_rows(totals, phys, geo.experts, geo.ranks, geo.local_slots)


def remaining_steps(rem_sum: float, count: int, accepted: float):
    """The shipped N_rem: the steps the in-flight requests still run, sum_i rem_i / a over them / count."""
    if count <= 0 or accepted is None or accepted <= 0:
        return None
    return rem_sum / accepted / count


def tables(layout: list, weights, experts: int):
    """The kernel's four per-layer arrays (expert copies, start offsets, slot ids, cumulative shares)."""
    count, start, slots = shares.tables_from_phys(layout, experts)
    return count, start, slots, shares.cumw_from_weights(weights, layout, (count, start, slots))


def _ctl_admit(ctl: dict, layer, plan: dict, gain_ms: float, hyst_next: dict, diag: dict) -> bool:
    """The admission tests after the shipped copy-basis test: the margin (gain per layer-step >
    gate_margin_us), then the hysteresis (the same plan digest passes gate_hyst consecutive renewals; ctl['hyst'] holds the
    previous renewal's streaks, hyst_next receives this renewal's; an admitted layer's streak restarts)."""
    diag["n_copy"] += 1
    margin = float(ctl.get("gate_margin_us") or 0.0)
    if margin > 0.0 and not gain_ms * 1e3 > margin:
        diag["n_margin"] += 1
        return False
    need = int(ctl.get("gate_hyst") or 1)
    if need <= 1:
        return True
    key = hashlib.blake2b(
        repr(sorted((int(e), tuple(int(r) for r in rs)) for e, rs in plan.items())).encode(),
        digest_size=8,
    ).hexdigest()
    last = (ctl.get("hyst") or {}).get(layer)
    wins = last[1] + 1 if last is not None and last[0] == key else 1
    if wins >= need:
        return True
    hyst_next[layer] = (key, wins)
    diag["n_hyst"] += 1
    return False


def decide(req: dict) -> dict:
    """_decide() under the request's objective / placement (_run_controls; without them the env values),
    set for this decision only."""
    global _RUN
    _RUN = _run_controls(req)
    try:
        return _decide(req)
    finally:
        _RUN = None


def _decide(req: dict) -> dict:
    """One renewal on every rank: shares of every layer's current layout and the layers whose new replicas pay.

    req: geo (Geometry fields), mode 'C' | 'B', b_params, gate_model, totals {layer: every expert's statistics
    EMA summed over ranks}, scales {layer: EMA scale to per-launch rows}, layouts {layer: current layout},
    retired {layer: slots the last install freed}, b_layers {layer: True when B's solver prices it}, n_rem (the
    payback horizon in steps), frozen (layers that may not install this renewal),
    c_gpu_ms (measured copy ms of one layer, None before the first install), total_active, cap, install
    (False: shares only), max_layers. As shipped: the plan and the C split read the totals, the gate and B
    read the per-launch rows totals x scale.
    -> dict(tables {layer: kernel arrays of the current layout at its new shares}, installs [dict(layer,
    layout, copies, tables)], gains {layer: ms/step}, candidates). The gate is the shipped copy basis: a
    layer installs when its gain per step x the remaining steps exceeds the measured copy time of a layer.
    """
    geo = Geometry(**req["geo"])
    E, n_rem = geo.experts, req["n_rem"]
    c_gpu = req["c_gpu_ms"] or 0.0
    c_row = req.get("c_row_ms")  # the fabric transport's per-row price; None: the per-layer c_gpu
    planes = int(req.get("planes") or 7)
    src_load = {}  # layer -> modeled rows per rank under the current layout (copy sources)
    frozen = set(req.get("frozen", ()))
    ctl = req.get("ctl")  # gate overrides and diagnostics; absent: the shipped gate
    if ctl is not None:
        home = geo.home_layout()
        w_home = costmodel.even_split(home, E)
        hyst_next = {}
        diag = dict(
            j_cur=0.0, cur_id=0.0, cur_new=0.0, adm=0.0, n_eval=0, n_copy=0, n_margin=0, n_hyst=0
        )
    out_tables, cands, plans = {}, {}, {}
    for layer in sorted(req["totals"]):
        totals, scale = req["totals"][layer], req["scales"][layer]
        cbar = [float(t) * scale for t in totals]
        cur = req["layouts"][layer]
        lmode = req["mode"] if req["b_layers"].get(layer, False) else "C"
        model = req["gate_model"] if lmode == "B" else GATE_MODEL_T
        if ctl is not None and lmode == "B" and ctl.get("gate_tau") is not None:
            model = dict(model, tau=float(ctl["gate_tau"]))  # the gate's LSE tau (0: hard max); the solver keeps its own
        w_cur = solve_weights(lmode, totals, scale, cur, geo, req["b_params"])
        out_tables[layer] = tables(cur, w_cur, E)
        if ctl is not None:  # the gate model's J of the current layout and its excess over identity (home slots only)
            j_cur = layer_us(model, cbar, cur, w_cur, geo)
            diag["j_cur"] += j_cur
            diag["cur_id"] += j_cur - layer_us(model, cbar, home, w_home, geo)
        if not req["install"] or n_rem is None or sum(totals) <= 0.0 or layer in frozen:
            continue
        plan = fixedhome_targets(totals, geo, req["total_active"], req["cap"])
        new, copies = sticky_layout(cur, plan, geo, req["retired"].get(layer, ()))
        if new == cur:
            continue
        w_new = solve_weights(lmode, totals, scale, new, geo, req["b_params"])
        gain_ms = (
            layer_us(model, cbar, cur, w_cur, geo) - layer_us(model, cbar, new, w_new, geo)
        ) / 1e3
        if ctl is not None:
            diag["n_eval"] += 1
            diag["cur_new"] += gain_ms * 1e3
        if gain_ms * n_rem > (c_gpu if c_row is None else c_row * copy_rows(copies, geo, planes)):
            if ctl is not None and not _ctl_admit(ctl, layer, plan, gain_ms, hyst_next, diag):
                continue
            cands[layer] = gain_ms
            plans[layer] = (new, copies, w_new)
            if req.get("sources"):
                src_load[layer] = costmodel.slot_rows(cbar, cur, w_cur).reshape(geo.ranks, geo.local_slots).sum(axis=1)
    order = sorted(cands, key=lambda li: (-cands[li], li))[: req["max_layers"]]
    installs = [
        dict(
            layer=li,
            layout=plans[li][0],
            copies=plans[li][1],
            tables=tables(plans[li][0], plans[li][2], E),
        )
        for li in order
    ]
    if req.get("sources"):  # every copy's source slot, the reads balanced over the whole install
        egress = [0] * geo.ranks
        for ins in installs:
            li = ins["layer"]
            ins["sources"] = copy_sources(req["layouts"][li], ins["copies"], geo, egress, src_load.get(li))
    out = dict(
        tables=out_tables,
        installs=installs,
        gains={li: cands[li] for li in order},
        candidates=len(cands),
    )
    if ctl is not None:
        diag["adm"] = 1e3 * sum(cands[li] for li in order)
        out.update(diag=diag, hyst=hyst_next)
    return out


def renewal_interval(floor: int, fixed_ms, slope_ms, growth, n_rem) -> int:
    """K = max(floor, ceil(sqrt(2 F / (S r)))), at most max(floor, N_rem); the floor without a positive F, S and r."""
    if not fixed_ms or not slope_ms or not growth or fixed_ms <= 0 or slope_ms <= 0 or growth <= 0:
        return floor
    k = max(floor, math.ceil(math.sqrt(2.0 * fixed_ms / (slope_ms * growth))))
    if n_rem is not None:
        k = min(k, max(floor, int(n_rem)))
    return k
