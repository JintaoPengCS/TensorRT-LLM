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

import heapq
import math
import struct

try:
    from . import costmodel, shares
    from .geometry import Geometry
except ImportError:  # imported standalone by the planner worker
    import costmodel
    import shares
    from geometry import Geometry

LAMBDA = 0.95  # the statistics kernel's EMA weight (stats_kernels.cu)
# The gate model of C (form T, the microbenchmark fit).
GATE_MODEL_T = dict(form="T", m=256, slope_us=0.0125, beta_us=0.009, tau_w_us=0.0)


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


def layer_us(model: dict, cbar, phys, weights, geo: Geometry) -> float:
    """Busiest GPU's MoE microseconds of one layer under ``phys`` / ``weights`` (form T or B)."""
    return float(costmodel.layer_us(model, cbar, phys, weights, geo.ranks))


def solve_weights(mode: str, totals, scale: float, phys, geo: Geometry, b_params):
    """Per-slot shares of ``phys`` (C: the rows split of the window totals; B: cost model B on the per-launch rows
    totals x scale; a refused B input falls back to the rows split, as shipped)."""
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


def decide(req: dict) -> dict:
    """One renewal on every rank: shares of every layer's current layout and the layers whose new replicas pay.

    req: geo (Geometry fields), mode 'C' | 'B', b_params, gate_model, totals {layer: every expert's statistics
    EMA summed over ranks}, scales {layer: EMA scale to per-launch rows}, layouts {layer: current layout},
    retired {layer: slots the last install freed}, b_layers {layer: True when B's solver prices it}, n_rem,
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
    out_tables, cands, plans = {}, {}, {}
    for layer in sorted(req["totals"]):
        totals, scale = req["totals"][layer], req["scales"][layer]
        cbar = [float(t) * scale for t in totals]
        cur = req["layouts"][layer]
        lmode = req["mode"] if req["b_layers"].get(layer, False) else "C"
        model = req["gate_model"] if lmode == "B" else GATE_MODEL_T
        w_cur = solve_weights(lmode, totals, scale, cur, geo, req["b_params"])
        out_tables[layer] = tables(cur, w_cur, E)
        if not req["install"] or n_rem is None or sum(totals) <= 0.0:
            continue
        plan = fixedhome_targets(totals, geo, req["total_active"], req["cap"])
        new, copies = sticky_layout(cur, plan, geo, req["retired"].get(layer, ()))
        if new == cur:
            continue
        w_new = solve_weights(lmode, totals, scale, new, geo, req["b_params"])
        gain_ms = (
            layer_us(model, cbar, cur, w_cur, geo) - layer_us(model, cbar, new, w_new, geo)
        ) / 1e3
        if gain_ms * n_rem > c_gpu:
            cands[layer] = gain_ms
            plans[layer] = (new, copies, w_new)
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
    return dict(
        tables=out_tables,
        installs=installs,
        gains={li: cands[li] for li in order},
        candidates=len(cands),
    )


def renewal_interval(floor: int, fixed_ms, slope_ms, growth, n_rem) -> int:
    """K = max(floor, ceil(sqrt(2 F / (S r)))), at most max(floor, N_rem); the floor without a positive F, S and r."""
    if not fixed_ms or not slope_ms or not growth or fixed_ms <= 0 or slope_ms <= 0 or growth <= 0:
        return floor
    k = max(floor, math.ceil(math.sqrt(2.0 * fixed_ms / (slope_ms * growth))))
    if n_rem is not None:
        k = min(k, max(floor, int(n_rem)))
    return k
