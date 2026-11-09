# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB movement-aware placement: keep-first assignment and the copy price of an install.

Standard library only (no numpy, no torch): the planner worker imports it standalone, the offline simulator and the CPU-only
tests import the same file, so the code that is simulated is the code that ships.

KEEP-FIRST (planner._move_pick builds it next to the default layout of every layer it evaluates)
  The placement rule decides how many non-home replicas each expert needs (rows-LPT: native doReplication counts - 1; P6
  flex: one per selected expert).  The current layout's replicas of an expert that still needs replicas are kept first
  (no copy): all of them when it needs at least as many, else the ones on the ranks of lowest cost-model rank cost (the
  planner's own model -- shipped B, form T or eF -- with the even split of the layout built so far; ties to the lower slot).
  A replica held on its expert's home rank, or a second one of the same expert on one rank, is not kept (it balances nothing).
  Only the remaining replicas are copied, by the same rule into FREE spare slots (empty in the current layout and not
  retired by the last install, so no in-flight step of any rank reads them): rows-LPT to the least loaded rank (homes + kept
  replicas at load / copies) that does not hold the expert, else the least loaded one that does (the default order); flex to the
  rank of lowest expected cost (+ cf / 2 per replica held or placed), never the home.  A replica that finds no free slot is
  left out, as sticky_layout leaves it out of a wave.

COPY PRICE (the install gate under move>=1)
  The measured copy time c_gpu_ms is the median of the last COPY_HISTORY installs' ms per layer (the startup probe first).
  On NCCL a layer's copy time is affine in the replicas the layer copies (a DEP16 generation-only pilot, per-layer medians:
  3.8 / 5.3 / 7.9 / 19 / 32 replicas -> 0.75 / 0.98 / 1.23 / 2.07 / 3.41 ms, i.e. ~0.35 ms + ~0.09 ms per replica, so
  COPY_FIXED ~ 4 replica-equivalents).  The balancer records the replicas per layer of every install as its measured time
  will see them (the max over its copy chunks of the chunk's mean, nhist_push) and sends the median (nref); a layer that
  copies n replicas is priced c_gpu_ms (COPY_FIXED + n) / (COPY_FIXED + nref): a full default-sized install keeps the default price,
  a keep-first install that copies 2 replicas pays for 2.  A layer that copies nothing is still priced at the fixed part
  (no drop-only installs that would put a ~0 ms copy time into the history).  move=1 keeps the default admission on top (a layer
  installs only when the default gate passes the default layout; the chosen layout must then pay for its own copies): with the live
  payback horizon (2048 replays) the per-copy price alone (move=2) admitted 1.7-15x more installs than move=0 in a replay
  simulation and lost 0.01-0.1 % predicted step time in 13 of 20 shipped-B / P5 cells.
"""

import heapq
import math

# Default move_tol_us (us per layer-step): the keep-first layout is taken while its predicted J is within this of the default
# layout's. A replay simulation (recorded DSV4 routing, DEP16 T 64..160 and DEP32 T32, 5 seeds, move=1): the copy cut grows to
# tol 2 and then saturates (DEP16 replicas/min -19..-79 % at 2 vs -14..-77 % at 0.5, within 1 point of tol 5 / inf) while the
# predicted step cost (61 x J of the layout in force + copy time) improves in every cell; 2 also bounds what one keep-first
# install may give up against the default layout (<= 2 us per layer-step, seen max 1.95, mean +0.1..+0.6 when chosen).
MOVE_TOL_US = 2.0
COPY_FIXED = 4.0  # a layer copy's fixed cost in replica-equivalents (NCCL, DEP16 pilot: ~0.35 ms / ~0.09 ms per replica)
COPY_HISTORY = 9  # balancer.py COPY_HISTORY: the measured copy times c_gpu_ms is the median of


# ---------------------------------------------------------------------------------------------------- layouts
def held_replicas(cur, R: int, H: int, M: int) -> dict:
    """{expert: [spare slots]} of the replicas of ``cur`` that can be kept: a spare slot holding e on a rank other than e's
    home rank, the lowest such slot per (expert, rank); slots ascending, experts in ascending order of insertion."""
    out, seen = {}, set()
    for s in range(R * M):
        e = int(cur[s])
        if e < 0 or s % M < H:
            continue
        r = s // M
        if r == e // H or (e, r) in seen:
            continue
        seen.add((e, r))
        out.setdefault(e, []).append(s)
    return out


def free_spares(cur, R: int, H: int, M: int, retired=()) -> list:
    """Per rank the FREE spare slots of ``cur`` (empty, not retired), ascending: where a copy may land in this wave."""
    ret = set(int(s) for s in retired)
    return [[r * M + j for j in range(H, M) if int(cur[r * M + j]) < 0 and r * M + j not in ret] for r in range(R)]


def plan_of(layout, R: int, H: int, M: int) -> dict:
    """{expert: [ranks of its spare-slot replicas]} of a layout (ranks ascending; experts ascending)."""
    out = {}
    for s in range(R * M):
        e = int(layout[s])
        if e >= 0 and s % M >= H:
            out.setdefault(e, []).append(s // M)
    return {e: sorted(out[e]) for e in sorted(out)}


def keep_first(cur, need: dict, R: int, H: int, M: int, retired, rank_cost, prio, place):
    """The keep-first layout of one layer.

    need: {expert: non-home replicas wanted}; rank_cost(layout, r) -> rank r's cost-model cost under ``layout`` (even split);
    prio(e) -> sort key of the experts that hold more replicas than they need (they choose in this order, then by id);
    place(layout, rem, free) -> [(expert, slot)] puts the remaining replicas ({expert: count}) into the free slots (mutates
    both).  -> (layout, copies [(expert, destination slot)], kept {expert: replicas kept})."""
    new = [-1] * (R * M)
    for r in range(R):
        for j in range(H):
            new[r * M + j] = int(cur[r * M + j])
    held = held_replicas(cur, R, H, M)
    kept, surplus = {}, []
    for e in sorted(held):
        n = int(need.get(e, 0))
        if n <= 0:
            continue  # the expert needs no replica now: its held ones leave the layout
        if len(held[e]) <= n:
            for s in held[e]:
                new[s] = e
            kept[e] = len(held[e])
        else:
            surplus.append(e)
    for e in sorted(surplus, key=lambda x: (prio(x), x)):
        cand, n = list(held[e]), int(need[e])
        while kept.get(e, 0) < n and cand:
            best = None
            for s in cand:
                new[s] = e
                key = (rank_cost(new, s // M), s)
                new[s] = -1
                if best is None or key < best:
                    best = key
            s = best[1]
            new[s] = e
            cand.remove(s)
            kept[e] = kept.get(e, 0) + 1
    rem = {}
    for e in sorted(need):
        k = int(need[e]) - kept.get(e, 0)
        if k > 0:
            rem[e] = k
    free = free_spares(cur, R, H, M, retired)
    copies = place(new, rem, free) if rem else []
    return new, copies, kept


def place_lpt(new, rem: dict, free: list, size, R: int, H: int, M: int) -> list:
    """Rows-LPT (planner.fixedhome_targets' order) of the remaining replicas into free slots: replicas by (-size, expert),
    each to the least loaded rank with a free slot that does not hold the expert (ties to the smaller rank), else to the
    least loaded one that does; rank load = sum of size over the copies the layout holds (homes and kept replicas)."""
    holds = [set() for _ in range(R)]
    load = [0.0] * R
    for s in range(R * M):
        e = int(new[s])
        if e >= 0:
            holds[s // M].add(e)
            load[s // M] += size[e]
    reps = sorted(((size[e], e) for e in rem for _ in range(rem[e])), key=lambda t: (-t[0], t[1]))
    heap = [(load[r], r) for r in range(R) if free[r]]
    heapq.heapify(heap)
    copies = []
    for sz, e in reps:
        aside, r = [], -1
        while heap:
            key = heapq.heappop(heap)
            if e in holds[key[1]]:
                aside.append(key)
                continue
            r = key[1]
            break
        if r < 0 and aside:
            r = aside.pop(0)[1]
        for key in aside:
            heapq.heappush(heap, key)
        if r < 0:
            break  # no rank has a free slot: the remaining replicas are left out of this wave
        s = free[r].pop(0)
        new[s] = e
        copies.append((e, s))
        holds[r].add(e)
        load[r] += sz
        if free[r]:
            heapq.heappush(heap, (load[r], r))
    return copies


def flex_u(m, rates, R: int, H: int) -> list:
    """placement_ef.flex's expected cost u_r of every rank under the identity layout (its destination order)."""
    try:
        from . import placement_ef as pe
    except ImportError:  # standalone (planner worker, simulator, tests)
        import placement_ef as pe
    st = pe.State(m, R)
    for e in range(len(rates)):
        st.add(e // H, pe.slot_terms(m, rates[e], 1.0))
    tau = pe.default_tau(m, st)
    return [st.u(r, tau) for r in range(R)]


def place_flex(new, rem: dict, free: list, u, cf: float, R: int, H: int, M: int) -> list:
    """P6 flex's destination rule for the remaining replicas: experts in flex's order (busiest home rank first, then id),
    each to the rank of lowest load (u_r + cf / 2 per replica held or placed) with a free slot, never its home rank, not a
    rank that holds it; ties to the smaller rank."""
    holds = [set() for _ in range(R)]
    load = list(u)
    for s in range(R * M):
        e = int(new[s])
        if e >= 0:
            holds[s // M].add(e)
            if s % M >= H:
                load[s // M] += 0.5 * cf
    copies = []
    for e in sorted(rem, key=lambda x: (-u[x // H], x)):
        for _ in range(rem[e]):
            ds = [x for x in range(R) if free[x] and x != e // H and e not in holds[x]]
            if not ds:
                break
            b = min(ds, key=lambda x: (load[x], x))
            s = free[b].pop(0)
            new[s] = e
            copies.append((e, s))
            holds[b].add(e)
            load[b] += 0.5 * cf
    return copies


# ---------------------------------------------------------------------------------------------------- copy price
def probe_copies(R: int, H: int, M: int) -> float:
    """Replicas per layer of balancer.after_warmup's startup copy probe (every rank receives max(1, (M - H) // 2))."""
    return float(R * max(1, (M - H) // 2))


def median(xs) -> float:
    """numpy.median of a non-empty list (the mean of the two middle values for an even count)."""
    v = sorted(float(x) for x in xs)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def nref(hist, R: int, H: int, M: int) -> float:
    """The replicas per layer the measured copy time stands for: the median of the history (the probe's when empty)."""
    return median(hist) if hist else probe_copies(R, H, M)


def nhist_push(hist, installs, c_gpu_ms, budget_ms: float, per_step: int, R: int, H: int, M: int) -> list:
    """The history after one more install: its replicas per layer as its measured copy time will see them (balancer
    _poll_installs: the max over its copy chunks of chunk ms / chunk layers; chunks of max(1, min(per_step, budget / c))
    layers in install order, as balancer._apply_decision), the last COPY_HISTORY entries, the probe's first."""
    h = list(hist) if hist else [probe_copies(R, H, M)]
    if not installs:
        return h
    c = c_gpu_ms
    chunk = max(1, min(per_step, int(budget_ms / c))) if c else per_step
    n = [len(ins["copies"]) for ins in installs]
    per_layer = max(sum(n[k:k + chunk]) / len(n[k:k + chunk]) for k in range(0, len(n), chunk))
    return (h + [float(per_layer)])[-COPY_HISTORY:]


def copy_price(c_gpu_ms: float, n: int, ref: float) -> float:
    """ms one layer's install of n replicas costs: c_gpu_ms (COPY_FIXED + n) / (COPY_FIXED + ref)."""
    if not c_gpu_ms:
        return 0.0
    return float(c_gpu_ms) * (COPY_FIXED + float(n)) / (COPY_FIXED + max(0.0, float(ref)))


def max_dest(copies, M: int) -> int:
    """The most copies one destination rank receives (the fabric gate's rows are planes x this)."""
    per = {}
    for _, s in copies:
        r = int(s) // M
        per[r] = per.get(r, 0) + 1
    return max(per.values()) if per else 0


def copy_bottleneck(copies, R: int, H: int, M: int) -> int:
    """The most replicas one rank sends or receives (sources = homes, as the NCCL copy): diagnostics only."""
    tx, rx = [0] * R, [0] * R
    for e, s in copies:
        tx[int(e) // H] += 1
        rx[int(s) // M] += 1
    return max(max(tx), max(rx)) if copies else 0


def note(d: dict) -> str:
    """The renewal line's move note from a decide reply's diag ('' without move)."""
    if not d or "mv_eval" not in d:
        return ""
    return (
        f"; move {d.get('mv_mode', 1)} tol {d.get('mv_tol', 0.0):g} us: keep-first {d.get('mv_k', 0)}/{d.get('mv_eval', 0)} "
        f"layers (same {d.get('mv_same', 0)}, go2 gate passed {d.get('mv_go2', 0)}), copies {d.get('mv_cp', 0)} vs go2 "
        f"{d.get('mv_cp_u', 0)}, dJ(keep-first - go2) {d.get('mv_djk', 0.0):+.2f} us, nref {d.get('mv_nref', 0.0):.1f}"
    )
