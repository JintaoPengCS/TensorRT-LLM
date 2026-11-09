# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB P5/P6: the activity-aware expected-max objective ("eF") and memory-traffic placement.

Standard library only (no numpy, no torch): the planner worker imports it standalone, and so does the offline
simulator, so the code that is simulated is the code that ships.

MODEL (per MoE layer, per decode replay; one copy = one physical slot holding an expert)
  A copy with planned rows x per replay (x = share x the expert's window rate) is ACTIVE (streams the expert's 37.2 MB of
  NVFP4 weights once: the first tile, cf) with probability
      a(x) = 1 - (1 + (x + lam0 w) / k)^-k            (negative binomial, shape k; k <= 0: Poisson 1 - exp(-x))
  where lam0 is the per-expert rate shrinkage (window estimates regress to the mean: experts with ZERO window load are
  active at 8-11 % of the next steps on real DSV4 routing) shared by the copies in proportion to their share w.
  Later 256-row tiles: E[extra tiles] = sum_{j>=1} P(X > j tile), X ~ Normal(x, disp x) (the shipped poisson pricing).
  Rank r:  mu_r = sum_s [cf a_s + cl t_s + cc x_s] + ch max_s t_s,      v_r = cf^2 sum_s a_s (1 - a_s)
  Layer:   J = tau log sum_r exp((mu_r + v_r / (2 tau)) / tau)
  J is the Gaussian moment-generating bound on E[max_r c_r] (E max <= tau log sum E exp(c_r / tau)); the default tau =
  sigma_med / sqrt(2 ln R) minimises the bound for R equal ranks (it is then mu + sigma sqrt(2 ln R), the Gaussian max).
  So J prices (a) the per-step activity of every copy (a split q=1 expert pays ~2 activations), (b) zero-load homes by
  their activity, (c) the per-step dispersion of every rank -- what the shipped objective (cf [mean rows > 0], LSE tau 0.5
  on the window means, zero-load homes free) does not.

PLACEMENT (P6; replaces the rows-LPT of planner.fixedhome_targets)
  flex (recommended): OPTIONS for the fast level.  Every rank's 2 most loaded (persistently active, p ~ 1) homes get one
    replica on another rank (destination: the lowest expected cost with room; a held replica keeps its rank, hysteresis
    keep_bonus).  The fast level (P5 shares, 0/1 at q = 1) then decides at EVERY renewal, without a copy, whether the home
    or the replica serves: a whole-expert move takes a full weight stream (cf) off a rank that is busy now, and gives it
    back when it is not.  In the steady-state simulation (cooldown 32, real DSV4 routing) this beats both the shipped
    rows-LPT and the greedy placement below at every T, with fewer copies than the shipped planner.
  greedy (experimental): greedy from the identity layout by the change of J on the CURRENT window -- MOVE a whole expert
    (q=1 lever: home share 0) or SPLIT one above a tile (q>=2: the home keeps one tile) from the TOP_SRC busiest to the
    TOP_DST least busy ranks; best move first while it gains min_gain_us (+ new_penalty_us for a replica not held).  It
    wins only when re-placed at every renewal (fresh); with a cooldown its window-specific moves go stale.

SHARES (P5 fast level on any layout, e.g. a stale one): coordinate descent over the replicated experts' options
  {all rows on copy c} U {one-tile fill on the home, rest on copy c}, by J; it never splits a q=1 expert.  (The planner
  uses l3b_core.c pricing 2 for the same objective; this Python twin is the simulator's and the tests' reference.)

The identity layout's J and the plan's J are both returned so the caller can gate against identity with a margin.
"""

import math

TILE = 256.0


# ---------------------------------------------------------------------------------------------------- the model
class Model:
    """Calibrated per-(EP, T-tier) parameters (data/costmodel_b_prior_v2.json row) plus the derived tau."""

    __slots__ = ("cf", "cl", "cc", "ch", "tile", "k", "lam0", "disp", "tau", "fill", "ranks")

    def __init__(self, cf, cl, cc, ch, k=8.0, lam0=0.1, disp=4.0, tile=TILE, tau=None, fill=None, ranks=16):
        self.cf, self.cl, self.cc, self.ch = float(cf), float(cl), float(cc), float(ch)
        self.tile, self.k, self.lam0, self.disp = float(tile), float(k), float(lam0), float(disp)
        self.tau = None if tau is None else float(tau)
        self.fill = float(fill) if fill is not None else self.tile - 3.0 * math.sqrt(self.tile)  # one tile minus 3 sigma
        self.ranks = int(ranks)

    @classmethod
    def from_params(cls, p: dict, ranks: int):
        return cls(
            p["cf"], p["cl"], p["cc"], p["ch"], k=p.get("act_k", 8.0), lam0=p.get("act_lam0", 0.1),
            disp=p.get("tile_disp", 4.0), tile=p.get("tile_rows", TILE), tau=p.get("tau_ef"), fill=p.get("fill_rows"),
            ranks=ranks,
        )


def act(x: float, k: float) -> float:
    """P(X >= 1) of a copy with expected rows x per replay (negative binomial shape k; k <= 0: Poisson)."""
    if not x > 0.0:
        return 0.0
    if k <= 0.0:
        return 1.0 - math.exp(-x)
    return 1.0 - math.exp(-k * math.log1p(x / k))


def extra_tiles(x: float, tile: float, disp: float) -> float:
    """E[ceil(X / tile)] - P(X >= 1) = sum_{j>=1} P(X > j tile), X ~ Normal(x, disp x) with continuity correction."""
    if not x > tile * 0.25:
        return 0.0
    sig = math.sqrt(disp * x) if disp > 0.0 else 0.0
    s, j = 0.0, 1
    while j <= 64:
        b = j * tile
        if sig > 0.0:
            z = (b + 0.5 - x) / sig
            if z > 8.0:
                break
            s += 0.5 * math.erfc(z * 0.7071067811865476)
        else:
            if x > b:
                s += 1.0
            else:
                break
        j += 1
    return s


def slot_terms(m: Model, rate: float, share: float):
    """(mean us, variance us^2, extra tiles) of one copy holding `share` of an expert whose window rate is `rate`."""
    if not share > 0.0:
        return 0.0, 0.0, 0.0
    x = share * rate
    a = act(x + share * m.lam0, m.k)
    t = extra_tiles(x, m.tile, m.disp)
    return m.cf * a + m.cl * t + m.cc * x, m.cf * m.cf * a * (1.0 - a), t


class State:
    """Per-rank sums of a layout under the model: mu_r (without the hot term), v_r, the per-slot extra tiles."""

    def __init__(self, m: Model, R: int):
        self.m, self.R = m, R
        self.mu = [0.0] * R
        self.v = [0.0] * R
        self.tiles = [[] for _ in range(R)]

    def add(self, r, terms, sign=1.0):
        mu, v, t = terms
        self.mu[r] += sign * mu
        self.v[r] += sign * v
        if t > 0.0:
            if sign > 0:
                self.tiles[r].append(t)
            else:
                try:
                    self.tiles[r].remove(t)
                except ValueError:
                    pass

    def u(self, r, tau, dmu=0.0, dv=0.0, hot=None):
        h = max(self.tiles[r]) if hot is None and self.tiles[r] else (hot or 0.0)
        return self.mu[r] + dmu + self.m.ch * h + (self.v[r] + dv) / (2.0 * tau)


def lse(us, tau):
    mx = max(us)
    return mx + tau * math.log(sum(math.exp((x - mx) / tau) for x in us))


def default_tau(m: Model, st: State) -> float:
    """sigma_med / sqrt(2 ln R): the bound's optimum for R equal ranks (floored at 0.5 us, the shipped tau)."""
    if m.tau is not None:
        return m.tau
    sd = sorted(math.sqrt(max(v, 0.0)) for v in st.v)
    sig = sd[len(sd) // 2]
    return max(0.5, sig / math.sqrt(2.0 * math.log(max(2, st.R))))


# ---------------------------------------------------------------------------------------------------- layer objective
def layer_J(m: Model, rates, phys, weights, R: int, M: int, tau=None):
    """(J, tau) of a layout `phys` (expert per slot, -1 free) with per-slot `weights` (shares) under rates[e] (rows per
    replay, window mean).  Zero-share copies cost nothing; every expert's home (share 1 by default) is charged by
    activity even at zero window load."""
    st = State(m, R)
    for s, e in enumerate(phys):
        if e is None or e < 0:
            continue
        w = weights[s]
        if w > 0.0:
            st.add(s // M, slot_terms(m, rates[e], w))
    tau = tau if tau is not None else default_tau(m, st)
    return lse([st.u(r, tau) for r in range(R)], tau), tau


# ---------------------------------------------------------------------------------------------------- P6 placement
TOP_SRC, TOP_DST = 4, 8


def place(m: Model, rates, R: int, H: int, max_replicas: int, cap: int, min_gain_us: float = 0.05,
          allow_split: bool = True, allow_move: bool = True, keep=(), new_penalty_us: float = 0.0, tau=None):
    """Greedy P6 placement from the identity layout.  rates[e] = window rows per replay of expert e (home rank e // H).
    keep: (expert, rank) replicas the current layout holds (no copy needed); any other replica must beat the others by
    new_penalty_us (its copy, amortised over the layout's life) -- the placement's hysteresis.
    -> dict(plan {e: [rank]}, shares {e: [(rank, share)]} for every expert with a replica (home first), J_identity, J,
    tau, moves [(kind, e, from, to, dJ)], new (replicas not in keep))."""
    keep = frozenset(keep)
    E = len(rates)
    st = State(m, R)
    home_terms = [None] * E
    for e in range(E):
        home_terms[e] = slot_terms(m, rates[e], 1.0)
        st.add(e // H, home_terms[e])
    tau = tau if tau is not None else default_tau(m, st)
    u = [st.u(r, tau) for r in range(R)]
    J0 = lse(u, tau)
    J = J0
    mx = max(u)
    Z = sum(math.exp((x - mx) / tau) for x in u)
    nrep, rep_on = 0, [0] * R
    moved = set()
    shares, plan, moves = {}, {}, []
    while nrep < max_replicas:
        order = sorted(range(R), key=lambda r: -u[r])
        srcs = order[:TOP_SRC]
        dsts = [r for r in reversed(order) if rep_on[r] < cap][:TOP_DST]
        if not dsts:
            break
        best = None  # (newJ, kind, e, a, b, terms_a_new, terms_b_new)
        for a in srcs:
            ea = math.exp((u[a] - mx) / tau)
            for e in range(a * H, (a + 1) * H):
                if e in moved:
                    continue
                lam = rates[e]
                opts = []
                if allow_move:
                    opts.append(("move", None, slot_terms(m, lam, 1.0)))
                if allow_split and lam > m.fill:
                    hs = m.fill / lam
                    opts.append(("split", slot_terms(m, lam, hs), slot_terms(m, lam, 1.0 - hs)))
                for kind, ta, tb in opts:
                    # rank a without e's home terms (+ the kept home share for a split)
                    st.add(a, home_terms[e], -1.0)
                    if ta is not None:
                        st.add(a, ta)
                    ua = st.u(a, tau)
                    if ta is not None:
                        st.add(a, ta, -1.0)
                    st.add(a, home_terms[e])
                    if ua >= u[a] - 1e-12:
                        continue
                    for b in dsts:
                        if b == a:
                            continue
                        st.add(b, tb)
                        ub = st.u(b, tau)
                        st.add(b, tb, -1.0)
                        eb = math.exp((u[b] - mx) / tau)
                        z = Z - ea - eb + math.exp((ua - mx) / tau) + math.exp((ub - mx) / tau)
                        nJ = mx + tau * math.log(z) if z > 0 else -math.inf
                        pen = 0.0 if (e, b) in keep else new_penalty_us
                        if best is None or nJ + pen < best[0] + best[7]:
                            best = (nJ, kind, e, a, b, ta, tb, pen)
        if best is None or J - best[0] - best[7] < min_gain_us:
            break
        nJ, kind, e, a, b, ta, tb, pen = best
        st.add(a, home_terms[e], -1.0)
        if ta is not None:
            st.add(a, ta)
        st.add(b, tb)
        moved.add(e)
        nrep += 1
        rep_on[b] += 1
        plan.setdefault(e, []).append(b)
        hs = (m.fill / rates[e]) if kind == "split" else 0.0
        shares[e] = [(a, hs), (b, 1.0 - hs)]
        moves.append((kind, e, a, b, J - nJ))
        u = [st.u(r, tau) for r in range(R)]
        mx = max(u)
        Z = sum(math.exp((x - mx) / tau) for x in u)
        J = mx + tau * math.log(Z)
    new = sum(1 for e, rs in plan.items() for b in rs if (e, b) not in keep)
    return dict(plan=plan, shares=shares, J_identity=J0, J=J, tau=tau, moves=moves, new=new)


def identity_tau(m: Model, rates, R: int, H: int) -> float:
    """The layer's tau from the identity (home-only) layout: one tau per layer and renewal for every layout compared."""
    st = State(m, R)
    for e in range(len(rates)):
        st.add(e // H, slot_terms(m, rates[e], 1.0))
    return default_tau(m, st)


# ---------------------------------------------------------------------------------------------------- P6 flex placement
def flex(m: Model, rates, R: int, H: int, max_replicas: int, cap: int, per_rank: int = 2, held=None,
         keep_bonus: float = 0.25):
    """P6 'flex' slow level: OPTIONS for the fast level's whole-expert moves.  Every rank's `per_rank` most loaded homes
    (by window rate, x (1 + keep_bonus) for one that already has a replica: hysteresis) get one replica each; ranks are
    served in the order of their expected cost u_r (busiest first) while replicas last; an expert that already has a
    replica keeps its destination (no copy); a new one goes to the rank of lowest expected cost with room (never its
    home; + cf / 2 per replica assigned, which spreads them).  The fast level (solve_shares / l3b_core.c pricing 2 with
    0/1 shares at q = 1) then decides at every renewal which copy serves -- the replica moves the expert's weight stream
    off a busy home rank without any copy, and moves it back when that rank is no longer busy.
    held: {expert: rank of its current replica}.  -> {expert: [rank]}"""
    held = held or {}
    st = State(m, R)
    for e in range(len(rates)):
        st.add(e // H, slot_terms(m, rates[e], 1.0))
    tau = default_tau(m, st)
    u = [st.u(r, tau) for r in range(R)]
    cand = []
    for r in range(R):
        score = lambda e: rates[e] * ((1.0 + keep_bonus) if e in held else 1.0)
        for e in sorted(range(r * H, (r + 1) * H), key=lambda e: (-score(e), e))[:per_rank]:
            cand.append((u[r], e))
    cand.sort(key=lambda t: (-t[0], t[1]))
    load, on, plan, n = list(u), [0] * R, {}, 0
    for _, e in cand:
        if n >= max_replicas:
            break
        a = e // H
        b = held.get(e)
        if b is None or b == a or on[b] >= cap:
            ds = [x for x in range(R) if x != a and on[x] < cap]
            if not ds:
                continue
            b = min(ds, key=lambda x: (load[x], x))
        plan.setdefault(e, []).append(b)
        on[b] += 1
        n += 1
        load[b] += 0.5 * m.cf
    return plan


# ---------------------------------------------------------------------------------------------------- P5 shares on a layout
def solve_shares(m: Model, rates, phys, R: int, M: int, H: int, sweeps: int = 3, tau=None):
    """Shares of every slot of `phys` by coordinate descent on J over the replicated experts' options (all rows on one
    copy, or a one-tile fill on the home and the rest on one other copy); non-replicated experts keep share 1 on their
    only copy.  -> (weights per slot, J, J of the home-only shares, tau)."""
    E = len(rates)
    copies = {}
    for s, e in enumerate(phys):
        if e is not None and e >= 0:
            copies.setdefault(e, []).append(s)
    w = [0.0] * len(phys)
    st = State(m, R)
    rep = []
    for e, ss in copies.items():
        hs = (e // H) * M + e % H
        if hs not in ss:
            hs = ss[0]
        w[hs] = 1.0
        st.add(hs // M, slot_terms(m, rates[e], 1.0))
        if len(ss) > 1:
            rep.append((e, hs, [s for s in ss if s != hs]))
    tau = tau if tau is not None else default_tau(m, st)
    J_home = lse([st.u(r, tau) for r in range(R)], tau)
    cur = {e: [(hs, 1.0)] for e, hs, _ in rep}
    J = J_home
    for _ in range(sweeps):
        improved = False
        for e, hs, others in sorted(rep, key=lambda t: -rates[t[0]]):
            lam = rates[e]
            opts = [[(hs, 1.0)]] + [[(s, 1.0)] for s in others]
            if lam > m.fill:
                f = m.fill / lam
                opts += [[(hs, f), (s, 1.0 - f)] for s in others]
            for s, sh in cur[e]:
                st.add(s // M, slot_terms(m, lam, sh), -1.0)
            bestJ, bestO = None, None
            for o in opts:
                for s, sh in o:
                    st.add(s // M, slot_terms(m, lam, sh))
                Jo = lse([st.u(r, tau) for r in range(R)], tau)
                for s, sh in o:
                    st.add(s // M, slot_terms(m, lam, sh), -1.0)
                if bestJ is None or Jo < bestJ - 1e-12:
                    bestJ, bestO = Jo, o
            for s, sh in bestO:
                st.add(s // M, slot_terms(m, lam, sh))
            if bestO != cur[e]:
                improved = True
            cur[e] = bestO
            J = bestJ
        if not improved:
            break
    for e, hs, others in rep:
        for s in [hs] + others:
            w[s] = 0.0
        for s, sh in cur[e]:
            w[s] = sh
    return w, J, J_home, tau
