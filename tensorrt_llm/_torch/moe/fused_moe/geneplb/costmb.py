# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB's warm-up cost-model micro-benchmark: cost model B's kernel terms
(cf, cl, cc, ch, a, tau) measured on this run's platform, geometry and the MegaMoE tactic the decode graphs capture.

WHERE  balancer.before_capture, called by DecoderCudaGraphs.warmup_and_capture (graphs.py) between the CUDA-graph warmup's pass 1
       (every decode bucket launched eagerly: tactic resolved, (local, shared) workspace pairing fenced) and pass 2 (capture), in the
       final-executor warmup only, once per process; balancer.after_capture re-checks the captured tactic. Collective on every GEN
       rank; MPI only (a Dup of the balancer's done communicator), never torch.distributed (NCCL here).
WHAT   per T tier (tokens per rank per decode step, the balancer's T): synthetic per-rank slot loads in six design families
       (make_designs), realised as real routings (T tokens per source rank, K distinct slots per token), launched through the real
       backends of the first staged layers in turn (set_adaptive_launch_tokens(T) + run_moe with physical slot ids rank * M + local
       slot, the B route stager off, under inference mode: the pass-1 staging buffers, symmetric regions and fenced pairings);
       per (design, rep) block: device sleep, warm launches, CUDA event, timed launches, CUDA event; y = per block the MIN over
       ranks of the per-launch time, the median over reps.
FIT    c_r = cf F + cl Lt + cc R + ch H (F active slots, Lt tiles after the first, R rows, H hottest-slot tiles - 1; tile = the
       token tile of the resolved tactic), y = a + tau logsumexp_r(c_r / tau); every candidate (base / hot at the hard max, hot at
       each LSE tau) fitted on the train designs only from cold starts; the lowest TRUE hold-out RMSE (every 4th design per
       family), then the smallest-tau LSE model within the tier's select_tol; that model refitted on all designs. Accepted when the
       hold-out RMSE <= max(3 us, 5 % of the mean y) and every coefficient is inside its bound; a tier where no design crosses a
       tile pins cl / ch (nearest accepted identifiable tier of its bucket, else its fallback row).
VOTES  (A) identity (geometry from the balancer's Geometry, buckets, per tier: bucket, tactic label, tile, fence liveness -- what the
       op's forward would resolve, read without launching), (B) designs (built in parts, all-gathered), (C) inputs (built and
       validated) with the design shas -- every check that can fail runs before the first MegaMoE launch (its NVLink barrier has
       no timeout); then the fixed warm launches, the all-gathered host / launch ms, the rank-identical sleep and wall cap, the
       timing loops, the all-gathered times; rank 0 fits and broadcasts the sealed table (pickle: bit-exact); every rank re-seals
       and compares its own y shas; the table sha is all-gathered. Any failure or difference anywhere: EVERY rank takes the same
       sealed fallback table (costmodel.fallback_table: prior v2's kernel terms of the geometry). The balancer finally all-gathers
       the table sha once more and raises on a difference (a differing fallback is a broken install, never a silent split).
Standard library + numpy at import (the CPU core also runs without a GPU); torch / TensorRT-LLM are reached through the GPU
environment passed in (run / check_capture build it).
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import costmodel
except ImportError:  # standalone (CPU-only tests)
    import costmodel

HERE = Path(__file__).resolve().parent
CONF_FILE = HERE / "data" / "costmb.json"
TAG = "[geneplb-costmb]"
PROTOCOL = "geneplb_costmb_v1"
ENV_OVERRIDES = (
    "TLLM_GENEPLB_COSTMB_TIERS",
    "TLLM_GENEPLB_COSTMB_BAR_US",
    "TLLM_GENEPLB_COSTMB_BAR_REL",
    "TLLM_GENEPLB_COSTMB_CAP_S",
    "TLLM_GENEPLB_COSTMB_PER_FAMILY",
    "TLLM_GENEPLB_COSTMB_REPS",
)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def _err(exc) -> str:
    return ("%s: %s" % (type(exc).__name__, exc))[:400]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(costmodel.canonical(obj), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ============================================================================================================================ config
def load_conf(env=None) -> dict:
    """data/costmb.json with the TLLM_GENEPLB_COSTMB_* overrides applied; conf['sha256'] = the sha of the effective values."""
    env = os.environ if env is None else env
    conf = json.loads(CONF_FILE.read_text())
    ov = {}

    def get(name):
        v = str(env.get(name, "")).strip()
        if v:
            ov[name] = v
        return v or None

    v = get("TLLM_GENEPLB_COSTMB_TIERS")
    if v:
        tiers = []
        for tok in v.replace(";", ",").split(","):
            if tok.strip() and int(tok) not in tiers:
                tiers.append(int(tok))
        conf["tiers"]["override"] = tiers
    for name, sec, key, kind in (
        ("TLLM_GENEPLB_COSTMB_BAR_US", "accept", "bar_abs_us", float),
        ("TLLM_GENEPLB_COSTMB_BAR_REL", "accept", "bar_rel", float),
        ("TLLM_GENEPLB_COSTMB_CAP_S", "timing", "cap_s", float),
        ("TLLM_GENEPLB_COSTMB_PER_FAMILY", "designs", "per_family", int),
        ("TLLM_GENEPLB_COSTMB_REPS", "timing", "reps", int),
    ):
        v = get(name)
        if v is not None:
            conf[sec][key] = kind(v)
    need(int(conf["designs"]["per_family"]) >= int(conf["designs"]["holdout_every"]) >= 2,
         "per_family %s: at least holdout_every designs per family (a true hold-out)" % conf["designs"]["per_family"])
    need(int(conf["timing"]["reps"]) >= 1 and int(conf["timing"]["layers"]) >= 1, "timing reps / layers >= 1")
    need(all(int(t) >= 1 for t in conf["tiers"].get("override", [])), "tiers are positive")
    conf["env"] = ov
    body = {k: v for k, v in conf.items() if k not in ("sha256",)}
    conf["sha256"] = _sha(body)
    return conf


def select_tiers(conf: dict, ranks: int, graph: dict, buckets) -> list:
    """The T tiers to measure (configured order = priority under the wall cap): the override as given; else the EP's production
    tiers up to the largest decode graph's T; else the graphs' T values >= 8 thinned to default_max_tiers."""
    tc = conf["tiers"]
    if tc.get("override"):
        return [int(t) for t in tc["override"]]
    tps = max(1, int(graph.get("tokens_per_step") or 1))
    gts = sorted({int(b) * tps for b in graph.get("graph_batch_sizes") or () if int(b) >= 1})
    top = max(gts) if gts else None
    fixed = tc.get("ep%d" % int(ranks))
    if fixed:
        return [int(t) for t in fixed if top is None or int(t) <= top]
    cand = [t for t in gts if t >= 8]
    k = int(tc.get("default_max_tiers", 6))
    if len(cand) > k:
        cand = [cand[round(i * (len(cand) - 1) / (k - 1))] for i in range(k)] if k > 1 else cand[-1:]
    return sorted(set(cand))


def select_tol(T: int, conf: dict) -> float:
    fc = conf["fit"]
    return float(fc["select_tol"].get(str(costmodel.tier_of(int(T))), fc["select_tol_default"]))


# ============================================================================================================================ designs
# Port of the reference warm-up micro-benchmark of cost model B: the same arithmetic in the same order (its golden
# design shas pass); one change in _fix_total (marked).
class Geom:
    """EP ranks, M = home + S local slots per rank, T tokens per source rank, K routes per token, token tile."""

    def __init__(self, ep, home, slots, tokens, topk, tile):
        self.ep, self.home, self.M, self.T, self.K, self.tile = int(ep), int(home), int(slots), int(tokens), int(topk), int(tile)
        self.S = self.M - self.home
        self.ntok = self.ep * self.T
        self.total = self.ntok * self.K
        self.cap = self.ntok
        self.mean = self.T * self.K
        self.maxq = -(-self.cap // self.tile)

    def as_dict(self):
        return dict(ep=self.ep, home=self.home, M=self.M, S=self.S, T=self.T, K=self.K, tile=self.tile)


def _hot_first(row):
    return -np.sort(-np.asarray(row))


def _fix_total(x, g, rng):
    """scale x [EP, M] (active slots stay >= 1, <= cap) and correct the rounding on random active slots so that x.sum() == total"""
    x = np.asarray(x, np.float64)
    act = x > 0
    if int(act.sum()) > g.total:  # more active slots than routes (small T): keep the `total` largest
        flat = x.reshape(-1).copy()
        drop = np.argsort(-flat, kind="stable")[g.total:]
        flat[drop] = 0.0
        x = flat.reshape(x.shape)
        act = x > 0
    x = np.where(act, np.maximum(1.0, x * g.total / x.sum()), 0.0)
    x = np.minimum(np.floor(x), g.cap)
    x = np.where(act, np.maximum(x, 1), 0).astype(np.int64)
    d = int(g.total - x.sum())
    flat = x.reshape(-1)
    while d != 0:
        idx = np.flatnonzero(flat > (1 if d < 0 else 0))
        if d > 0:
            idx = idx[flat[idx] < g.cap]
        if idx.size == 0:
            raise ValueError("fix_total: no slot left to correct the row total")
        pick = rng.choice(idx, size=min(abs(d), idx.size), replace=False)
        flat[pick] += 1 if d > 0 else -1
        d = int(g.total - flat.sum())
    return flat.reshape(x.shape)


def _rank_row(F, R, M, rng, hot=(), disp=0.6):
    """one rank's loads: F active slots summing to ~R (the hot slots get their row counts first), hot-first order"""
    F = int(max(1, min(M, F)))
    hot = [int(h) for h in hot][:F]
    row = np.zeros(M)
    rest = max(F - len(hot), 0)
    rem = max(R - sum(hot), rest)
    w = rng.lognormal(0.0, disp, rest) if rest else np.zeros(0)
    vals = list(hot) + (list(np.maximum(1.0, rem * w / w.sum())) if rest else [])
    row[: len(vals)] = vals
    return _hot_first(row)


def _balance(x, g, beta, rng):
    """water-filling into the helper slots (beta 0: no move, an A-like split; beta 1: rows level, a C-like split)"""
    x = x.copy()
    if g.S <= 0 or beta <= 0:
        return x
    rows = x.sum(1)
    target = g.mean + (1.0 - beta) * (rows.max() - g.mean)
    free = [list(range(g.home, g.M)) for _ in range(g.ep)]
    for _ in range(g.ep * g.S):
        r1 = int(np.argmax(rows))
        if rows[r1] <= target + 0.5:
            break
        cands = [r for r in range(g.ep) if free[r] and r != r1]
        if not cands:
            break
        r2 = min(cands, key=lambda r: rows[r])
        s1 = int(np.argmax(x[r1]))
        m = math.floor(min(x[r1, s1] - 1, rows[r1] - target, target - rows[r2]) * rng.uniform(0.7, 1.0))
        if m < 1:
            break
        h = free[r2].pop(0)
        x[r1, s1] -= m
        x[r2, h] += m
        rows[r1] -= m
        rows[r2] += m
    return x


def _pop(g, rng, beta):
    """a decode-like routing: expert popularity ~ power law, up to 3 super-hot experts, experts on their home slots, then a partial
    helper-slot balance of strength beta"""
    E = g.ep * g.home
    alpha = float(rng.uniform(0.5, 1.3))
    w = (np.arange(E) + 1.0) ** -alpha * rng.lognormal(0.0, 0.3, E)
    w = w[rng.permutation(E)]
    nsh = int(rng.choice([0, 0, 1, 2, 3]))
    hot_e = rng.choice(E, size=nsh, replace=False) if nsh else np.zeros(0, np.int64)
    load = np.zeros(E, np.int64)
    for e in hot_e:
        load[e] = int(g.cap * rng.uniform(0.3, 0.97))
    free = np.ones(E, bool)
    free[hot_e] = False
    load[free] = rng.multinomial(g.total - int(load.sum()), w[free] / w[free].sum())
    over = int(np.maximum(load - g.cap, 0).sum())
    load = np.minimum(load, g.cap)
    while over > 0:
        room = np.flatnonzero((load < g.cap) & free)
        add = np.minimum(rng.multinomial(over, w[room] / w[room].sum()), g.cap - load[room])
        load[room] += add
        over -= int(add.sum())
    x = np.zeros((g.ep, g.M))
    x[:, : g.home] = load.reshape(g.ep, g.home)
    return _balance(x, g, beta, rng), dict(alpha=round(alpha, 3), beta=round(float(beta), 3), superhot=nsh)


def design(fam, g, rng):
    """slot loads x [EP, M] (sum = EP T K, each <= EP T, hot-first per rank) of one design of family fam"""
    info = {"family": fam}
    if fam == "pop":
        x, i = _pop(g, rng, float(rng.uniform(0.0, 1.0)))
        info.update(i)
    elif fam == "act":
        x, i = _pop(g, rng, float(rng.uniform(0.0, 1.0)))
        info.update(i)
        for r in range(g.ep):
            if rng.random() < 0.6:
                free = np.flatnonzero(x[r] == 0)
                n = int(rng.integers(1, min(10, free.size) + 1)) if free.size else 0
                for s in rng.choice(free, size=n, replace=False) if n else []:
                    if x[r].max() < 2:
                        break
                    take = float(max(1, min(int(rng.integers(1, 17) * max(1, g.T // 128)), x[r].max() - 1)))
                    x[r, np.argmax(x[r])] -= take
                    x[r, s] = take
    elif fam == "cons":
        x, i = _pop(g, rng, float(rng.uniform(0.5, 1.0)))
        info.update(i)
        for r in range(g.ep):
            if rng.random() < 0.6:
                act = np.flatnonzero(x[r] > 0)
                m = int(rng.integers(1, min(8, act.size - 1) + 1)) if act.size > 1 else 0
                small = act[np.argsort(x[r, act], kind="stable")[:m]]
                big = act[np.argmax(x[r, act])]
                if m and x[r, big] + x[r, small].sum() <= g.cap:
                    x[r, big] += x[r, small].sum()
                    x[r, small] = 0
    elif fam == "ties":
        k = int(rng.choice([v for v in (1, 2, 4, 8, g.ep // 2, g.ep) if 1 <= v <= g.ep]))
        F0 = int(rng.integers(max(2, g.M // 2), g.M + 1))
        R0 = float(g.mean * rng.uniform(1.0, 1.7 if k <= max(1, g.ep // 4) else 1.0))
        H0 = (
            [int(rng.integers(g.tile + 1, min(g.cap, g.tile * int(rng.integers(2, 6))) + 1))]
            if (g.maxq >= 2 and rng.random() < 0.5)
            else []
        )
        base = _rank_row(F0, R0, g.M, rng, H0)
        x = np.zeros((g.ep, g.M))
        tied = rng.choice(g.ep, size=k, replace=False)
        x[tied] = base
        tied_set = set(tied.tolist())
        others = [r for r in range(g.ep) if r not in tied_set]
        if others:
            Rb = max((g.total - k * base.sum()) / len(others), 4.0)
            for r in others:
                x[r] = _rank_row(max(4, F0 - int(rng.integers(2, 11))), Rb * rng.uniform(0.85, 1.0), g.M, rng)
        info.update(k=k, F0=F0)
    elif fam == "fswp":
        x = np.stack(
            [_rank_row(int(rng.integers(min(4, g.M), g.M + 1)), g.mean * rng.uniform(0.5, 1.8), g.M, rng, disp=0.25)
             for _ in range(g.ep)]
        )
    elif fam == "hot":
        nh = int(rng.integers(1, max(2, g.ep // 4) + 1))
        hot_r = rng.choice(g.ep, size=nh, replace=False).tolist()
        x = np.zeros((g.ep, g.M))
        used, budget = 0.0, 0.6 * g.total
        for r in hot_r:
            h = []
            if g.maxq >= 2:
                h = [min(g.cap, int(rng.integers(2, min(12, g.maxq) + 1)) * g.tile - int(rng.integers(0, g.tile // 2)))
                     for _ in range(int(rng.integers(1, 4)))]
            while h and (used + sum(h) > budget or sum(h) > 5 * g.mean):
                h.pop()
            x[r] = _rank_row(int(rng.integers(max(len(h) + 1, g.M // 2), g.M + 1)), sum(h) + g.mean * rng.uniform(0.3, 1.0),
                             g.M, rng, h)
            used += x[r].sum()
        others = [r for r in range(g.ep) if r not in set(hot_r)]
        Rb = max((g.total - used) / max(len(others), 1), 4.0)
        for r in others:
            x[r] = _rank_row(int(rng.integers(max(2, g.M // 2), g.M + 1)), Rb * rng.uniform(0.85, 1.0), g.M, rng)
        info.update(hot_ranks=nh)
    else:
        raise ValueError(fam)
    x = _fix_total(x, g, rng)
    return np.stack([_hot_first(r) for r in x]).astype(np.int64), info


def assign(x, g, rng):
    """slot loads x [EP * M] -> slots [EP, T, K]: largest slot first, each slot to the tokens with the most routes left (random
    tie-break) -- the bipartite Havel-Hakimi order, so every token ends with K distinct slots (None if it got stuck)"""
    NT = g.ntok
    rem = np.full(NT, g.K, np.int64)
    out = np.empty((NT, g.K), np.int64)
    fill = np.zeros(NT, np.int64)
    for s in np.argsort(-x, kind="stable"):
        n = int(x[s])
        if n == 0:
            break
        key = rem + rng.random(NT) * 0.5
        idx = np.argpartition(-key, n - 1)[:n]
        if (rem[idx] <= 0).any():
            return None
        out[idx, fill[idx]] = s
        fill[idx] += 1
        rem[idx] -= 1
    return out.reshape(g.ep, g.T, g.K) if (rem == 0).all() else None


def loads_of(slots, g):
    return np.bincount(np.asarray(slots, np.int64).ravel(), minlength=g.ep * g.M).reshape(g.ep, g.M)


def make_designs(g, conf, part=None):
    """The designs of one tier: [dict(family, index, x [EP, M] int64, slots [EP, T, K] int16 | int32, info)], deterministic in
    (seed, geometry); part = (j, n): only the list positions = j mod n (the ranks build the parts, all-gathered, merge_parts)."""
    d = conf["designs"]
    seed, per_family, families = int(d["seed"]), int(d["per_family"]), tuple(d["families"])
    out = []
    for fi, fam in enumerate(families):
        for i in range(per_family):
            if part is not None and (fi * per_family + i) % part[1] != part[0]:
                continue
            rng = np.random.default_rng([seed, fi, i, g.ep, g.M, g.T, g.K, g.tile])
            for attempt in range(int(d["attempts"])):
                x, info = design(fam, g, rng)
                sl = assign(x.reshape(-1), g, rng)
                if sl is not None:
                    break
            else:
                raise RuntimeError("%s #%d: no realisable design in %d attempts" % (fam, i, d["attempts"]))
            need((loads_of(sl, g) == x).all(), "design %s #%d: the routing does not realise its loads" % (fam, i))
            s = np.sort(sl.reshape(-1, g.K), axis=1)
            need((s[:, 1:] != s[:, :-1]).all(), "design %s #%d: a token has two routes to one slot" % (fam, i))
            info["attempts"] = attempt + 1
            out.append(dict(family=fam, index=i, x=x, slots=sl.astype(np.int16 if g.ep * g.M < 32768 else np.int32), info=info))
    return out


def merge_parts(parts):
    """The per-rank parts of make_designs(..., part=(j, n)) back in list order (part j holds list positions j, j + n, ...)."""
    n = len(parts)
    total = sum(len(p) for p in parts)
    return [parts[j % n][j // n] for j in range(total)]


def designs_sha(designs):
    h = hashlib.sha256()
    for d in designs:
        h.update(("%s:%d:" % (d["family"], d["index"])).encode())
        h.update(np.ascontiguousarray(d["slots"]).tobytes())
    return h.hexdigest()


def holdout_mask(designs, conf):
    every = int(conf["designs"]["holdout_every"])
    return np.array([d["index"] % every == every - 1 for d in designs], bool)


# ============================================================================================================================ fit
def features(x, tile):
    """x [..., EP, M] slot rows -> [..., EP, 4]: F (active slots), Lt (tiles after the first), R (rows), H (hottest-slot tiles - 1)"""
    x = np.asarray(x, np.float64)
    q = np.ceil(x / tile)
    return np.stack([(q >= 1).sum(-1), np.maximum(q - 1, 0).sum(-1), x.sum(-1), np.maximum(q.max(-1) - 1, 0)], -1)


def _forms(rows_unit):
    u = float(rows_unit)
    return {
        "base": (lambda f: np.stack([f[..., 0], f[..., 1], f[..., 2] / u], -1), ["cf", "cl", "cc_per_unit"]),
        "hot": (lambda f: np.stack([f[..., 0], f[..., 1], f[..., 2] / u, f[..., 3]], -1), ["cf", "cl", "cc_per_unit", "ch"]),
    }


def predict(Z, w, a, tau):
    """Z [n, EP, d], w [d], intercept a, tau -> (y_hat [n], softmax weights [n, EP])"""
    c = Z @ w
    m = c.max(1, keepdims=True)
    e = np.exp((c - m) / tau)
    s = e.sum(1, keepdims=True)
    return a + (m[:, 0] + tau * np.log(s[:, 0])), e / s


def fit(Z, y, tau, w0, iters):
    """damped Gauss-Newton with w >= 0 (projected) on sum (y - a - tau logsumexp(Z w / tau))^2"""
    w = np.maximum(np.asarray(w0, np.float64), 0.0)
    a = float(np.median(y - predict(Z, w, 0.0, tau)[0]))
    lam, prev = 1e-3, np.inf
    for _ in range(iters):
        yh, pi = predict(Z, w, a, tau)
        res = y - yh
        sse = float(res @ res)
        G = np.einsum("ne,ned->nd", pi, Z)
        J = np.concatenate([np.ones((len(y), 1)), G], 1)
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), J.T @ res)
        a_n, w_n = a + float(step[0]), np.maximum(w + step[1:], 0.0)
        sse_n = float(((y - predict(Z, w_n, a_n, tau)[0]) ** 2).sum())
        if sse_n < sse:
            a, w, lam = a_n, w_n, lam * 0.3
            if prev - sse_n < 1e-9 * max(prev, 1.0):
                break
            prev = sse_n
        else:
            lam *= 10.0
            if lam > 1e8:
                break
    return w, a


def _rmse(e):
    e = np.asarray(e, np.float64)
    return float(np.sqrt(np.mean(np.square(e)))) if e.size else float("nan")


def fit_models(feat, y, hold, tol, conf):
    """feat [D, EP, 4], y [D] us, hold [D] bool (the true hold-out) -> dict(models, selected, final): every candidate fitted on the
    train designs from the cold starts (the reference fitter's constants), the lowest hold-out RMSE,
    then the smallest-tau LSE model within tol of it, refitted on all designs (warm start: the final parameters)."""
    c = conf["fit"]
    taus = [float(t) for t in c["taus"]]
    hard = taus[0]
    y = np.asarray(y, np.float64)
    hold = np.asarray(hold, bool)
    tr = ~hold if (~hold).sum() >= 4 else np.ones_like(hold)
    forms = _forms(c["rows_unit"])
    models = {}
    for name in ("base", "hot"):
        fz, names = forms[name]
        Z = fz(feat)
        d = Z.shape[-1]
        starts = [np.full(d, 1.0), np.full(d, 3.0)] + [np.random.default_rng(k).uniform(0.2, 4.0, d) for k in range(int(c["starts"]) - 2)]
        best, lse_best = None, {}
        for w0 in starts:  # homotopy in tau (smooth -> hard max) from every start
            w = w0
            for tau in taus[::-1]:
                w, a = fit(Z[tr], y[tr], tau, w, int(c["iters"]))
                if name == "hot" and tau != hard:
                    sse = float(((y[tr] - predict(Z[tr], w, a, tau)[0]) ** 2).sum())
                    if tau not in lse_best or sse < lse_best[tau][0]:
                        lse_best[tau] = (sse, w, a)
            sse = float(((y[tr] - predict(Z[tr], w, a, hard)[0]) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, w, a)
        cand = [(name, hard, best[1], best[2])] + [("lse%g" % tau, tau, w, a) for tau, (_, w, a) in sorted(lse_best.items())]
        for key, tau, w, a in cand:
            yh = predict(Z, w, a, tau)[0]
            models[key] = dict(form=name, tau=float(tau), w=[float(v) for v in w], names=names, a=float(a),
                               train_rmse_us=_rmse((y - yh)[tr]),
                               holdout_rmse_us=_rmse((y - yh)[hold & ~tr]) if (hold & ~tr).any() else None)
    crit = "holdout_rmse_us" if (hold & ~tr).any() else "train_rmse_us"
    sel = min(models, key=lambda k: (models[k][crit], k))
    near = sorted((k for k in models if models[k]["tau"] > float(c["tau_hard"])
                   and models[k][crit] <= models[sel][crit] * (1.0 + float(tol))), key=lambda k: models[k]["tau"])
    if near:
        sel = near[0]
    m = models[sel]
    Z = forms[m["form"]][0](feat)
    w, a = fit(Z, y, m["tau"], np.asarray(m["w"]), int(c["iters"]))
    yh = predict(Z, w, a, m["tau"])[0]
    final = dict(model=sel, form=m["form"], tau=m["tau"], w=[float(v) for v in w], names=m["names"], a=float(a),
                 all_rmse_us=_rmse(y - yh), holdout_rmse_us=m["holdout_rmse_us"], train_rmse_us=m["train_rmse_us"], criterion=crit)
    return dict(models=models, selected=sel, final=final)


def accepted(row: dict, holdout_rmse, mean_y: float, conf: dict):
    """(ok, why, bar): the hold-out RMSE <= max(bar_abs_us, bar_rel x mean y) and every coefficient inside its bound."""
    acc = conf["accept"]
    bar = max(float(acc["bar_abs_us"]), float(acc["bar_rel"]) * float(mean_y))
    if holdout_rmse is None or not math.isfinite(holdout_rmse):
        return False, "no hold-out error", bar
    if not holdout_rmse <= bar:
        return False, "hold-out rmse %.3f us > bar %.3f us" % (holdout_rmse, bar), bar
    for k, (lo, hi) in sorted(acc["bounds"].items()):
        v = float(row[k])
        if not (math.isfinite(v) and lo <= v <= hi):
            return False, "%s %.6g outside [%g, %g]" % (k, v, lo, hi), bar
    for k in ("tau", "a"):
        if not math.isfinite(float(row[k])):
            return False, "%s not finite" % k, bar
    return True, None, bar


def y_sha(y) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(y, np.float64)).tobytes()).hexdigest()


def fit_table(*, tiers, measured, designs, y, info, fb, conf, row_meta):
    """Rank 0: every measured tier's fit (identifiable tiers first, so a tier that cannot identify cl / ch is pinned to the nearest
    accepted identifiable tier of its bucket, else to its fallback row) -> dict(table, fits). A rejected or unmeasured tier keeps
    its fallback row (fb's)."""
    fc = conf["fit"]
    fb_rows = {int(k): r for k, r in fb["rows"].items()}
    rows = dict(fb_rows)
    outcome = {int(T): "fallback" for T in tiers}
    fits = {}

    def crossing(T):
        X = np.stack([d["x"] for d in designs[T]])
        return int((X > int(info[T]["tile"])).any(axis=(1, 2)).sum())

    cross = {T: crossing(T) for T in measured}
    pin_min = int(fc["pin_min_crossing_designs"])
    ident_ok = {}  # T -> (cl, ch) of an accepted identifiable fit
    for T in sorted(measured, key=lambda t: (cross[t] < pin_min, t)):
        ds = designs[T]
        tile = int(info[T]["tile"])
        X = np.stack([d["x"] for d in ds])
        hold = holdout_mask(ds, conf)
        res = fit_models(features(X, tile), y[T], hold, select_tol(T, conf), conf)
        fin = res["final"]
        w = dict(zip(fin["names"], fin["w"]))
        cl, ch, pinned, pin_src = float(w["cl"]), float(w.get("ch", 0.0)), [], None
        if cross[T] < pin_min:
            same = [t for t in ident_ok if info[t]["bucket"] == info[T]["bucket"]]
            if same:
                src = min(same, key=lambda t: (abs(math.log2(t) - math.log2(T)), -t))
                cl, ch = ident_ok[src]
                pin_src = "warmup T=%d" % src
            else:
                cl, ch = float(fb_rows[T]["cl"]), float(fb_rows[T]["ch"])
                pin_src = "fallback row (%s)" % fb_rows[T].get("kernel")
            pinned = ["cl", "ch"]
        row = dict(
            schema=costmodel.PARAMS_SCHEMA, source="warmup", kernel="warmup", model=fin["model"], form=fin["form"],
            cf=float(w["cf"]), cl=cl, cc=float(w["cc_per_unit"]) / float(fc["rows_unit"]), ch=ch, tau=float(fin["tau"]),
            tau_hard=costmodel.TAU_HARD, a=float(fin["a"]),
            holdout_rmse_us=fin["holdout_rmse_us"], train_rmse_us=fin["train_rmse_us"], rmse_us=fin["all_rmse_us"],
            n=len(ds), n_holdout=int(hold.sum()), designs_sha256=designs_sha(ds), y_sha256=y_sha(y[T]),
            crossing_designs=cross[T], pinned=pinned, pin_source=pin_src,
            T=int(T), tier=int(T), bucket=int(info[T]["bucket"]), tile_rows=tile, tactic=str(info[T]["tactic"]),
        )
        row.update(row_meta)
        row = costmodel.sealed(row)
        ok, why, bar = accepted(row, fin["holdout_rmse_us"], float(np.mean(y[T])), conf)
        fits[T] = dict(
            outcome="warmup" if ok else "fallback", why=why, bar_us=bar, model=fin["model"], form=fin["form"], tau=fin["tau"],
            holdout_rmse_us=fin["holdout_rmse_us"], train_rmse_us=fin["train_rmse_us"], all_rmse_us=fin["all_rmse_us"],
            mean_y_us=float(np.mean(y[T])), y_min_us=float(np.min(y[T])), y_max_us=float(np.max(y[T])),
            crossing_designs=cross[T], pinned=pinned, pin_source=pin_src, row_sha256=row["sha256"],
            cf=row["cf"], cl=row["cl"], cc=row["cc"], ch=row["ch"], a=row["a"],
            models={k: dict(tau=v["tau"], w=[round(x, 6) for x in v["w"]], a=round(v["a"], 4), train=v["train_rmse_us"],
                            holdout=v["holdout_rmse_us"]) for k, v in res["models"].items()},
        )
        if ok:
            rows[T], outcome[T] = row, "warmup"
            if not pinned:
                ident_ok[T] = (row["cl"], row["ch"])
    return dict(table=costmodel.make_table(rows, outcome, **costmodel.table_meta(fb)), fits=fits)


# ============================================================================================================================ GPU
class GpuEnv:
    """What the GPU half reads: torch, the MegaMoE custom-op module (the fence probe and the capture recorder) and MoERunContext;
    tuning() -> whether the AutoTuner is in tuning mode. Tests pass fakes."""

    def __init__(self, torch, op, Ctx, tuning=None):
        self.torch, self.op, self.Ctx = torch, op, Ctx
        self.tuning = tuning or (lambda: False)

    @classmethod
    def load(cls):
        import torch

        from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op
        from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext

        def tuning():
            from tensorrt_llm._torch.autotuner import AutoTuner

            return bool(AutoTuner.get().is_tuning_mode)

        return cls(torch, op, MoERunContext, tuning)


def tactic_of(op, runner, bucket: int):
    """(label, token tile, kernel, tactic) the runner's forward resolves for a staged launch of the bucket with tactic=-1 (the
    deterministic default of its problem key, num_tokens = the bucket's activation rows; cute_dsl_megamoe_custom_op
    Sm100MegaMoENvfp4Runner.forward)."""
    t = op._default_megamoe_tactic_for_key(runner.unique_id(), num_tokens=int(bucket))
    kernel, t = runner._resolve_kernel_descriptor(t, allow_fallback=True)
    u = tuple(op._unpack_tactic(t))
    return repr(u), int(u[0][1]), kernel, t


def _runner_args_match(a, backend, top_k, bucket=None) -> bool:
    """A _MEGAMOE_RUNNER_CACHE key (_megamoe_get_runner's 19 positional args) of this backend's launches: world, rank, top_k, M,
    hidden, (bucket), combine format, no tactic autotune, no helper slots."""
    try:
        return (
            (int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]))
            == (int(backend.ep_size), int(backend.ep_rank), int(top_k), int(backend.compute_slot_count), int(backend.hidden_size))
            and (bucket is None or int(a[7]) == int(bucket))
            and str(a[16]) == str(backend.combine_format)
            and not bool(a[17])
            and int(a[18]) == 0
        )
    except (TypeError, IndexError, ValueError):
        return False


def probe_bucket(gpu, backend, bucket: int, top_k: int) -> dict:
    """What a launch of ``backend`` in ``bucket`` would resolve, WITHOUT launching: the runner pass 1 cached, its default tactic
    (label, tile) and whether its (local, shared) workspace pairing is fenced and live (the op's forward runs the WORLD-group
    barrier fence otherwise: NCCL, and a RuntimeError once the shared side was captured). why = None: launchable."""
    op, torch = gpu.op, gpu.torch
    out = dict(bucket=int(bucket), runners=0, tactic=None, tile=None, fenced=False, why=None)
    try:
        cache = op._MEGAMOE_RUNNER_CACHE
        hits = [r for a, r in list(cache.items()) if _runner_args_match(a, backend, top_k, bucket)]
        out["runners"] = len(hits)
        if len(hits) != 1:
            out["why"] = "%d cached MegaMoE runners for bucket %d (pass 1 must have launched it, one geometry)" % (len(hits), bucket)
            return out
        runner = hits[0]
        label, tile, kernel, tac = tactic_of(op, runner, bucket)
        out.update(tactic=label, tile=tile)
        local = op._MEGAMOE_LOCAL_WORKSPACE_CACHE.get(runner._tactic_cache_key(tac))
        if local is None:
            out["why"] = "no cached local workspace for bucket %d's tactic" % bucket
            return out
        with torch.inference_mode():
            shared = backend._symm_providers[int(bucket)].get_regions().shared_workspace
        proto = op._megamoe_barrier_reset_slices(kernel)[4]
        fkey = (int(local.data_ptr()), int(shared.data_ptr()), int(backend.ep_size), proto)
        out["fenced"] = bool(op._megamoe_fence_key_live(fkey))
        if not out["fenced"]:
            out["why"] = "bucket %d's workspace pairing is not fenced live (a launch would run the WORLD barrier fence)" % bucket
    except Exception as exc:  # noqa: BLE001 -- voted: the tier is not measured on any rank
        out["why"] = "probe: " + _err(exc)
    return out


def identity(gpu, backends, layer_ids, geo, rank: int, size: int, conf: dict) -> dict:
    """The run's identity (every field but 'rank' must agree across ranks): geometry from the balancer's Geometry (homes = E / EP,
    never backend.expert_size_per_partition, which is M under GenEPLB), the launched layers, the ladder, the configuration and
    prior files, the device. Raises on anything that makes this rank unable to measure."""
    torch = gpu.torch
    need(gpu.op is not None, "the MegaMoE custom-op module is not importable")
    b0 = backends[0]
    need((int(b0.ep_size), int(b0.ep_rank)) == (int(size), int(rank)),
         "EP %s rank %s is not the MPI session's %d / %d (pure DEP expected)" % (b0.ep_size, b0.ep_rank, size, rank))
    M, K, H = int(geo.local_slots), int(geo.top_k), int(geo.homes)
    sig = lambda b: (int(b.ep_size), int(b.ep_rank), int(b.hidden_size), str(b.combine_format),  # noqa: E731
                     [int(v) for v in b._maxt_buckets], str(b._topk_idx_dtype))
    for b in backends:
        need(int(b.compute_slot_count) == M, "the kernel's slot axis %s != GenEPLB's %d local slots" % (b.compute_slot_count, M))
        need(sig(b) == sig(b0), "the launched layers differ in geometry, ladder or index dtype")
        need(not bool(getattr(b, "tactic_autotune", False)), "MEGAMOE_TACTIC_AUTOTUNE=1: the tactic is the AutoTuner's")
        need(getattr(b, "_route_stager", None) is not None, "a launched layer is not staged by cost model B")
    need(not torch.cuda.is_current_stream_capturing(), "called inside a CUDA-graph capture")
    need(not gpu.tuning(), "the AutoTuner is in tuning mode")
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    out = dict(
        protocol=PROTOCOL, rank=int(rank), ep=int(size), M=M, homes=H, K=K, experts=int(geo.experts),
        hidden=int(b0.hidden_size), layers=[int(i) for i in layer_ids], combine_format=str(b0.combine_format),
        buckets=[int(v) for v in b0._maxt_buckets], topk_dtype=str(b0._topk_idx_dtype), conf_sha256=conf["sha256"],
        device=str(getattr(props, "name", "?")), sms=int(getattr(props, "multi_processor_count", 0)),
    )
    out.update(costmodel.prior_files_sha())
    return out


def tier_identity(gpu, backends, T: int, top_k: int, probes: dict) -> dict:
    """Tier T's launch identity on this rank: bucket, tactic label, tile, fence liveness (why = None: measurable)."""
    info = dict(T=int(T), bucket=None, tactic=None, tile=None, fenced=False, runners=0, why=None)
    try:
        bucket = int(backends[0]._select_launch_max_tokens(int(T)))
        need(all(int(b._select_launch_max_tokens(int(T))) == bucket for b in backends), "the layers' buckets differ")
        info["bucket"] = bucket
        if not int(T) < bucket:
            info["why"] = "T %d is not below its bucket %d (run_moe would take the direct-input path B never uses)" % (T, bucket)
            return info
        if bucket not in probes:
            probes[bucket] = probe_bucket(gpu, backends[0], bucket, top_k)
        p = probes[bucket]
        info.update(tactic=p["tactic"], tile=p["tile"], fenced=p["fenced"], runners=p["runners"], why=p["why"])
    except Exception as exc:  # noqa: BLE001 -- voted
        info["why"] = "tier: " + _err(exc)
    return info


def build_inputs(gpu, backends, designs, T: int, top_k: int, hidden: int, rank: int, slots_total: int, conf: dict) -> dict:
    """Every tensor the launches read, made and validated now: the activations (fixed seed per rank) quantized per launched
    backend, the routing weights 1/K, per design this rank's physical slot ids [T, K] in the backend's index dtype."""
    torch = gpu.torch
    dev = torch.device("cuda", torch.cuda.current_device())
    dt = backends[0]._topk_idx_dtype
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(conf["designs"]["seed"]) * 7919 + int(rank))
    x = (torch.randn((T, hidden), generator=gen, device=dev, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    xq = [b.quantize_input(x) for b in backends]
    scales = torch.full((T, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
    slots = []
    for d in designs:
        s = np.ascontiguousarray(d["slots"][rank], dtype=np.int64)
        need(s.shape == (T, top_k) and int(s.min()) >= 0 and int(s.max()) < slots_total,
             "design %s #%d: slot ids out of range / shape %s" % (d["family"], d["index"], s.shape))
        slots.append(torch.from_numpy(s).to(device=dev, dtype=dt))
    out_dtype = getattr(backends[0], "dtype", None) or torch.bfloat16
    torch.cuda.synchronize()
    return dict(backends=backends, T=int(T), x=x, xq=xq, scales=scales, slots=slots, out_dtype=out_dtype, Ctx=gpu.Ctx)


def _launch(inp, layer, d):
    b = inp["backends"][layer]
    b.set_adaptive_launch_tokens(inp["T"])
    b.run_moe(inp["Ctx"](token_selected_experts=inp["slots"][d], token_final_scales=inp["scales"], x=inp["xq"][layer][0],
                         x_sf=inp["xq"][layer][1], output_dtype=inp["out_dtype"]))


def warm_probe(torch, inp, tc):
    """One launch per launched layer, then the host enqueue ms and the device ms per launch of host_probe_launches back-to-back
    launches, then the device clock's cycles per us (as the reference fitter, plus the device ms for the wall cap)."""
    L = len(inp["backends"])
    for layer in range(L):
        _launch(inp, layer, 0)
    torch.cuda.synchronize()
    n = int(tc["host_probe_launches"])
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    h0 = time.perf_counter()
    for i in range(n):
        _launch(inp, i % L, 0)
    host_ms = (time.perf_counter() - h0) * 1e3 / n
    e1.record()
    torch.cuda.synchronize()
    dev_ms = e0.elapsed_time(e1) / n
    best = math.inf
    for _ in range(2):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda._sleep(int(tc["sleep_calibration_cycles"]))
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0)
    return float(host_ms), float(dev_ms), float(tc["sleep_calibration_cycles"]) / max(best, 1e-3)


def time_tier(torch, inp, order, tc, sleep_cycles):
    """Per block of `order`: sleep, warm launches, e0, timed launches, e1 (the launched layers in turn); one sync -> us per launch
    (as the reference fitter)."""
    L = len(inp["backends"])
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in order]
    k = 0
    for (e0, e1), d in zip(events, order):
        torch.cuda._sleep(sleep_cycles)
        for _ in range(int(tc["warm"])):
            _launch(inp, k % L, d)
            k += 1
        e0.record()
        for _ in range(int(tc["timed"])):
            _launch(inp, k % L, d)
            k += 1
        e1.record()
    torch.cuda.synchronize()
    return [e0.elapsed_time(e1) * 1000.0 / int(tc["timed"]) for e0, e1 in events]


def cap_plan(run, n_designs: dict, tc: dict, sleep_ms: float, launch_ms: float):
    """Rank-identical wall cap (all inputs all-gathered / configured): -> ({T: reps}, {T: why dropped}, estimated s). Over the cap
    the reps drop to 1, then the tiers past the cap (configured order) are dropped."""
    block_ms = float(sleep_ms) + (int(tc["warm"]) + int(tc["timed"])) * max(float(launch_ms), 0.0)
    cap, reps = float(tc["cap_s"]), int(tc["reps"])

    def est(r, Ts):
        return sum(r * n_designs[T] * block_ms for T in Ts) / 1e3

    if est(reps, run) <= cap:
        return {T: reps for T in run}, {}, est(reps, run)
    if reps > 1 and est(1, run) <= cap:
        return {T: 1 for T in run}, {}, est(1, run)
    keep, total = [], 0.0
    for T in run:
        e = est(1, [T])
        if total + e <= cap:
            keep.append(T)
            total += e
    why = "wall cap %.0f s (est %.1f s per tier at reps 1: block %.2f ms)" % (cap, est(1, run[:1]) if run else 0.0, block_ms)
    return {T: 1 for T in keep}, {T: why for T in run if T not in keep}, total


# ============================================================================================================================ protocol
class MpiComm:
    """A Dup of an mpi4py communicator: pickled allgather / bcast / barrier (never torch.distributed)."""

    def __init__(self, comm):
        self.c = comm.Dup()
        self.rank, self.size = self.c.Get_rank(), self.c.Get_size()

    def allgather(self, obj):
        return self.c.allgather(obj)

    def bcast(self, obj, root=0):
        return self.c.bcast(obj, root=root)

    def barrier(self):
        self.c.Barrier()

    def free(self):
        self.c.Free()


def empty_table(geo) -> dict:
    """The table every rank takes when the identity vote fails (no tier, a function of the geometry only): every renewal's row is
    then its fallback row (costmodel.b_params_for)."""
    return costmodel.fallback_table(int(geo.ranks), int(geo.local_slots), (), (), homes=int(geo.homes), top_k=int(geo.top_k),
                                    protocol=PROTOCOL)


def measure(*, comm, rank: int, geo, layers, graph: dict, gpu_factory, env=None, keep_detail=False):
    """The collective sequence of the module docstring on this rank -> (table, report); the table is identical on every rank (the
    caller votes its sha once more). Every step that can fail runs between two votes, all of them before the first launch."""
    t_all = time.perf_counter()
    size = comm.size
    rep = dict(rank=int(rank), protocol=PROTOCOL, tiers=[], outcome={}, why={}, wall_s={}, fits={})
    wall = rep["wall_s"]

    def done(table, why=None):
        rep["table_sha256"] = table["sha256"]
        for k, o in (table.get("outcome") or {}).items():
            rep["outcome"][int(k)] = o
        if why:
            rep["why"]["all"] = why
        wall["total"] = round(time.perf_counter() - t_all, 3)
        return table, rep

    # (A) identity, the tiers, every tier's launch identity
    conf = tiers = fb = ident = None
    per_tier, err, gpu, backends = {}, None, None, []
    try:
        gpu = gpu_factory()
        conf = load_conf(env)
        backends = [li.moe.backend for li in layers][: int(conf["timing"]["layers"])]
        need(backends, "no staged MegaMoE layer to launch")
        ident = identity(gpu, backends, [li.index for li in layers[: len(backends)]], geo, rank, size, conf)
        tiers = select_tiers(conf, geo.ranks, graph, ident["buckets"])
        ident["tiers"] = tiers
        fb = costmodel.fallback_table(ident["ep"], ident["M"], tiers, ident["buckets"], int(conf["tile_rows_default"]),
                                      homes=ident["homes"], top_k=ident["K"], protocol=PROTOCOL, conf_sha256=conf["sha256"])
        probes = {}
        per_tier = {int(T): tier_identity(gpu, backends, T, ident["K"], probes) for T in tiers}
    except Exception as exc:  # noqa: BLE001 -- voted: every rank takes the empty fallback table together
        err = _err(exc)
    votes = comm.allgather(dict(rank=int(rank), err=err, fb=None if fb is None else fb["sha256"], tiers=per_tier,
                                ident=None if ident is None else {k: v for k, v in ident.items() if k != "rank"}))
    wall["identity"] = round(time.perf_counter() - t_all, 3)
    rep["identity"] = ident
    errs = sorted({v["err"] for v in votes if v["err"]})
    same = (
        [v["rank"] for v in votes] == list(range(size))
        and len({json.dumps(v["ident"], sort_keys=True, default=str) for v in votes}) == 1
        and len({v["fb"] for v in votes}) == 1
    )
    if errs or not same:
        return done(empty_table(geo), "identity vote: " + ("; ".join(errs) if errs else "the ranks' identity differs")[:400])
    rep["tiers"] = list(tiers)
    excluded, todo = {}, []
    for T in tiers:
        infos = [v["tiers"].get(int(T)) for v in votes]
        if len({json.dumps(i, sort_keys=True, default=str) for i in infos}) != 1:
            excluded[T] = "tier identity differs across ranks"
        elif infos[0]["why"]:
            excluded[T] = infos[0]["why"]
        else:
            todo.append(int(T))
    info = {T: votes[0]["tiers"][T] for T in todo}

    def fallback(why=None):
        for T in tiers:
            rep["why"].setdefault(int(T), excluded.get(T, why))
        return done(fb, why)

    if not todo:
        return fallback("no measurable tier")
    # (B) designs, built in parts and all-gathered
    t0 = time.perf_counter()
    M, K, H, ep = ident["M"], ident["K"], ident["homes"], ident["ep"]
    parts = {}
    for T in todo:
        try:
            parts[T] = (None, make_designs(Geom(ep, H, M, T, K, info[T]["tile"]), conf, part=(rank, size)))
        except Exception as exc:  # noqa: BLE001 -- voted
            parts[T] = (_err(exc), None)
    gathered = comm.allgather(parts)
    designs = {}
    for T in list(todo):
        e = sorted({g[T][0] for g in gathered if g[T][0]})
        if e:
            excluded[T] = "designs: " + "; ".join(e)[:300]
            todo.remove(T)
            continue
        designs[T] = merge_parts([g[T][1] for g in gathered])
    del gathered, parts
    wall["designs"] = round(time.perf_counter() - t0, 3)
    # (C) inputs, built and validated before any launch; voted with the design shas
    t0 = time.perf_counter()
    inputs, ierr = {}, {}
    shas = {T: designs_sha(designs[T]) for T in todo}
    try:
        with gpu.torch.inference_mode():
            for T in todo:
                try:
                    inputs[T] = build_inputs(gpu, backends, designs[T], T, K, ident["hidden"], rank, ep * M, conf)
                except Exception as exc:  # noqa: BLE001 -- voted
                    ierr[T] = _err(exc)
    except Exception as exc:  # noqa: BLE001 -- voted
        ierr = {T: _err(exc) for T in todo}
    votes = comm.allgather(dict(err=ierr, sha=shas))
    for T in list(todo):
        e = sorted({v["err"].get(T) for v in votes if v["err"].get(T)})
        if e or len({v["sha"].get(T) for v in votes}) != 1:
            excluded[T] = ("inputs: " + "; ".join(e)) if e else "design sha differs across ranks"
            todo.remove(T)
            inputs.pop(T, None)
    wall["inputs"] = round(time.perf_counter() - t0, 3)
    if not todo:
        inputs.clear()
        return fallback("no tier passed the inputs vote")
    rep["designs"] = {T: dict(n=len(designs[T]), sha256=shas[T]) for T in todo}
    # ---- the launches: every loop count below comes from the votes / all-gathered readings (no rank-local decision) ----------
    tc = conf["timing"]
    torch = gpu.torch
    stagers = [(b, b._route_stager) for b in backends]
    times = {}
    t0 = time.perf_counter()
    try:
        for b, _ in stagers:
            b._route_stager = None  # physical slot ids straight into the bucket's staging (restored below)
        with torch.inference_mode():
            comm.barrier()
            host_ms, launch_ms, cyc = warm_probe(torch, inputs[todo[0]], tc)
            readings = comm.allgather((host_ms, launch_ms, cyc))
            host_max = max(r[0] for r in readings)
            launch_max = max(r[1] for r in readings)
            sleep_ms = max(float(tc["sleep_min_ms"]), float(tc["sleep_host_factor"]) * (int(tc["warm"]) + int(tc["timed"])) * host_max)
            plan, dropped, est = cap_plan(todo, {T: len(designs[T]) for T in todo}, tc, sleep_ms, launch_max)
            rep["sleep"] = dict(host_ms=round(host_ms, 4), host_ms_max=round(host_max, 4), launch_ms_max=round(launch_max, 4),
                                cycles_per_us=round(cyc, 3), sleep_ms=round(sleep_ms, 3), est_s=round(est, 2),
                                reps={int(T): r for T, r in plan.items()})
            excluded.update(dropped)
            seed = int(conf["designs"]["seed"])
            for T in todo:
                if T not in plan:
                    continue
                D = len(designs[T])
                order = [int(d) for r in range(plan[T]) for d in np.random.default_rng([seed, 99, r, T]).permutation(D)]
                comm.barrier()
                t1 = time.perf_counter()
                times[T] = (order, time_tier(torch, inputs[T], order, tc, int(sleep_ms * 1000.0 * cyc)))
                wall["T%d" % T] = round(time.perf_counter() - t1, 3)
    finally:
        for b, s in stagers:
            b._route_stager = s
        try:
            torch.cuda.synchronize()
        finally:
            inputs.clear()
    wall["timing"] = round(time.perf_counter() - t0, 3)
    allt = comm.allgather({T: v[1] for T, v in times.items()})
    measured = [T for T in todo if T in times]
    y = {}
    for T in measured:
        mat = np.asarray([a[T] for a in allt], np.float64)  # [EP, blocks] us per launch
        per = mat.min(0)  # per block: the MIN over ranks
        od = np.asarray(times[T][0])
        y[T] = np.array([float(np.median(per[od == d])) for d in range(len(designs[T]))])
        rep.setdefault("y_us", {})[T] = dict(min=round(float(y[T].min()), 3), median=round(float(np.median(y[T])), 3),
                                             max=round(float(y[T].max()), 3),
                                             rank_spread_p50_us=round(float(np.median(mat.max(0) - mat.min(0))), 3))
    ysha = {T: y_sha(y[T]) for T in measured}
    # (D) rank 0 fits and broadcasts the sealed table
    t0 = time.perf_counter()
    payload = None
    if rank == 0:
        try:
            row_meta = dict(ranks=ep, slots_per_rank=M, homes=H, top_k=K, layers=ident["layers"],
                            combine_format=ident["combine_format"], protocol=PROTOCOL, conf_sha256=conf["sha256"])
            payload = fit_table(tiers=tiers, measured=measured, designs=designs, y=y, info=info, fb=fb, conf=conf,
                                row_meta=row_meta)
            payload["why"] = None
        except Exception as exc:  # noqa: BLE001 -- broadcast as a failed fit: every rank takes the fallback table
            payload = dict(table=None, fits={}, why="fit: " + _err(exc))
    payload = comm.bcast(payload, root=0)
    wall["fit"] = round(time.perf_counter() - t0, 3)
    # (E) every rank re-seals the table and compares its own y shas; the table sha all-gathered
    table = payload.get("table")
    mine = None
    if table is not None:
        try:
            ok = costmodel.table_ok(table) and all(
                r.get("source") != "warmup" or r.get("y_sha256") == ysha.get(int(k)) for k, r in table["rows"].items()
            )
            mine = table["sha256"] if ok else "bad"
        except Exception:  # noqa: BLE001
            mine = "bad"
    shas = comm.allgather(mine)
    rep["fits"] = payload.get("fits") or {}
    for T, f in rep["fits"].items():
        if f.get("why"):
            rep["why"][int(T)] = f["why"]
    if keep_detail and rank == 0:
        rep["_detail"] = {T: dict(family=[d["family"] for d in designs[T]], index=[d["index"] for d in designs[T]],
                                  x=[d["x"].tolist() for d in designs[T]], y_us=[float(v) for v in y[T]]) for T in measured}
    if table is None or len(set(shas)) != 1 or shas[0] != table["sha256"]:
        return fallback(payload.get("why") or "the table's sha differs across ranks")
    for T in tiers:
        if T in excluded:
            rep["why"].setdefault(int(T), excluded[T])
    return done(table)


def final_vote(comm, table: dict) -> dict:
    """Every rank's table sha all-gathered: equal -> the table; else raise on every rank (all see the same gathered shas)."""
    shas = comm.allgather(table["sha256"])
    if len(set(shas)) != 1:
        raise RuntimeError("GenEPLB costmb: the ranks' cost tables differ even after the fallback (%s): the prior files or the "
                           "geometry differ across ranks" % sorted(set(s[:16] for s in shas)))
    return table


# ============================================================================================================================ capture check
class CaptureRecorder:
    """While pass 2 captures: the runners the decode graphs bake in, per bucket, for the launched layers' geometry (a wrapper of the
    op's module-global _megamoe_get_runner, which run_cute_dsl_megamoe_nvfp4_blackwell looks up at every call)."""

    def __init__(self, gpu, backend, top_k: int):
        self.op, self.torch, self.backend, self.top_k = gpu.op, gpu.torch, backend, int(top_k)
        self.captured, self.eager = {}, {}
        self._orig = self.op._megamoe_get_runner
        rec = self

        def get_runner(*args, **kwargs):
            runner = rec._orig(*args, **kwargs)
            try:
                if not kwargs and _runner_args_match(args, rec.backend, rec.top_k):
                    b = int(args[7])
                    if rec.torch.cuda.is_current_stream_capturing():
                        rec.captured.setdefault(b, {})[id(runner)] = runner
                    else:
                        rec.eager[b] = rec.eager.get(b, 0) + 1
            except Exception:  # noqa: BLE001 -- a recorder must never break a launch
                pass
            return runner

        self._wrapper = get_runner
        self.op._megamoe_get_runner = get_runner

    def remove(self):
        if getattr(self.op, "_megamoe_get_runner", None) is self._wrapper:
            self.op._megamoe_get_runner = self._orig

    def labels(self, bucket: int):
        return sorted({tactic_of(self.op, r, bucket)[0] for r in self.captured.get(int(bucket), {}).values()})


def check_capture(comm, table: dict, recorder, gpu, backend, top_k: int):
    """balancer.after_capture: every fitted tier's tactic label == the label of the runners pass 2 captured for its bucket (a bucket
    no graph captured: the cached runner's label, i.e. what an eager launch resolves) -> (table, report); any mismatch or error on
    any rank: the fallback table on every rank. No collective when the table holds no fitted row (rank-identical)."""
    rows = {int(k): r for k, r in (table.get("rows") or {}).items() if r.get("source") == "warmup"}
    rep = dict(checked={}, ok=True)
    if not rows:
        return table, rep
    mine, err = {}, None
    try:
        for T, r in sorted(rows.items()):
            b = int(r["bucket"])
            labels = recorder.labels(b) if recorder is not None else []
            how = "captured"
            if not labels:
                p = probe_bucket(gpu, backend, b, top_k)
                labels, how = ([p["tactic"]] if p["tactic"] else []), "cached runner (bucket not captured)"
            mine[T] = dict(bucket=b, fit=r["tactic"], graph=labels, how=how, ok=labels == [r["tactic"]],
                           eager_launches=recorder.eager.get(b, 0) if recorder is not None else None)
    except Exception as exc:  # noqa: BLE001 -- voted
        err = _err(exc)
    votes = comm.allgather((err, mine))
    rep["checked"] = mine
    bad = [v[0] for v in votes if v[0]] + ["T=%d on a rank: fit %s graph %s" % (T, c["fit"][:60], c["graph"])
                                            for v in votes for T, c in (v[1] or {}).items() if not c["ok"]]
    if bad:
        rep.update(ok=False, why="; ".join(bad)[:500])
        return costmodel.fallback_like(table), rep
    return table, rep


# ============================================================================================================================ entries
def _log(rank, msg):
    print("%s rank %d %s" % (TAG, rank, msg), file=sys.stderr, flush=True)


def _fmt_row(r: dict) -> str:
    pin = set(r.get("pinned") or ())
    co = " ".join("%s %.4g%s" % (k, r[k], "(pinned)" if k in pin else "") for k in ("cf", "cl", "cc", "ch"))
    return "%s %s tau %g a %.2f" % (r.get("model"), co, r["tau"], r["a"])


def log_table(rank: int, table: dict, rep: dict, fb_sha=None) -> None:
    """One line per tier and one for the table, on every rank (stderr: every rank's log)."""
    fits = rep.get("fits") or {}
    for k in table.get("tiers") or []:
        r = table["rows"][str(k)]
        f = fits.get(int(k)) or fits.get(str(k)) or {}
        o = (table.get("outcome") or {}).get(str(k))
        if o == "warmup":
            _log(rank, "T=%d bucket=%s warmup %s | hold-out rmse %.3f us (bar %.3f) train %.3f all %.3f | tile %s | wall %s s | row %s"
                 % (int(k), r.get("bucket"), _fmt_row(r), r["holdout_rmse_us"], f.get("bar_us", float("nan")),
                    r["train_rmse_us"], r["rmse_us"], r["tile_rows"], rep["wall_s"].get("T%d" % int(k)), r["sha256"][:12]))
        else:
            _log(rank, "T=%d bucket=%s fallback (%s) %s | row %s" % (int(k), r.get("bucket"),
                 str((rep.get("why") or {}).get(int(k)) or rep.get("why", {}).get("all") or "?")[:200], _fmt_row(r), r["sha256"][:12]))
    n = len(table.get("tiers") or [])
    nw = sum(1 for o in (table.get("outcome") or {}).values() if o == "warmup")
    _log(rank, "table %s tiers %s: warmup %d / %d%s | sleep %s | wall %s" % (
        table["sha256"], ",".join(str(t) for t in table.get("tiers") or []) or "-", nw, n,
        "" if fb_sha is None else " | fallback table %s" % fb_sha, rep.get("sleep"), rep.get("wall_s")))


def _jsonable(obj):
    """Dict keys as strings, recursively (the report mixes int tiers and names)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_report(rank: int, table: dict, rep: dict, out_dir: str, name: str = "report") -> None:
    """TLLM_GENEPLB_COSTMB_DIR: every rank <name>.rank<r>.json, rank 0 table.json and the designs / y (report files only: never
    raises)."""
    try:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        detail = rep.pop("_detail", None)
        (d / ("%s.rank%d.json" % (name, rank))).write_text(json.dumps(_jsonable(rep), indent=1, sort_keys=True, default=str) + "\n")
        if rank == 0:
            (d / "table.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
            for T, v in (detail or {}).items():
                (d / ("design.T%d.json" % int(T))).write_text(json.dumps(v, separators=(",", ":")) + "\n")
    except OSError as exc:
        rep["write_error"] = _err(exc)


def run(lb, **graph):
    """balancer.before_capture's body -> (table, report): measure() on a Dup of the balancer's done communicator, then the final sha
    vote. Collective on every rank."""
    out_dir = os.environ.get("TLLM_GENEPLB_COSTMB_DIR", "")
    layers = [li for li in lb.layers if li.index in lb.staged]
    comm = MpiComm(lb.comm_done)
    try:
        table, rep = measure(comm=comm, rank=lb.ep_rank, geo=lb.geo, layers=layers, graph=graph, gpu_factory=GpuEnv.load,
                             keep_detail=bool(out_dir))
        table = final_vote(comm, table)
    finally:
        comm.free()
    rep["graph"] = {k: v for k, v in graph.items() if k != "purpose"}
    rep["fallback_table_sha256"] = costmodel.fallback_like(table)["sha256"]
    log_table(lb.ep_rank, table, rep, rep["fallback_table_sha256"])
    if out_dir:
        write_report(lb.ep_rank, table, rep, out_dir)
    return table, rep


def after_capture(lb, table: dict, recorder, **graph):
    """balancer.after_capture's body -> (table, report): the capture check on a Dup of the done communicator, the final vote.
    Collective on every rank whose table holds a fitted row (all or none: the table is rank-identical)."""
    gpu = backend = None
    try:
        gpu = GpuEnv.load()
        backend = next(li.moe.backend for li in lb.layers if li.index in lb.staged)
    except Exception:  # noqa: BLE001 -- check_capture records the missing environment as this rank's error (voted)
        pass
    comm = MpiComm(lb.comm_done)
    try:
        table2, rep = check_capture(comm, table, recorder, gpu, backend, lb.geo.top_k)
        if table2 is not table:
            table2 = final_vote(comm, table2)
    finally:
        comm.free()
    rep["graphs_captured"] = graph.get("graphs_captured")
    rep["table_sha256"] = table2["sha256"]
    for T, c in sorted(rep.get("checked", {}).items()):
        _log(lb.ep_rank, "tactic@capture T=%d bucket=%d %s (%s, eager launches %s): fit %s graph %s" % (
            T, c["bucket"], "OK" if c["ok"] else "MISMATCH", c["how"], c.get("eager_launches"), c["fit"], c["graph"]))
    _log(lb.ep_rank, "after capture: graphs captured %s, table %s%s" % (
        graph.get("graphs_captured"), table2["sha256"], "" if rep.get("ok", True) else " (FALLBACK: %s)" % rep.get("why")))
    out_dir = os.environ.get("TLLM_GENEPLB_COSTMB_DIR", "")
    if out_dir:
        write_report(lb.ep_rank, table2, rep, out_dir, name="capture")
    return table2, rep
