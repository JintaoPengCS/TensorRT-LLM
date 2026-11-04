// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
/* Bit-faithful C port of the L3 weight solve's inner loop of the frozen Python reference solver: the bisection on the
 * target T in l3_weights() and the Dinic max-flow _Flow.run(). Same graph (built as the reference _Flow class builds it
 * and passed in as CSR in the same adjacency order), same BFS / DFS order with the same current-arc
 * pointers, the same eps tests and the same double-precision operations in the same order, so every value equals the
 * Python result bit for bit. Build: -O2 -ffp-contract=off (no FMA fusion; no -ffast-math).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n;
    const int *ptr, *idx, *to;
    double *cap, eps;
    int *level, *it, *queue, t;
} G;

static double dfs(G *g, int u, double f)
{
    if (u == g->t)
        return f;
    while (g->it[u] < g->ptr[u + 1] - g->ptr[u]) {
        int ei = g->idx[g->ptr[u] + g->it[u]];
        int v = g->to[ei];
        if (g->cap[ei] > g->eps && g->level[v] == g->level[u] + 1) {
            double c = g->cap[ei];
            double d = dfs(g, v, (c < f) ? c : f);          /* Python min(f, cap[ei]) */
            if (d > g->eps) {
                g->cap[ei] -= d;
                g->cap[ei ^ 1] += d;
                return d;
            }
        }
        g->it[u] += 1;
    }
    return 0.0;
}

static double run(G *g, int s)
{
    double flow = 0.0;
    for (;;) {
        for (int i = 0; i < g->n; i++)
            g->level[i] = -1;
        g->level[s] = 0;
        int head = 0, tail = 0;
        g->queue[tail++] = s;
        while (head < tail) {
            int u = g->queue[head++];
            for (int k = g->ptr[u]; k < g->ptr[u + 1]; k++) {
                int ei = g->idx[k];
                int v = g->to[ei];
                if (g->level[v] < 0 && g->cap[ei] > g->eps) {
                    g->level[v] = g->level[u] + 1;
                    g->queue[tail++] = v;
                }
            }
        }
        if (g->level[g->t] < 0)
            return flow;
        memset(g->it, 0, sizeof(int) * (size_t)g->n);
        for (;;) {
            double f = dfs(g, s, INFINITY);
            if (f <= g->eps)
                break;
            flow += f;
        }
    }
}

int l3_bisect(int n, int n_edges, const int *ptr, const int *idx, const int *to, const double *cap0, int src, int snk,
              int R, const int *rank_edge, const double *B, double lo, double hi, double need_flow, double eps,
              double *best, double *hi_out);

/* Guided bisection: the same lo / hi / mid sequence and the same final flow as l3_bisect, with fewer max-flow runs.
 * 1. A lower bound Ts of the optimum T* by Newton (Dinkelbach) steps from lo: a max flow at T; if it passes the frozen
 *    feasibility test, stop; else the ranks Q reachable from the source in the residual graph give the next
 *    T = (B(Q) + tokens of the experts reachable) / |Q|, the density of a rank set, i.e. a lower bound of T*.
 * 2. The frozen bisection, deciding mid < Ts (1 - g) "infeasible" and mid > Ts (1 + g) "feasible" by comparison and
 *    running the exact Dinic (run) in the band |mid - Ts| <= g Ts, where the frozen code's own tolerances act
 *    (need x 1e-12 and eps = 1e-13 x total; g = 1e-9 >> both).
 * 3. The final flow is always the exact Dinic at the final hi (as in the frozen code). If it is infeasible, or Newton
 *    does not converge in max_newton steps, the exact l3_bisect is run from the original bounds instead (fallback).
 * Equality with the frozen solver is established by the bit-identity tests (randomised layers), not by
 * this comment. *stats: [0] max flows run, [1] fallback taken. */
static int guided(int n, int n_edges, const int *ptr, const int *idx, const int *to, const double *cap0, int src, int snk,
                  int R, const int *rank_edge, const double *B, double lo, double hi, double need_flow, double eps,
                  const double *N, int m, double g, int max_newton, double *best, double *hi_out, int *stats)
{
    double *cap = malloc(sizeof(double) * (size_t)n_edges);
    int *level = malloc(sizeof(int) * (size_t)n), *it = malloc(sizeof(int) * (size_t)n);
    int *queue = malloc(sizeof(int) * (size_t)n);
    if (!cap || !level || !it || !queue) {
        free(cap); free(level); free(it); free(queue);
        return -1;
    }
    G gr = {n, ptr, idx, to, cap, eps, level, it, queue, snk};
    double need = need_flow * (1.0 - 1e-12), lo0 = lo, hi0 = hi;
    int flows = 0, ok = 1;
#define SETCAP(T) do { memcpy(cap, cap0, sizeof(double) * (size_t)n_edges); \
                       for (int r_ = 0; r_ < R; r_++) { double x_ = (T) - B[r_]; cap[rank_edge[r_]] = (x_ > 0.0) ? x_ : 0.0; } } while (0)
    /* 1. Newton from lo */
    double Ts = lo;
    int steps = 0;
    for (;;) {
        SETCAP(Ts);
        double f = run(&gr, src);
        flows++;
        if (f >= need)
            break;
        if (++steps > max_newton) {
            ok = 0;
            break;
        }
        /* reachable set from the source in the residual graph (level[] of the last BFS of run() is not it: redo) */
        for (int i = 0; i < n; i++)
            level[i] = -1;
        level[src] = 0;
        int head = 0, tail = 0;
        queue[tail++] = src;
        while (head < tail) {
            int u = queue[head++];
            for (int k = ptr[u]; k < ptr[u + 1]; k++) {
                int ei = idx[k], v = to[ei];
                if (level[v] < 0 && cap[ei] > eps) {
                    level[v] = 0;
                    queue[tail++] = v;
                }
            }
        }
        double num = 0.0;
        int q = 0;
        for (int r = 0; r < R; r++)
            if (level[1 + m + r] >= 0) {
                num += B[r];
                q++;
            }
        for (int i = 0; i < m; i++)
            if (level[1 + i] >= 0)
                num += N[i];
        double Tn = (q > 0) ? num / q : Ts;
        if (!(Tn > Ts)) {   /* no progress: should not happen for an infeasible T; be safe */
            ok = 0;
            break;
        }
        Ts = Tn;
    }
    int found = 0, have = 0;
    if (ok) {
        /* 2. the frozen bisection, decided by comparison outside the band */
        while (hi - lo > 1e-9 * hi) {
            double mid = 0.5 * (lo + hi);
            if (mid < Ts * (1.0 - g)) {
                lo = mid;
            } else if (mid > Ts * (1.0 + g)) {
                hi = mid;
                found = 1;
                have = 0;
            } else {
                SETCAP(mid);
                double f = run(&gr, src);
                flows++;
                if (f >= need) {
                    hi = mid;
                    memcpy(best, cap, sizeof(double) * (size_t)n_edges);
                    found = 1;
                    have = 1;
                } else {
                    lo = mid;
                }
            }
        }
        /* 3. the exact final flow at the final hi */
        if (found && !have) {
            SETCAP(hi);
            double f = run(&gr, src);
            flows++;
            if (f >= need)
                memcpy(best, cap, sizeof(double) * (size_t)n_edges);
            else
                ok = 0;
        }
    }
#undef SETCAP
    free(cap); free(level); free(it); free(queue);
    if (!ok) {   /* fallback: the exact bisection from the original bounds */
        stats[1] = 1;
        found = l3_bisect(n, n_edges, ptr, idx, to, cap0, src, snk, R, rank_edge, B, lo0, hi0, need_flow, eps, best, hi_out);
        stats[0] = flows;
        return found;
    }
    *hi_out = hi;
    stats[0] = flows;
    return found;
}

/* The graph of l3_weights(), built exactly as the frozen _Flow would build it (integer bookkeeping and copies of N only,
 * no float arithmetic): nodes src = 0, experts 1..m, ranks 1+m..m+R, snk = m+R+1; per expert i in order:
 * edge(src, 1+i, N[i]) then edge(1+i, 1+m+r, big) for its ranks r in the given order; then edge(1+m+r, snk, 0.0) for
 * r = 0..R-1. edge(u, v, c) appends forward id k to adj[u] and reverse id k+1 (cap 0.0) to adj[v].
 * Outputs (caller-allocated): to / cap0 [n_edges], ptr [n+1], idx [n_edges], mid_ids [rptr[m]], rank_edge [R]. */
static void build(int m, const double *N, const int *rptr, const int *ridx, int R, double big,
                  int *to, double *cap0, int *ptr, int *idx, int *mid_ids, int *rank_edge)
{
    int n = m + R + 2, src = 0, snk = m + R + 1, k = 0;
    int *deg = calloc((size_t)n, sizeof(int)), *fill = calloc((size_t)n, sizeof(int));
    int *eu = malloc(sizeof(int) * (size_t)(2 * (m + rptr[m] + R)));   /* owner node of each edge id */
#define EDGE(u, v, c) do { to[k] = (v); cap0[k] = (c); eu[k] = (u); deg[u]++; \
                           to[k + 1] = (u); cap0[k + 1] = 0.0; eu[k + 1] = (v); deg[v]++; k += 2; } while (0)
    for (int i = 0; i < m; i++) {
        EDGE(src, 1 + i, N[i]);
        for (int j = rptr[i]; j < rptr[i + 1]; j++) {
            mid_ids[j] = k;
            EDGE(1 + i, 1 + m + ridx[j], big);
        }
    }
    for (int r = 0; r < R; r++) {
        rank_edge[r] = k;
        EDGE(1 + m + r, snk, 0.0);
    }
#undef EDGE
    ptr[0] = 0;
    for (int u = 0; u < n; u++)
        ptr[u + 1] = ptr[u] + deg[u];
    for (int e = 0; e < k; e++) {   /* edge ids in creation order -> per-node append order */
        int u = eu[e];
        idx[ptr[u] + fill[u]++] = e;
    }
    free(deg); free(fill); free(eu);
}

/* Test hook: the graph build() produces (for comparison with the reference _Flow). */
void l3_graph(int m, const double *N, const int *rptr, const int *ridx, int R, double big,
              int *to, double *cap0, int *ptr, int *idx, int *mid_ids, int *rank_edge)
{
    build(m, N, rptr, ridx, R, big, to, cap0, ptr, idx, mid_ids, rank_edge);
}

int l3_bisect(int n, int n_edges, const int *ptr, const int *idx, const int *to, const double *cap0, int src, int snk,
              int R, const int *rank_edge, const double *B, double lo, double hi, double need_flow, double eps,
              double *best, double *hi_out);

/* CPython 3.12 builtin sum() over a sequence of floats with the default start 0 (Python/bltinmodule.c builtin_sum_impl):
 * the int start becomes 0 + x[0] at the first float, then Neumaier's improved Kahan-Babuska summation (Neumaier 1974),
 * and the compensation is added at the end when it is non-zero and finite. */
double l3_pysum(const double *a, int n)
{
    if (n == 0)
        return 0.0;
    double f = 0.0 + a[0], c = 0.0;
    for (int i = 1; i < n; i++) {
        double x = a[i], t = f + x;
        if (fabs(f) >= fabs(x))
            c += (f - t) + x;
        else
            c += (x - t) + f;
        f = t;
    }
    if (c != 0.0 && isfinite(c))
        f += c;
    return f;
}

int l3_solve(int m, const double *N, const int *rptr, const int *ridx, int R, const double *B, double lo, double hi,
             double need_flow, double eps, double big, double *x, double *hi_out, int mode, double g, int max_newton,
             int *stats);

static double pymax2(double a, double b) { return (b > a) ? b : a; }   /* Python max(a, b): the first unless b is larger */

/* The whole frozen reference l3_weights(load, phys, align_tile, tile), statement for statement
 * (mode 0: the exact bisection; mode 1: the guided bisection, same sequence and final flow, fewer max flows):
 * E experts, R ranks, LS slots per rank, load[E] (float(load[e])), phys[n_slots] (expert or -1), tile (the config's
 * integer tile_rows). Outputs w[n_slots], *T_out, *improved. Returns 0, or -1 on allocation failure. Python sum() over
 * floats is l3_pysum; the tile path's integer arithmetic (math.floor returns int) is exact on integer-valued doubles. */
int l3_weights_full(int E, int R, int LS, const double *load, int n_slots, const int *phys, int align_tile, double tile,
                    double *w, double *T_out, int *improved, int mode, double g, int max_newton, int *stats)
{
    int ret = -1;
    int *cnt = calloc((size_t)E + 1, sizeof(int)), *sptr = malloc(sizeof(int) * ((size_t)E + 1)), *sidx = malloc(sizeof(int) * ((size_t)n_slots + 1));
    int *fillc = calloc((size_t)E + 1, sizeof(int)), *var_e = malloc(sizeof(int) * ((size_t)E + 1));
    int *rptr = malloc(sizeof(int) * ((size_t)E + 2)), *ridx = malloc(sizeof(int) * ((size_t)E * (size_t)R + 1));
    double *B = calloc((size_t)R, sizeof(double)), *even = malloc(sizeof(double) * (size_t)R);
    double *Nv = malloc(sizeof(double) * ((size_t)E + 1)), *x = malloc(sizeof(double) * ((size_t)E * (size_t)R + 1));
    double *planned = malloc(sizeof(double) * ((size_t)n_slots + 1)), *rank_tok = malloc(sizeof(double) * (size_t)R);
    char *seen = malloc((size_t)R);
    if (!cnt || !sptr || !sidx || !fillc || !var_e || !rptr || !ridx || !B || !even || !Nv || !x || !planned || !rank_tok || !seen)
        goto out;
    /* slots_of[e]: slots in increasing order */
    for (int s = 0; s < n_slots; s++)
        if (phys[s] >= 0)
            cnt[phys[s]]++;
    sptr[0] = 0;
    for (int e = 0; e < E; e++)
        sptr[e + 1] = sptr[e] + cnt[e];
    for (int s = 0; s < n_slots; s++)
        if (phys[s] >= 0)
            sidx[sptr[phys[s]] + fillc[phys[s]]++] = s;
    for (int s = 0; s < n_slots; s++)
        w[s] = 0.0;
    int m = 0;
    rptr[0] = 0;
    for (int e = 0; e < E; e++) {
        int len = sptr[e + 1] - sptr[e], nr = 0;
        double N = load[e];
        memset(seen, 0, (size_t)R);
        for (int k = sptr[e]; k < sptr[e + 1]; k++)
            seen[sidx[k] / LS] = 1;
        int first = -1;
        for (int r = 0; r < R; r++)                 /* sorted({s // LS for s in sl}) */
            if (seen[r]) {
                if (first < 0)
                    first = r;
                nr++;
            }
        if (N <= 0.0 || nr == 1) {
            for (int k = sptr[e]; k < sptr[e + 1]; k++)
                w[sidx[k]] = 1.0 / len;
            if (N > 0.0)
                B[first] += N;
            continue;
        }
        var_e[m] = e;
        Nv[m] = N;
        rptr[m + 1] = rptr[m] + nr;
        int j = rptr[m];
        for (int r = 0; r < R; r++)
            if (seen[r])
                ridx[j++] = r;
        m++;
    }
    if (m == 0) {
        double T = B[0];
        for (int r = 1; r < R; r++)
            T = pymax2(T, B[r]);
        *T_out = T;
        *improved = 0;
        ret = 0;
        goto out;
    }
    for (int r = 0; r < R; r++)
        even[r] = B[r];
    for (int i = 0; i < m; i++) {
        int e = var_e[i], len = sptr[e + 1] - sptr[e];
        for (int k = sptr[e]; k < sptr[e + 1]; k++)
            even[sidx[k] / LS] += Nv[i] / len;
    }
    double total = l3_pysum(B, R) + l3_pysum(Nv, m);
    double maxB = B[0], hi = even[0];
    for (int r = 1; r < R; r++) {
        maxB = pymax2(maxB, B[r]);
        hi = pymax2(hi, even[r]);
    }
    double lo = pymax2(maxB, total / R);
    double big = 2.0 * total, need_flow = l3_pysum(Nv, m), eps = 1e-13 * total, hi_out = hi;
    int found = l3_solve(m, Nv, rptr, ridx, R, B, lo, hi, need_flow, eps, big, x, &hi_out, mode, g, max_newton, stats);
    if (found < 0)
        goto out;
    if (found == 0) {
        for (int i = 0; i < m; i++) {
            int e = var_e[i], len = sptr[e + 1] - sptr[e];
            for (int k = sptr[e]; k < sptr[e + 1]; k++)
                w[sidx[k]] = 1.0 / len;
        }
        *T_out = hi_out;
        *improved = 0;
        ret = 0;
        goto out;
    }
    for (int i = 0; i < m; i++) {               /* planned[s], in var order then slot order (the dict's insertion order) */
        int e = var_e[i], nr = rptr[i + 1] - rptr[i];
        double tot = l3_pysum(x + rptr[i], nr), N = Nv[i];
        for (int k = sptr[e]; k < sptr[e + 1]; k++) {
            int s = sidx[k], r = s / LS, n_r = 0, j = rptr[i];
            for (int kk = sptr[e]; kk < sptr[e + 1]; kk++)
                if (sidx[kk] / LS == r)
                    n_r++;
            while (ridx[j] != r)
                j++;
            planned[s] = ((tot > 0) ? N * x[j] / tot : N / nr) / n_r;
        }
    }
    if (align_tile) {
        for (int r = 0; r < R; r++)
            rank_tok[r] = B[r];
        for (int i = 0; i < m; i++) {
            int e = var_e[i];
            for (int k = sptr[e]; k < sptr[e + 1]; k++) {
                int s = sidx[k];
                planned[s] = floor(planned[s] / tile + 1e-9) * tile;   /* int in Python: exact here (integer-valued) */
                rank_tok[s / LS] += planned[s];
            }
        }
        for (int i = 0; i < m; i++) {
            int e = var_e[i];
            double isum = 0.0;                                         /* Python int sum: exact */
            for (int k = sptr[e]; k < sptr[e + 1]; k++)
                isum += planned[sidx[k]];
            double rem = Nv[i] - isum;
            if (sptr[e] == sptr[e + 1]) {                              /* Python: min() of an empty sequence raises */
                ret = -2;
                goto out;
            }
            int s_star = sidx[sptr[e]];
            for (int k = sptr[e] + 1; k < sptr[e + 1]; k++) {          /* min(key=(rank_tok[s // LS], s)): first minimum */
                int s = sidx[k];
                double a = rank_tok[s / LS], b = rank_tok[s_star / LS];
                if (a < b || (a == b && s < s_star))
                    s_star = s;
            }
            planned[s_star] += rem;
            rank_tok[s_star / LS] += rem;
        }
    }
    for (int i = 0; i < m; i++) {
        int e = var_e[i];
        for (int k = sptr[e]; k < sptr[e + 1]; k++)
            w[sidx[k]] = planned[sidx[k]] / Nv[i];
    }
    *T_out = hi_out;
    *improved = 1;
    ret = 0;
out:
    free(cnt); free(sptr); free(sidx); free(fillc); free(var_e); free(rptr); free(ridx); free(B); free(even); free(Nv);
    free(x); free(planned); free(rank_tok); free(seen);
    return ret;
}

/* Build + bisection; x[j] = the final flow on mid edge j (best[mid_ids[j] ^ 1], CSR order of the ranks). */
int l3_solve(int m, const double *N, const int *rptr, const int *ridx, int R, const double *B, double lo, double hi,
             double need_flow, double eps, double big, double *x, double *hi_out, int mode, double g, int max_newton,
             int *stats)
{
    int n = m + R + 2, n_edges = 2 * (m + rptr[m] + R);
    int *to = malloc(sizeof(int) * (size_t)n_edges), *ptr = malloc(sizeof(int) * (size_t)(n + 1));
    int *idx = malloc(sizeof(int) * (size_t)n_edges), *mid_ids = malloc(sizeof(int) * (size_t)(rptr[m] + 1));
    int *rank_edge = malloc(sizeof(int) * (size_t)R);
    double *cap0 = malloc(sizeof(double) * (size_t)n_edges), *best = malloc(sizeof(double) * (size_t)n_edges);
    int found = -1;
    if (to && ptr && idx && mid_ids && rank_edge && cap0 && best) {
        build(m, N, rptr, ridx, R, big, to, cap0, ptr, idx, mid_ids, rank_edge);
        if (mode == 1)
            found = guided(n, n_edges, ptr, idx, to, cap0, 0, m + R + 1, R, rank_edge, B, lo, hi, need_flow, eps, N, m,
                           g, max_newton, best, hi_out, stats);
        else
            found = l3_bisect(n, n_edges, ptr, idx, to, cap0, 0, m + R + 1, R, rank_edge, B, lo, hi, need_flow, eps, best, hi_out);
        if (found == 1)
            for (int j = 0; j < rptr[m]; j++)
                x[j] = best[mid_ids[j] ^ 1];
    }
    free(to); free(ptr); free(idx); free(mid_ids); free(rank_edge); free(cap0); free(best);
    return found;
}

/* The bisection of l3_weights():
 *   while hi - lo > 1e-9 * hi: mid = 0.5 * (lo + hi); cap = cap0 with cap[rank_edge[r]] = max(0.0, mid - B[r]);
 *   f = run(cap); if f >= need_flow * (1.0 - 1e-12): hi, best = mid, cap  else: lo = mid
 * Returns 1 and the final feasible caps in best (after that run) when some mid was feasible, else 0; *hi_out = hi.
 * Returns -1 on allocation failure. */
int l3_bisect(int n, int n_edges, const int *ptr, const int *idx, const int *to, const double *cap0, int src, int snk,
              int R, const int *rank_edge, const double *B, double lo, double hi, double need_flow, double eps,
              double *best, double *hi_out)
{
    double *cap = malloc(sizeof(double) * (size_t)n_edges);
    int *level = malloc(sizeof(int) * (size_t)n), *it = malloc(sizeof(int) * (size_t)n);
    int *queue = malloc(sizeof(int) * (size_t)n);
    if (!cap || !level || !it || !queue) {
        free(cap); free(level); free(it); free(queue);
        return -1;
    }
    G g = {n, ptr, idx, to, cap, eps, level, it, queue, snk};
    int found = 0;
    double need = need_flow * (1.0 - 1e-12);
    while (hi - lo > 1e-9 * hi) {
        double mid = 0.5 * (lo + hi);
        memcpy(cap, cap0, sizeof(double) * (size_t)n_edges);
        for (int r = 0; r < R; r++) {
            double x = mid - B[r];
            cap[rank_edge[r]] = (x > 0.0) ? x : 0.0;          /* Python max(0.0, x) */
        }
        double f = run(&g, src);
        if (f >= need) {
            hi = mid;
            memcpy(best, cap, sizeof(double) * (size_t)n_edges);
            found = 1;
        } else {
            lo = mid;
        }
    }
    *hi_out = hi;
    free(cap); free(level); free(it); free(queue);
    return found;
}
