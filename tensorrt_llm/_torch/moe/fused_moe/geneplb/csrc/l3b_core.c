// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
/* The "B" share solver of one MegaMoE layer (cost model B), deterministic C.
 *
 * Given the per-replay rows N_e of every expert and the routed layout phys[slot] (expert or -1), choose the planned rows y of every
 * (replicated expert, holder rank) pair and return per-slot weights w[s] = planned_s / N_e (0 = the copy is deactivated: the route
 * kernel never picks it), even within a rank as in l3_weights_full (l3_core.c).  Objective per layer:
 *     f(x) = cf [x > 0] + cl (q(x) - 1)+ + cc x,  q(x) = ceil((x + d(x)) / tile),  d(x) = ceil(margin sqrt(x)) (0 when margin = 0)
 *     c_r  = sum_{s in r} f(x_s) + ch (max_{s in r} q(x_s) - 1)+
 *     J    = tau log sum_r exp(c_r / tau), or max_r c_r for tau <= 0.06
 * Pricing (par[14]): 0 = mean (the above); 1 = poisson: q(x) is the expected tile count of a copy whose per-step rows X are Poisson
 * around the planned x, qe(x) = 1 + sum_{j >= 1} P(X > j tile), P from a normal with variance disp x (par[15]) and the continuity
 * correction (qexp below; the margin is then not priced, it only places the polish's boundary moves through q_of).
 * Candidates S_C (C's own split, l3_weights_full of l3_core.c, linked into the same library), S_B (bisection on T + Dinic
 * max-flow with cost-derived rank capacities, then <= max_act_steps greedy activations pre-screened in closed form), S_A (home
 * copies only); each hot-polished (hot_polish) and, for tau > 0.06, LSE-polished (lse_polish), the best of raw / hot / lse
 * kept; pick the minimum J, ties S_C, S_B, S_A.  J(B) <= J(S_C) by construction.
 *
 * exp / log are the deterministic polynomials below (no libm transcendental); the poisson pricing's erfc is the C99 libm
 * erfc().  Ties by index.  Single thread, no globals written, no clock, no randomness.  Build: -O2 -ffp-contract=off (no FMA
 * fusion; no -ffast-math), with l3_core.c.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int l3_weights_full(int E, int R, int LS, const double *load, int n_slots, const int *phys, int align_tile, double tile,
                    double *w, double *T_out, int *improved, int mode, double g, int max_newton, int *stats);

#define N_PAR 16
#define N_DIAG 24
#define TAU_MAX_RULE 0.06
#define HOT_EPS 1e-9
#define LSE_EPS 1e-6
#define LSE_SLACK 1e-9
#define SNAP_REL 1e-9
#define FLOW_NEED_REL 1e-12
#define FLOW_EPS_REL 1e-13
#define MAX_BISECT 200
#define MAX_MOVES 6
#define MAX_INT_PAR 1e6         /* max_act_steps, polish_iters, c_mode, c_max_newton, lse_top: integer-valued, <= this */
#define MAX_LOAD 1e15           /* rows per replay of one expert */
#define DISP_MAX 64.0           /* pricing poisson: 0 < disp <= this */
#define ZCUT 8.0                /* pricing poisson: boundaries more than ZCUT sigma away count 1 (below) or 0 (above) */
#define MAXJ 64                 /* pricing poisson: at most this many boundary terms per copy */
#define QCACHE 4096             /* pricing poisson: entries of the per-solve qexp memo (a power of two) */

/* ------------------------------------------------------------------------------------------- deterministic exp / log */
/* exp: x = (64 k + j) ln2 / 64 + r, |r| <= ln2 / 128; exp(x) = 2^k 2^(j/64) p(r), p = degree-5 Taylor (Horner), 2^(j/64) correctly
 * rounded, ln2 / 64 split hi (36 bits: k' hi exact) + lo; max relative error ~4e-16.  log: x = m 2^e, m in [sqrt(1/2), sqrt(2)),
 * log m = 2 atanh(s), s = (m - 1) / (m + 1), odd series to s^21.  Literals in hex (parsing can never differ); l3b_tables exports them. */
static const double EXP_INV = 0x1.71547652b82fep+6, EXP_HI = 0x1.62e42fefa0000p-7, EXP_LO = 0x1.cf79abc9e3b3ap-46;
static const double EXP_C[6] = {0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p-1, 0x1.5555555555555p-3,
                                0x1.5555555555555p-5, 0x1.1111111111111p-7};
static const double EXP_T[64] = {
    0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0, 0x1.0874518759bc8p+0,
    0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0, 0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0,
    0x1.172b83c7d517bp+0, 0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
    0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0, 0x1.2d285a6e4030bp+0,
    0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0, 0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0,
    0x1.3dea64c123422p+0, 0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
    0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0, 0x1.56f4736b527dap+0,
    0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0, 0x1.6247eb03a5585p+0, 0x1.6623882552225p+0,
    0x1.6a09e667f3bcdp+0, 0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
    0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0, 0x1.868d99b4492edp+0,
    0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0, 0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e50p+0,
    0x1.9c49182a3f090p+0, 0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdbfp+0,
    0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0, 0x1.bcc1e904bc1d2p+0,
    0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0, 0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0,
    0x1.d5818dcfba487p+0, 0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
    0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0, 0x1.fa7c1819e90d8p+0,
};
static const double LN2_HI = 0x1.62e42fee00000p-1, LN2_LO = 0x1.a39ef35793c76p-33, SQRT_HALF = 0x1.6a09e667f3bcdp-1;
/* 2 / (2 n + 1), the correctly rounded quotients */
static const double LOG_C[11] = {0x1.0000000000000p+1, 0x1.5555555555555p-1, 0x1.999999999999ap-2, 0x1.2492492492492p-2,
                                 0x1.c71c71c71c71cp-3, 0x1.745d1745d1746p-3, 0x1.3b13b13b13b14p-3, 0x1.1111111111111p-3,
                                 0x1.e1e1e1e1e1e1ep-4, 0x1.af286bca1af28p-4, 0x1.8618618618618p-4};

static double det_exp(double x)
{
    if (x != x)
        return x;
    if (x < -708.0)
        return 0.0;
    if (x > 709.0)
        return INFINITY;
    double kf = floor(x * EXP_INV + 0.5);
    double r = (x - kf * EXP_HI) - kf * EXP_LO;
    int ki = (int)kf, j = ki & 63, k = (ki - j) / 64;
    double p = EXP_C[5];
    for (int n = 4; n >= 0; n--)
        p = p * r + EXP_C[n];
    /* ldexp(EXP_T[j] p, k) with k in [-1022, 1022]: 2^k is a normal double, the product is the same correctly rounded value */
    uint64_t bits = (uint64_t)(k + 1023) << 52;
    double s;
    memcpy(&s, &bits, sizeof(s));
    return (EXP_T[j] * p) * s;
}

static double det_log(double x)
{
    if (!(x > 0.0))
        return (x == 0.0) ? -INFINITY : NAN;
    if (x == INFINITY)
        return x;
    int e;
    double m = frexp(x, &e);
    if (m < SQRT_HALF) {
        m = m * 2.0;
        e -= 1;
    }
    double s = (m - 1.0) / (m + 1.0);
    double s2 = s * s;
    double p = LOG_C[10];
    for (int n = 9; n >= 0; n--)
        p = p * s2 + LOG_C[n];
    return e * LN2_HI + (s * p + e * LN2_LO);
}

/* ------------------------------------------------------------------------------------------- poisson pricing (pricing = 1) */
/* erfc is the C99 libm erfc() (one image computes the same bits on every rank); 1 / sqrt(2) correctly rounded. */
static const double INV_SQRT2 = 0x1.6a09e667f3bcdp-1;

/* the expected tile count of a copy with x > 0 planned rows: qe = 1 + sum_{j >= 1} P(X > j tile), X normal(x, disp x) with the
 * continuity correction, P(X > b) = Qn((b + 0.5 - x) / sigma), Qn(z) = erfc(z / sqrt 2) / 2.  The boundaries more than ZCUT sigma
 * below x count 1 each, those above count 0; at most MAXJ terms are evaluated, the rest of the window (absurd loads only) priced as
 * the mean mode ([x > j tile]).  0 for x <= 0; max(1, ceil(x / tile)) if sigma underflows (x < 1 row, no boundary near). */
static double qexp(double x, double tile, double disp)
{
    if (!(x > 0.0))
        return 0.0;
    double sigma = sqrt(disp * x);
    if (!(sigma > 0.0)) {
        double c = ceil(x / tile);
        return (c < 1.0) ? 1.0 : c;
    }
    double j = ceil((x - 0.5 - ZCUT * sigma) / tile);
    if (j < 1.0)
        j = 1.0;
    double q = j;
    for (int n = 0;; n++) {
        double z = (j * tile + 0.5 - x) / sigma;
        if (!(z <= ZCUT))
            break;
        if (n >= MAXJ) {
            double j2 = floor((x - 0.5 + ZCUT * sigma) / tile);
            double jm = ceil(x / tile) - 1.0;
            double hi = (j2 < jm) ? j2 : jm;
            if (hi >= j)
                q = q + (hi - j + 1.0);
            break;
        }
        q = q + 0.5 * erfc(z * INV_SQRT2);
        j = j + 1.0;
    }
    return q;
}

/* test hooks */
double l3b_det_exp(double x) { return det_exp(x); }
double l3b_det_log(double x) { return det_log(x); }
double l3b_erfc(double u) { return erfc(u); }
double l3b_qexp(double x, double tile, double disp) { return qexp(x, tile, disp); }
void l3b_tables(double *exp_t, double *exp_c, double *log_c, double *consts)
{
    memcpy(exp_t, EXP_T, sizeof(EXP_T));
    memcpy(exp_c, EXP_C, sizeof(EXP_C));
    memcpy(log_c, LOG_C, sizeof(LOG_C));
    consts[0] = EXP_INV;
    consts[1] = EXP_HI;
    consts[2] = EXP_LO;
    consts[3] = LN2_HI;
    consts[4] = LN2_LO;
    consts[5] = SQRT_HALF;
}

static double lse(const double *cost, int R, double tau)
{
    double M = cost[0];
    for (int r = 1; r < R; r++)
        if (cost[r] > M)
            M = cost[r];
    if (tau <= TAU_MAX_RULE)
        return M;
    double Z = 0.0;
    for (int r = 0; r < R; r++)
        Z = Z + det_exp((cost[r] - M) / tau);
    return M + tau * det_log(Z);
}

double l3b_lse(const double *cost, int R, double tau) { return lse(cost, R, tau); }

static int argmax(const double *v, int n)
{
    int hi = 0;
    for (int r = 1; r < n; r++)
        if (v[r] > v[hi])
            hi = r;
    return hi;
}

/* ------------------------------------------------------------------------------------------- the layer */
typedef struct {
    int E, R, LS, n_slots;
    double cf, cl, cc, ch, tau, tile, margin, tol;
    double inv_tile;              /* 1 / tile when tile is a power of two (x / tile == x * inv_tile exactly), else 0 */
    int pricing;                  /* 0 mean, 1 poisson (qexp with disp) */
    double disp;
    double *qc_x, *qc_q;          /* pricing poisson: [QCACHE] memo of qexp (a pure function of x within a solve; 0 = empty slot) */
    int max_act, polish_iters, top;
    int m, P;
    int *sptr, *sidx;             /* slots of expert e: sidx[sptr[e] .. sptr[e + 1]) ascending */
    int *var;                     /* [m] expert of replicated expert i */
    double *N;                    /* [m] */
    int *pexp, *pr, *pk;          /* [P] pair -> i, rank, copies on the rank */
    int *hptr, *home;             /* [m + 1] pairs of i; [m] home pair of i */
    int *rptr, *ridx;             /* [R + 1], [P] pairs of rank r, ascending i */
    double *fixF, *fixQ, *Ff;     /* [R] fixed part: sum f, max q, flow-model fixed cost (with its hot term) */
    long flows;
} Lay;

static double delta_of(const Lay *L, double x)
{
    if (L->margin > 0.0)
        return ceil(L->margin * sqrt(x));
    return 0.0;
}

static double tiles(const Lay *L, double v)   /* v / tile, exactly */
{
    return (L->inv_tile > 0.0) ? v * L->inv_tile : v / L->tile;
}

static double q_of(const Lay *L, double x)
{
    if (!(x > 0.0))
        return 0.0;
    return ceil(tiles(L, x + delta_of(L, x)));
}

/* qexp(x, tile, disp) of this solve through a direct-mapped memo (x > 0): the polish prices the same rows again and again (~93 % of
 * the calls on the DSV4 layers); a hit returns the stored bits of the same pure function, so the result is qexp's exactly. */
static double qexp_memo(const Lay *L, double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof(b));
    b ^= b >> 31;
    b *= 0x9e3779b97f4a7c15ULL;
    b ^= b >> 29;
    size_t i = (size_t)(b & (uint64_t)(QCACHE - 1));
    if (L->qc_x[i] == x)
        return L->qc_q[i];
    double q = qexp(x, L->tile, L->disp);
    L->qc_x[i] = x;
    L->qc_q[i] = q;
    return q;
}

static double fq(const Lay *L, double x, double *q)
{
    if (!(x > 0.0)) {
        *q = 0.0;
        return 0.0;
    }
    double qq = L->pricing ? qexp_memo(L, x) : ceil(tiles(L, x + delta_of(L, x)));
    *q = qq;
    return L->cf + L->cl * (qq - 1.0) + L->cc * x;
}

static double capfn(const Lay *L, double b, int n)
{
    double free_ = L->tile * n;
    if (L->cc * free_ >= b)
        return (L->cc > 0.0) ? b / L->cc : free_;
    return free_ + (b - L->cc * free_) / (L->cl / L->tile + L->cc);
}

/* ------------------------------------------------------------------------------------------- a split and its exact costs */
typedef struct {
    double *y, *fv, *qv, *qex;    /* [P] */
    double *lin, *Q, *cost;       /* [R] */
} St;

static void refresh(const Lay *L, St *S, int r)
{
    double lin = L->fixF[r], Q = L->fixQ[r];
    for (int j = L->rptr[r]; j < L->rptr[r + 1]; j++) {
        int p = L->ridx[j];
        double yp = S->y[p];
        if (yp > 0.0) {
            int k = L->pk[p];
            double q, f = fq(L, (k == 1) ? yp : yp / k, &q);
            S->fv[p] = (k == 1) ? f : k * f;
            S->qv[p] = q;
            lin = lin + S->fv[p];
            if (q > Q)
                Q = q;
        } else {
            S->fv[p] = 0.0;
            S->qv[p] = 0.0;
        }
    }
    for (int j = L->rptr[r]; j < L->rptr[r + 1]; j++) {
        int p = L->ridx[j];
        double qx = L->fixQ[r];
        for (int jj = L->rptr[r]; jj < L->rptr[r + 1]; jj++)
            if (jj != j && S->qv[L->ridx[jj]] > qx)
                qx = S->qv[L->ridx[jj]];
        S->qex[p] = qx;
    }
    S->lin[r] = lin;
    S->Q[r] = Q;
    S->cost[r] = (Q > 1.0) ? lin + L->ch * (Q - 1.0) : lin;
}

static void state_init(const Lay *L, St *S, const double *y)
{
    memcpy(S->y, y, sizeof(double) * (size_t)L->P);
    for (int r = 0; r < L->R; r++)
        refresh(L, S, r);
}

static double cost_with(const Lay *L, const St *S, int p, double ny)
{
    int r = L->pr[p];
    double fvn, qn;
    if (ny > 0.0) {
        int k = L->pk[p];
        double f = fq(L, (k == 1) ? ny : ny / k, &qn);
        fvn = (k == 1) ? f : k * f;
    } else {
        fvn = 0.0;
        qn = 0.0;
    }
    double lin = (S->lin[r] - S->fv[p]) + fvn;
    double Q = (S->qex[p] > qn) ? S->qex[p] : qn;
    return (Q > 1.0) ? lin + L->ch * (Q - 1.0) : lin;
}

/* the moves of pair p1's rows whose source value depends on p1 only, in order: all, down to the tile boundary below (minus the
 * margin), one tile per copy, half, third.  ny1[t], m[t], add[t] (snap residual, has_add[t]), a[t] = the source rank's cost after
 * the move; the target gets y2 + m (then + add when the source snapped to exactly 0: a copy left with < SNAP_REL x its rows is
 * emptied).  Returns the count. */
typedef struct {
    double ny1[MAX_MOVES], m[MAX_MOVES], add[MAX_MOVES], a[MAX_MOVES];
    int has_add[MAX_MOVES], c;
} Src;

static void src_moves(const Lay *L, const St *S, int p1, Src *M)
{
    double n = S->y[p1];
    int k1 = L->pk[p1], c = 0;
    M->ny1[c] = 0.0;
    M->m[c] = n;
    c++;
    double xc = n / k1, qc = q_of(L, xc);
    if (qc >= 2.0) {
        double Bd = L->tile * (qc - 1.0);
        double t = Bd - delta_of(L, Bd);
        if (t > 0.0) {
            double ny1 = k1 * t;
            if (ny1 < n) {
                M->ny1[c] = ny1;
                M->m[c] = n - ny1;
                c++;
            }
        }
    }
    double m = k1 * L->tile;
    if (m < n) {
        M->ny1[c] = n - m;
        M->m[c] = m;
        c++;
    }
    m = n * 0.5;
    M->ny1[c] = n - m;
    M->m[c] = m;
    c++;
    m = n / 3.0;
    M->ny1[c] = n - m;
    M->m[c] = m;
    c++;
    for (int t = 0; t < c; t++) {
        M->has_add[t] = 0;
        M->add[t] = 0.0;
        if (M->ny1[t] < SNAP_REL * n) {
            M->has_add[t] = 1;
            M->add[t] = M->ny1[t];
            M->ny1[t] = 0.0;
        }
        M->a[t] = cost_with(L, S, p1, M->ny1[t]);
    }
    M->c = c;
}

/* (ny1, ny2) filling pair p2's copies up to their tile boundary (minus the margin) from p1; returns 0 when there is no such move */
static int room_move(const Lay *L, const St *S, int p1, int p2, double *ny1, double *ny2)
{
    double n = S->y[p1], y2 = S->y[p2];
    int k2 = L->pk[p2];
    double x2 = y2 / k2, q2 = q_of(L, x2);
    double b_up = L->tile * ((q2 > 1.0) ? q2 : 1.0);
    double t2 = b_up - delta_of(L, b_up);
    double target = k2 * t2;
    double m = target - y2;
    if (!(m > 0.0 && m < n))
        return 0;
    *ny1 = n - m;
    *ny2 = target;
    if (*ny1 < SNAP_REL * n) {
        *ny2 = *ny2 + *ny1;
        *ny1 = 0.0;
    }
    return 1;
}

/* the candidate moves p1 -> p2 in their fixed order (src_moves, then the room move): (n1, n2, a); returns the count */
static int pair_moves(const Lay *L, const St *S, const Src *M, int p1, int p2, double *n1, double *n2, double *a)
{
    int c = 0;
    for (int t = 0; t < M->c; t++) {
        double ny2 = S->y[p2] + M->m[t];
        if (M->has_add[t])
            ny2 = ny2 + M->add[t];
        n1[c] = M->ny1[t];
        n2[c] = ny2;
        a[c] = M->a[t];
        c++;
    }
    double r1, r2;
    if (room_move(L, S, p1, p2, &r1, &r2)) {
        n1[c] = r1;
        n2[c] = r2;
        a[c] = cost_with(L, S, p1, r1);
        c++;
    }
    return c;
}

static void apply_move(const Lay *L, St *S, int p1, int p2, double ny1, double ny2)
{
    S->y[p1] = ny1;
    S->y[p2] = ny2;
    refresh(L, S, L->pr[p1]);
    refresh(L, S, L->pr[p2]);
}

/* The hot polish on fractional rows: the best (smallest new max of the two ranks) move off the max rank while it lowers the
 * max by > 1e-9; returns the iterations applied.  A candidate whose source cost alone cannot win is skipped before its target cost
 * is computed (same result as evaluating every candidate). */
static int hot_polish(const Lay *L, St *S, int iters)
{
    double n1[MAX_MOVES], n2[MAX_MOVES], av[MAX_MOVES];
    Src M;
    int done = 0;
    for (int it = 0; it < iters; it++) {
        int hi = argmax(S->cost, L->R);
        double lim = S->cost[hi] - HOT_EPS;
        int bp1 = -1, bp2 = -1;
        double bmm = 0.0, by1 = 0.0, by2 = 0.0;
        for (int j = L->rptr[hi]; j < L->rptr[hi + 1]; j++) {
            int p1 = L->ridx[j];
            if (!(S->y[p1] > 0.0))
                continue;
            int i = L->pexp[p1], have_src = 0;
            for (int p2 = L->hptr[i]; p2 < L->hptr[i + 1]; p2++) {
                if (p2 == p1)
                    continue;
                if (!have_src) {
                    src_moves(L, S, p1, &M);
                    have_src = 1;
                }
                int c = pair_moves(L, S, &M, p1, p2, n1, n2, av);
                for (int t = 0; t < c; t++) {
                    double a = av[t];
                    if (a >= lim || (bp1 >= 0 && a >= bmm))
                        continue;
                    double b = cost_with(L, S, p2, n2[t]);
                    double mm = (b > a) ? b : a;
                    if (mm < lim && (bp1 < 0 || mm < bmm)) {
                        bmm = mm;
                        bp1 = p1;
                        bp2 = p2;
                        by1 = n1[t];
                        by2 = n2[t];
                    }
                }
            }
        }
        if (bp1 < 0)
            break;
        apply_move(L, S, bp1, bp2, by1, by2);
        done++;
    }
    return done;
}

/* canonical LSE terms: ev[r] = exp((c_r - M) / tau), Z = sum ev (r ascending); returns J */
static double lse_terms(const double *cost, int R, double tau, double *M_out, double *Z_out, double *ev)
{
    double M = cost[0];
    for (int r = 1; r < R; r++)
        if (cost[r] > M)
            M = cost[r];
    double Z = 0.0;
    for (int r = 0; r < R; r++) {
        ev[r] = det_exp((cost[r] - M) / tau);
        Z = Z + ev[r];
    }
    *M_out = M;
    *Z_out = Z;
    return M + tau * det_log(Z);
}

/* the `top` costliest ranks by (cost descending, rank ascending) */
static int top_ranks(const double *cost, int R, int top, char *chosen, int *out)
{
    int k = (top < R) ? top : R;
    memset(chosen, 0, (size_t)R);
    for (int t = 0; t < k; t++) {
        int b = -1;
        for (int r = 0; r < R; r++) {
            if (chosen[r])
                continue;
            if (b < 0 || cost[r] > cost[b])
                b = r;
        }
        chosen[b] = 1;
        out[t] = b;
    }
    return k;
}

/* The LSE polish on fractional rows: from the top ranks, moves of an expert's rows to a cheaper holder, the best (smallest J)
 * kept while J drops by > 1e-6.  A candidate's J' = M + tau log z (z = its partition sum relative to the current max M): the best is
 * the smallest z below zlim = exp((J - 1e-6 - M) / tau), first in enumeration order.  Exact pruning before the exp: J' >= max(a, b),
 * and J' >= lb = J + grad J . dc (LSE is convex), so a candidate with lb >= the limit, or lb > the best J' so far + LSE_SLACK
 * (>> rounding), cannot win. */
static int lse_polish(const Lay *L, St *S, int iters, double tau, int top, double *ev, char *chosen, int *order)
{
    double n1[MAX_MOVES], n2[MAX_MOVES], av[MAX_MOVES], M, Z;
    Src SM;
    int done = 0;
    double J = lse_terms(S->cost, L->R, tau, &M, &Z, ev);
    for (int it = 0; it < iters; it++) {
        double lim = J - LSE_EPS;
        double zlim = det_exp((lim - M) / tau);
        int bp1 = -1, bp2 = -1;
        double bz = 0.0, bo = 0.0, by1 = 0.0, by2 = 0.0;
        int k = top_ranks(S->cost, L->R, top, chosen, order);
        for (int t1 = 0; t1 < k; t1++) {
            int r1 = order[t1];
            double c1 = S->cost[r1];
            double g1 = ev[r1] / Z;
            for (int j = L->rptr[r1]; j < L->rptr[r1 + 1]; j++) {
                int p1 = L->ridx[j];
                if (!(S->y[p1] > 0.0))
                    continue;
                int i = L->pexp[p1], have_src = 0;
                for (int p2 = L->hptr[i]; p2 < L->hptr[i + 1]; p2++) {
                    if (p2 == p1)
                        continue;
                    int r2 = L->pr[p2];
                    double c2 = S->cost[r2];
                    if (!(c2 < c1))
                        continue;
                    double g2 = ev[r2] / Z;
                    if (!have_src) {
                        src_moves(L, S, p1, &SM);
                        have_src = 1;
                    }
                    int c = pair_moves(L, S, &SM, p1, p2, n1, n2, av);
                    for (int t = 0; t < c; t++) {
                        double a = av[t];
                        if (a >= lim)
                            continue;
                        double b = cost_with(L, S, p2, n2[t]);
                        if (b >= lim)
                            continue;
                        double lb = J + (g1 * (a - c1) + g2 * (b - c2));
                        if (lb >= lim || (bp1 >= 0 && lb > bo + LSE_SLACK))
                            continue;
                        double z = Z - ev[r1];
                        z = z - ev[r2];
                        if (z < 0.0)
                            z = 0.0;
                        z = z + det_exp((a - M) / tau);
                        z = z + det_exp((b - M) / tau);
                        if (z < zlim && (bp1 < 0 || z < bz)) {
                            bz = z;
                            bp1 = p1;
                            bp2 = p2;
                            by1 = n1[t];
                            by2 = n2[t];
                            bo = M + tau * det_log(z);
                        }
                    }
                }
            }
        }
        if (bp1 < 0)
            break;
        apply_move(L, S, bp1, bp2, by1, by2);
        J = lse_terms(S->cost, L->R, tau, &M, &Z, ev);
        done++;
    }
    return done;
}

typedef struct {
    St S;
    double *ev;
    char *chosen;
    int *order;
} Work;

/* raw -> hot polish -> LSE polish (tau > 0.06): the best state's J / y / stage, the raw J, iterations */
static double run_start(const Lay *L, Work *W, const double *y0, double *best_y, int *stage, double *J_raw, int *iters)
{
    St *S = &W->S;
    state_init(L, S, y0);
    double J0 = lse(S->cost, L->R, L->tau);
    double bestJ = J0;
    memcpy(best_y, S->y, sizeof(double) * (size_t)L->P);
    *stage = 0;
    int it = hot_polish(L, S, L->polish_iters);
    double J1 = lse(S->cost, L->R, L->tau);
    if (J1 < bestJ) {
        bestJ = J1;
        memcpy(best_y, S->y, sizeof(double) * (size_t)L->P);
        *stage = 1;
    }
    if (L->tau > TAU_MAX_RULE) {
        it += lse_polish(L, S, L->polish_iters, L->tau, L->top, W->ev, W->chosen, W->order);
        double J2 = lse(S->cost, L->R, L->tau);
        if (J2 < bestJ) {
            bestJ = J2;
            memcpy(best_y, S->y, sizeof(double) * (size_t)L->P);
            *stage = 2;
        }
    }
    *J_raw = J0;
    *iters = it;
    return bestJ;
}

/* ------------------------------------------------------------------------------------------- S_B: bisection + Dinic */
typedef struct {
    int n, src, snk, n_edges;
    int *ptr, *idx, *to, *eu, *deg, *fill;   /* CSR of the edges in creation order */
    double *cap0, *cap, *best;
    int *mid, *rank_edge;                    /* [P] edge of an active pair (-1), [R] */
    int *level, *it, *queue;
    double need, eps;
} Flow;

static double fdfs(Flow *G, int u, double f)
{
    if (u == G->snk)
        return f;
    int base = G->ptr[u], deg = G->ptr[u + 1] - G->ptr[u];
    while (G->it[u] < deg) {
        int ei = G->idx[base + G->it[u]];
        int v = G->to[ei];
        if (G->cap[ei] > G->eps && G->level[v] == G->level[u] + 1) {
            double c = G->cap[ei];
            double d = fdfs(G, v, (c < f) ? c : f);
            if (d > G->eps) {
                G->cap[ei] -= d;
                G->cap[ei ^ 1] += d;
                return d;
            }
        }
        G->it[u] += 1;
    }
    return 0.0;
}

static double fmaxflow(Flow *G)
{
    double flow = 0.0;
    for (;;) {
        for (int i = 0; i < G->n; i++)
            G->level[i] = -1;
        G->level[G->src] = 0;
        int head = 0, tail = 0;
        G->queue[tail++] = G->src;
        while (head < tail) {
            int u = G->queue[head++];
            for (int k = G->ptr[u]; k < G->ptr[u + 1]; k++) {
                int ei = G->idx[k];
                int v = G->to[ei];
                if (G->level[v] < 0 && G->cap[ei] > G->eps) {
                    G->level[v] = G->level[u] + 1;
                    G->queue[tail++] = v;
                }
            }
        }
        if (G->level[G->snk] < 0)
            return flow;
        for (int i = 0; i < G->n; i++)
            G->it[i] = 0;
        for (;;) {
            double f = fdfs(G, G->src, INFINITY);
            if (f <= G->eps)
                break;
            flow += f;
        }
    }
}

/* src 0 -> expert 1 + i (cap N_i) -> rank 1 + m + r of each ACTIVE pair (cap big) -> sink (cap set per T); the arrays of G are
 * allocated by the caller for the largest graph (all pairs active) */
static void flow_build(const Lay *L, Flow *G, const char *act)
{
    int m = L->m, R = L->R, k = 0;
    G->n = m + R + 2;
    G->src = 0;
    G->snk = m + R + 1;
    double need = 0.0;
    for (int i = 0; i < m; i++)
        need = need + L->N[i];
    double big = 2.0 * need;
    memset(G->deg, 0, sizeof(int) * (size_t)G->n);
#define EDGE(u, v, c) do { G->to[k] = (v); G->cap0[k] = (c); G->eu[k] = (u); G->deg[u]++; \
                           G->to[k + 1] = (u); G->cap0[k + 1] = 0.0; G->eu[k + 1] = (v); G->deg[v]++; k += 2; } while (0)
    for (int i = 0; i < m; i++) {
        EDGE(G->src, 1 + i, L->N[i]);
        for (int p = L->hptr[i]; p < L->hptr[i + 1]; p++) {
            G->mid[p] = -1;
            if (act[p]) {
                G->mid[p] = k;
                EDGE(1 + i, 1 + m + L->pr[p], big);
            }
        }
    }
    for (int r = 0; r < R; r++) {
        G->rank_edge[r] = k;
        EDGE(1 + m + r, G->snk, 0.0);
    }
#undef EDGE
    G->n_edges = k;
    G->ptr[0] = 0;
    for (int u = 0; u < G->n; u++) {
        G->ptr[u + 1] = G->ptr[u] + G->deg[u];
        G->fill[u] = 0;
    }
    for (int e = 0; e < k; e++) {
        int u = G->eu[e];
        G->idx[G->ptr[u] + G->fill[u]++] = e;
    }
    G->need = need;
    G->eps = FLOW_EPS_REL * need;
}

static void count_active(const Lay *L, const char *act, int *nact)
{
    memset(nact, 0, sizeof(int) * (size_t)L->R);
    for (int p = 0; p < L->P; p++)
        if (act[p])
            nact[L->pr[p]] += L->pk[p];
}

/* feasibility at T; on success G->cap holds the residual caps */
static int feasible(Lay *L, Flow *G, const int *nact, double T)
{
    memcpy(G->cap, G->cap0, sizeof(double) * (size_t)G->n_edges);
    for (int r = 0; r < L->R; r++) {
        double b = T - L->Ff[r] - L->cf * nact[r];
        if (b < 0.0)
            return 0;
        G->cap[G->rank_edge[r]] = capfn(L, b, nact[r]);
    }
    double f = fmaxflow(G);
    L->flows++;
    return f >= G->need * (1.0 - FLOW_NEED_REL);
}

/* plain bisection to tol; have_hi: G->best holds feasible caps at hi.  Returns 1 with G->best = the final feasible caps, else 0 */
static int bisect(Lay *L, Flow *G, const int *nact, double lo, double hi, int have_hi, double *T_out)
{
    int have = have_hi, steps = 0;
    while (hi - lo > L->tol && steps < MAX_BISECT) {
        double mid = 0.5 * (lo + hi);
        if (feasible(L, G, nact, mid)) {
            hi = mid;
            memcpy(G->best, G->cap, sizeof(double) * (size_t)G->n_edges);
            have = 1;
        } else {
            lo = mid;
        }
        steps++;
    }
    if (!have && feasible(L, G, nact, hi)) {
        memcpy(G->best, G->cap, sizeof(double) * (size_t)G->n_edges);
        have = 1;
    }
    *T_out = hi;
    return have;
}

static void flow_split(const Lay *L, const Flow *G, const char *act, const double *cap, double *y)
{
    for (int p = 0; p < L->P; p++)
        y[p] = 0.0;
    for (int i = 0; i < L->m; i++) {
        double tot = 0.0;
        for (int p = L->hptr[i]; p < L->hptr[i + 1]; p++)
            if (act[p])
                tot = tot + cap[G->mid[p] ^ 1];
        if (tot > 0.0) {
            for (int p = L->hptr[i]; p < L->hptr[i + 1]; p++)
                if (act[p])
                    y[p] = L->N[i] * cap[G->mid[p] ^ 1] / tot;
        } else {
            y[L->home[i]] = L->N[i];
        }
    }
}

static double lo_bound(const Lay *L, const int *nact)
{
    double lo = L->Ff[0] + L->cf * nact[0];
    for (int r = 1; r < L->R; r++) {
        double v = L->Ff[r] + L->cf * nact[r];
        if (v > lo)
            lo = v;
    }
    return lo;
}

typedef struct {
    Flow G[2];                    /* current graph, trial graph */
    char *act, *act2, *tried;     /* [P] */
    int *nact, *nact2;            /* [R] */
    double *load, *lc, *X;        /* [R] */
} BWork;

static void swap_flow(Flow *a, Flow *b)
{
    Flow t = *a;
    *a = *b;
    *b = t;
}

/* S_B: y; returns 1 when the flow solved (else y = home only) */
static int solve_b(Lay *L, BWork *B, const double *yC, double *y, double *T_out, int *accepted, int *tried_n)
{
    int R = L->R, P = L->P;
    double theta = L->cf / (L->cl / L->tile + L->cc);
    if (theta < L->tile / 8.0)
        theta = L->tile / 8.0;
    if (theta > L->tile)
        theta = L->tile;
    memset(B->act, 0, (size_t)P);
    for (int i = 0; i < L->m; i++)
        B->act[L->home[i]] = 1;
    for (int p = 0; p < P; p++)
        if (yC[p] / L->pk[p] >= theta)
            B->act[p] = 1;
    count_active(L, B->act, B->nact);
    for (int r = 0; r < R; r++)
        B->X[r] = 0.0;
    for (int i = 0; i < L->m; i++)
        B->X[L->pr[L->home[i]]] += L->N[i];
    double hi = 0.0;
    for (int r = 0; r < R; r++) {
        double ex = B->X[r] - L->tile * B->nact[r];
        double v = L->Ff[r] + L->cf * B->nact[r] + L->cc * B->X[r] + (L->cl / L->tile) * ((ex > 0.0) ? ex : 0.0);
        if (r == 0 || v > hi)
            hi = v;
    }
    Flow *G = &B->G[0], *G2 = &B->G[1];
    flow_build(L, G, B->act);
    double T;
    *accepted = *tried_n = 0;
    if (!bisect(L, G, B->nact, lo_bound(L, B->nact), hi, 0, &T)) {
        for (int p = 0; p < P; p++)
            y[p] = 0.0;
        for (int i = 0; i < L->m; i++)
            y[L->home[i]] = L->N[i];
        *T_out = T;
        return 0;
    }
    flow_split(L, G, B->act, G->best, y);
    memset(B->tried, 0, (size_t)P);
    while (*tried_n < L->max_act) {
        /* closed-form screen: rows of the max (linearised) cost rank a not yet active holder could take at the current T */
        for (int r = 0; r < R; r++) {
            double s = 0.0;
            for (int j = L->rptr[r]; j < L->rptr[r + 1]; j++)
                s = s + y[L->ridx[j]];
            B->load[r] = s;
        }
        for (int r = 0; r < R; r++) {
            double ex = B->load[r] - L->tile * B->nact[r];
            B->lc[r] = L->Ff[r] + L->cf * B->nact[r] + L->cc * B->load[r] + (L->cl / L->tile) * ((ex > 0.0) ? ex : 0.0);
        }
        int h = argmax(B->lc, R);
        int cand = -1;
        double score = 0.0;
        for (int j = L->rptr[h]; j < L->rptr[h + 1]; j++) {
            int p1 = L->ridx[j];
            if (!(y[p1] > 0.0))
                continue;
            int i = L->pexp[p1];
            for (int p2 = L->hptr[i]; p2 < L->hptr[i + 1]; p2++) {
                if (B->act[p2] || B->tried[p2])
                    continue;
                int r2 = L->pr[p2];
                int n2 = B->nact[r2] + L->pk[p2];
                double b = T - L->Ff[r2] - L->cf * n2;
                if (b < 0.0)
                    continue;
                double room = capfn(L, b, n2) - B->load[r2];
                if (!(room > 0.0))
                    continue;
                double a = (y[p1] < room) ? y[p1] : room;
                if (cand < 0 || a > score) {
                    cand = p2;
                    score = a;
                }
            }
        }
        if (cand < 0)
            break;
        (*tried_n)++;
        memcpy(B->act2, B->act, (size_t)P);
        B->act2[cand] = 1;
        count_active(L, B->act2, B->nact2);
        flow_build(L, G2, B->act2);
        int ok = feasible(L, G2, B->nact2, T);
        double T2 = T;
        int have = 0;
        if (ok) {
            memcpy(G2->best, G2->cap, sizeof(double) * (size_t)G2->n_edges);
            double lo2 = 0.7 * T, lb = lo_bound(L, B->nact2);
            if (lb > lo2)
                lo2 = lb;
            have = bisect(L, G2, B->nact2, lo2, T, 1, &T2);
        }
        if (ok && have && T2 < T) {
            char *ta = B->act;
            B->act = B->act2;
            B->act2 = ta;
            int *tn = B->nact;
            B->nact = B->nact2;
            B->nact2 = tn;
            swap_flow(G, G2);
            T = T2;
            flow_split(L, G, B->act, G->best, y);
            (*accepted)++;
            memset(B->tried, 0, (size_t)P);
        } else {
            B->tried[cand] = 1;
        }
    }
    *T_out = T;
    return 1;
}

/* ------------------------------------------------------------------------------------------- entry point */
static int check_par(const double *par)
{
    for (int k = 0; k < N_PAR; k++)
        if (!isfinite(par[k]))
            return 0;
    double cf = par[0], cl = par[1], cc = par[2], ch = par[3], tau = par[4], tile = par[5], margin = par[6];
    if (cf < 0.0 || cl < 0.0 || cc < 0.0 || ch < 0.0 || tau < 0.0 || margin < 0.0 || !(tile > 0.0))
        return 0;
    if (!(cc > 0.0 || cl > 0.0))
        return 0;
    if (par[7] < 0.0 || par[8] < 0.0 || par[9] < 0.0 || par[10] < 0.0 || par[11] < 0.0 || !(par[12] > 0.0) || par[13] < 1.0)
        return 0;
    if (par[7] > MAX_INT_PAR || par[8] > MAX_INT_PAR || par[9] > MAX_INT_PAR || par[11] > MAX_INT_PAR || par[13] > MAX_INT_PAR)
        return 0;
    if (!(par[14] == 0.0 || par[14] == 1.0))
        return 0;
    if (par[14] == 1.0 && !(par[15] > 0.0 && par[15] <= DISP_MAX))
        return 0;
    return 1;
}

#define ALLOC(p, n) do { (p) = calloc((size_t)(n) + 1, sizeof(*(p))); if (!(p)) goto out; } while (0)

/* l3b_weights: E experts, R ranks, LS slots per rank, load[E] (rows per replay), phys[R * LS] (expert or -1), par[N_PAR]
 * (cf, cl, cc, ch, tau, tile, margin, max_act_steps, polish_iters, c_mode, c_band_rel, c_max_newton, bisect_tol, lse_top, pricing,
 * disp): pricing 0 (mean, disp ignored) or 1 (poisson, 0 < disp <= DISP_MAX).
 * Outputs w_out[R * LS], diag_out[N_DIAG] (shares.DIAG_NAMES).  Returns 0; -1 allocation failure; -2 phys out of range or an
 * expert with rows but no routed slot; -3 bad load or parameters (w_out / diag_out zero on error). */
int l3b_weights(int E, int R, int LS, const double *load, const int *phys, const double *par, double *w_out, double *diag_out)
{
    int ret = -1, n_slots = R * LS;
    Lay Ls;
    memset(&Ls, 0, sizeof(Ls));
    Lay *L = &Ls;
    Work W;
    memset(&W, 0, sizeof(W));
    BWork B;
    memset(&B, 0, sizeof(B));
    double *wC = NULL, *yC = NULL, *yCb = NULL, *yB0 = NULL, *yBb = NULL, *yA0 = NULL, *yAb = NULL;
    int *cnt = NULL, *fill = NULL;
    char *is_var = NULL, *seen = NULL;
    for (int s = 0; s < n_slots; s++)
        w_out[s] = 0.0;
    for (int k = 0; k < N_DIAG; k++)
        diag_out[k] = 0.0;
    if (E <= 0 || R <= 0 || LS <= 0 || !check_par(par))
        return -3;
    for (int e = 0; e < E; e++)
        if (!isfinite(load[e]) || load[e] > MAX_LOAD)
            return -3;
    ALLOC(cnt, E);
    for (int s = 0; s < n_slots; s++) {
        if (phys[s] < -1 || phys[s] >= E) {
            ret = -2;
            goto out;
        }
        if (phys[s] >= 0)
            cnt[phys[s]]++;
    }
    for (int e = 0; e < E; e++)
        if (load[e] > 0.0 && cnt[e] == 0) {
            ret = -2;
            goto out;
        }
    L->E = E; L->R = R; L->LS = LS; L->n_slots = n_slots;
    L->cf = par[0]; L->cl = par[1]; L->cc = par[2]; L->ch = par[3]; L->tau = par[4]; L->tile = par[5]; L->margin = par[6];
    L->max_act = (int)par[7];
    L->polish_iters = (int)par[8];
    L->tol = par[12];
    L->top = (int)par[13];
    L->pricing = (par[14] == 1.0) ? 1 : 0;
    L->disp = par[15];
    if (L->pricing) {
        ALLOC(L->qc_x, QCACHE);
        ALLOC(L->qc_q, QCACHE);
    }
    {
        int ex;
        L->inv_tile = (frexp(L->tile, &ex) == 0.5 && ex - 1 >= -1000 && ex - 1 <= 1000) ? ldexp(1.0, 1 - ex) : 0.0;
    }
    /* C's split */
    ALLOC(wC, n_slots);
    {
        double Tc = 0.0;
        int improved = 0, stats[2] = {0, 0};
        int rc = l3_weights_full(E, R, LS, load, n_slots, phys, 0, par[5], wC, &Tc, &improved, (int)par[9], par[10], (int)par[11], stats);
        if (rc != 0) {
            ret = (rc == -2) ? -2 : -1;
            goto out;
        }
        diag_out[10] = (double)stats[0];
        diag_out[11] = (double)stats[1];
        diag_out[19] = improved ? 1.0 : 0.0;
    }
    /* partition */
    ALLOC(L->sptr, E + 1);
    ALLOC(L->sidx, n_slots);
    ALLOC(fill, E);
    for (int e = 0; e < E; e++)
        L->sptr[e + 1] = L->sptr[e] + cnt[e];
    for (int s = 0; s < n_slots; s++)
        if (phys[s] >= 0)
            L->sidx[L->sptr[phys[s]] + fill[phys[s]]++] = s;
    int homes = (E % R == 0) ? E / R : 0;
    ALLOC(is_var, E);
    ALLOC(seen, R);
    ALLOC(L->var, E);
    int m = 0, P = 0;
    for (int e = 0; e < E; e++) {
        if (!(load[e] > 0.0))
            continue;
        memset(seen, 0, (size_t)R);
        int nr = 0;
        for (int k = L->sptr[e]; k < L->sptr[e + 1]; k++)
            if (!seen[L->sidx[k] / LS]) {
                seen[L->sidx[k] / LS] = 1;
                nr++;
            }
        if (nr >= 2) {
            is_var[e] = 1;
            L->var[m++] = e;
            P += nr;
        }
    }
    L->m = m;
    L->P = P;
    ALLOC(L->N, m);
    ALLOC(L->pexp, P);
    ALLOC(L->pr, P);
    ALLOC(L->pk, P);
    ALLOC(L->hptr, m + 1);
    ALLOC(L->home, m);
    {
        int *rc_ = NULL;
        ALLOC(rc_, R);
        int p = 0;
        for (int i = 0; i < m; i++) {
            int e = L->var[i];
            L->N[i] = load[e];
            memset(rc_, 0, sizeof(int) * (size_t)R);
            for (int k = L->sptr[e]; k < L->sptr[e + 1]; k++)
                rc_[L->sidx[k] / LS]++;
            int h = -1;
            for (int r = 0; r < R; r++)
                if (rc_[r] > 0) {
                    if (homes > 0 && r == e / homes)
                        h = p;
                    L->pexp[p] = i;
                    L->pr[p] = r;
                    L->pk[p] = rc_[r];
                    p++;
                }
            if (h < 0)
                h = L->hptr[i];
            L->home[i] = h;
            L->hptr[i + 1] = p;
        }
        free(rc_);
    }
    ALLOC(L->rptr, R + 1);
    ALLOC(L->ridx, P);
    {
        int *f2 = NULL;
        ALLOC(f2, R);
        for (int p = 0; p < P; p++)
            L->rptr[L->pr[p] + 1]++;
        for (int r = 0; r < R; r++)
            L->rptr[r + 1] += L->rptr[r];
        for (int r = 0; r < R; r++)
            f2[r] = L->rptr[r];
        for (int p = 0; p < P; p++)
            L->ridx[f2[L->pr[p]]++] = p;
        free(f2);
    }
    ALLOC(L->fixF, R);
    ALLOC(L->fixQ, R);
    ALLOC(L->Ff, R);
    for (int s = 0; s < n_slots; s++) {
        int e = phys[s];
        if (e < 0 || is_var[e] || !(load[e] > 0.0))
            continue;
        double x = load[e] / (L->sptr[e + 1] - L->sptr[e]);
        int r = s / LS;
        double q, f = fq(L, x, &q);
        L->fixF[r] = L->fixF[r] + f;
        if (q > L->fixQ[r])
            L->fixQ[r] = q;
    }
    for (int r = 0; r < R; r++)
        L->Ff[r] = (L->fixQ[r] > 1.0) ? L->fixF[r] + L->ch * (L->fixQ[r] - 1.0) : L->fixF[r];
    /* work arrays */
    ALLOC(W.S.y, P); ALLOC(W.S.fv, P); ALLOC(W.S.qv, P); ALLOC(W.S.qex, P);
    ALLOC(W.S.lin, R); ALLOC(W.S.Q, R); ALLOC(W.S.cost, R);
    ALLOC(W.ev, R); ALLOC(W.chosen, R); ALLOC(W.order, R);
    ALLOC(yC, P); ALLOC(yCb, P); ALLOC(yB0, P); ALLOC(yBb, P); ALLOC(yA0, P); ALLOC(yAb, P);
    /* C's split as pair rows */
    for (int p = 0; p < P; p++) {
        int i = L->pexp[p], e = L->var[i], s0 = -1;
        for (int k = L->sptr[e]; k < L->sptr[e + 1]; k++)
            if (L->sidx[k] / LS == L->pr[p]) {
                s0 = L->sidx[k];
                break;
            }
        yC[p] = L->N[i] * wC[s0] * L->pk[p];
    }
    state_init(L, &W.S, yC);
    double J_C_raw = lse(W.S.cost, R, L->tau);
    diag_out[16] = (double)m;
    int rep_copies = 0;
    for (int p = 0; p < P; p++)
        rep_copies += L->pk[p];
    diag_out[8] = (double)rep_copies;
    if (m == 0) {
        for (int s = 0; s < n_slots; s++)
            w_out[s] = wC[s];
        diag_out[0] = diag_out[1] = diag_out[2] = diag_out[3] = diag_out[4] = diag_out[17] = diag_out[18] = J_C_raw;
        ret = 0;
        goto out;
    }
    int sC, sB, sA, itC, itB, itA, acc = 0, tried_n = 0;
    double J0, JB0, JA0, TB = 0.0;
    double JC = run_start(L, &W, yC, yCb, &sC, &J0, &itC);
    /* S_B work: graphs sized for every pair active */
    {
        int nmax = m + R + 2, emax = 2 * (m + P + R);
        for (int g = 0; g < 2; g++) {
            Flow *G = &B.G[g];
            ALLOC(G->ptr, nmax + 1); ALLOC(G->idx, emax); ALLOC(G->to, emax); ALLOC(G->eu, emax); ALLOC(G->deg, nmax);
            ALLOC(G->fill, nmax); ALLOC(G->cap0, emax); ALLOC(G->cap, emax); ALLOC(G->best, emax); ALLOC(G->mid, P);
            ALLOC(G->rank_edge, R); ALLOC(G->level, nmax); ALLOC(G->it, nmax); ALLOC(G->queue, nmax);
        }
        ALLOC(B.act, P); ALLOC(B.act2, P); ALLOC(B.tried, P);
        ALLOC(B.nact, R); ALLOC(B.nact2, R);
        ALLOC(B.load, R); ALLOC(B.lc, R); ALLOC(B.X, R);
    }
    int okB = solve_b(L, &B, yC, yB0, &TB, &acc, &tried_n);
    double JB = run_start(L, &W, yB0, yBb, &sB, &JB0, &itB);
    for (int p = 0; p < P; p++)
        yA0[p] = 0.0;
    for (int i = 0; i < m; i++)
        yA0[L->home[i]] = L->N[i];
    double JA = run_start(L, &W, yA0, yAb, &sA, &JA0, &itA);
    int pick = 0, sp = sC;
    double Jp = JC;
    const double *yp = yCb;
    if (JB < Jp) {
        pick = 1; Jp = JB; yp = yBb; sp = sB;
    }
    if (JA < Jp) {
        pick = 2; Jp = JA; yp = yAb; sp = sA;
    }
    if (pick == 0 && sp == 0) {
        for (int s = 0; s < n_slots; s++)
            w_out[s] = wC[s];
    } else {
        for (int e = 0; e < E; e++) {
            int len = L->sptr[e + 1] - L->sptr[e];
            for (int k = L->sptr[e]; k < L->sptr[e + 1]; k++)
                w_out[L->sidx[k]] = 1.0 / len;
        }
        for (int p = 0; p < P; p++) {
            int i = L->pexp[p], e = L->var[i];
            double v = (yp[p] / L->pk[p]) / L->N[i];
            for (int k = L->sptr[e]; k < L->sptr[e + 1]; k++)
                if (L->sidx[k] / LS == L->pr[p])
                    w_out[L->sidx[k]] = v;
        }
    }
    int active = 0;
    for (int p = 0; p < P; p++) {
        int e = L->var[L->pexp[p]];
        for (int k = L->sptr[e]; k < L->sptr[e + 1]; k++)
            if (L->sidx[k] / LS == L->pr[p] && w_out[L->sidx[k]] > 0.0)
                active++;
    }
    diag_out[0] = Jp;
    diag_out[1] = J_C_raw;
    diag_out[2] = JC;
    diag_out[3] = JB;
    diag_out[4] = JA;
    diag_out[5] = (double)pick;
    diag_out[6] = (double)sp;
    diag_out[7] = (double)active;
    diag_out[9] = (double)L->flows;
    diag_out[12] = (double)acc;
    diag_out[13] = (double)tried_n;
    diag_out[14] = (double)(itC + itB + itA);
    diag_out[15] = TB;
    diag_out[17] = JB0;
    diag_out[18] = JA0;
    diag_out[20] = (double)(rep_copies - active);
    diag_out[21] = okB ? 0.0 : 1.0;
    ret = 0;
out:
    if (ret != 0) {
        for (int s = 0; s < n_slots; s++)
            w_out[s] = 0.0;
        for (int k = 0; k < N_DIAG; k++)
            diag_out[k] = 0.0;
    }
    for (int g = 0; g < 2; g++) {
        Flow *G = &B.G[g];
        free(G->ptr); free(G->idx); free(G->to); free(G->eu); free(G->deg); free(G->fill); free(G->cap0); free(G->cap);
        free(G->best); free(G->mid); free(G->rank_edge); free(G->level); free(G->it); free(G->queue);
    }
    free(B.act); free(B.act2); free(B.tried); free(B.nact); free(B.nact2); free(B.load); free(B.lc); free(B.X);
    free(W.S.y); free(W.S.fv); free(W.S.qv); free(W.S.qex); free(W.S.lin); free(W.S.Q); free(W.S.cost);
    free(W.ev); free(W.chosen); free(W.order);
    free(yC); free(yCb); free(yB0); free(yBb); free(yA0); free(yAb);
    free(L->sptr); free(L->sidx); free(L->var); free(L->N); free(L->pexp); free(L->pr); free(L->pk); free(L->hptr);
    free(L->home); free(L->rptr); free(L->ridx); free(L->fixF); free(L->fixQ); free(L->Ff); free(L->qc_x); free(L->qc_q);
    free(wC); free(cnt); free(fill); free(is_var); free(seen);
    return ret;
}

/* the library's parameter / diagnostic layout (csolvers.py checks it) */
int l3b_layout(int *n_par, int *n_diag)
{
    *n_par = N_PAR;
    *n_diag = N_DIAG;
    return 1;
}
