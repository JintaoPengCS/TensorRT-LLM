# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerics of GenEPLB (tensorrt_llm/_torch/moe/fused_moe/geneplb: csolvers, shares, costmodel, native, csrc/).

CPU parts need only numpy, gcc / g++ (no torch, no tensorrt_llm import: the numpy modules are loaded standalone, as the
planner worker loads them); the CUDA parts are skipped without torch and a GPU. The golden values were computed with the
shipped GenEPLB reference modules (l3_route_solver, l3b_solver, wroute_ref / l3_route_weights, lean_route_device,
eplb_gate, costmodel_b) on the inputs generated below (E 384, EP 32, 16 slots per rank).
"""

import hashlib
import importlib
import importlib.machinery
import importlib.util
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

GENEPLB_DIR = (
    Path(__file__).resolve().parents[4]
    / "tensorrt_llm"
    / "_torch"
    / "moe"
    / "fused_moe"
    / "geneplb"
)
HAS_GCC = shutil.which("gcc") is not None
HAS_GXX = shutil.which("g++") is not None
E, R, M = 384, 32, 16
S = R * M
M32 = (1 << 32) - 1


def _standalone():
    """geneplb's numpy-only modules as a private package: no tensorrt_llm import, no torch (as the planner worker)."""
    name = "_geneplb_numerics_under_test"
    if name not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(name, None, is_package=True)
        )
        pkg.__path__ = [str(GENEPLB_DIR)]
        sys.modules[name] = pkg
    return tuple(
        importlib.import_module(f"{name}.{m}") for m in ("csolvers", "shares", "costmodel")
    )


csolvers, shares, costmodel = _standalone()


@pytest.fixture(scope="module")
def solvers(tmp_path_factory):
    if not HAS_GCC:
        pytest.skip("gcc needed for the share solvers")
    shares.configure(tmp_path_factory.mktemp("geneplb-cache"))
    yield shares
    shares.configure(None)


# [gen-begin] deterministic inputs (pure Python: the same values on every platform and numpy version)
class _Rng:
    """splitmix64."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):
        return (self.u64() >> 11) * 2.0**-53

    def below(self, n):
        return self.u64() % n


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def gen_load(rng, experts, total, zero_frac=0.0, f32=False):
    """Zipf-like rows per expert in a random expert order, some experts idle."""
    order = list(range(experts))
    for i in range(experts - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    alpha = 0.6 + 0.8 * rng.uniform()
    raw = [0.0] * experts
    for rank, e in enumerate(order):
        raw[e] = (
            0.0 if rng.uniform() < zero_frac else (1.0 + 0.25 * rng.uniform()) / (rank + 1) ** alpha
        )
    scale = total / sum(raw)
    return [_f32(x * scale) if f32 else x * scale for x in raw]


def gen_layout(rng, experts, ranks, local, hot, fill):
    """Homes (expert r * H + j at slot r * M + j) plus replicas of `hot` experts on FREE slots, never twice on a
    rank."""
    h = experts // ranks
    lay = [r * h + j if j < h else -1 for r in range(ranks) for j in range(local)]
    for r in range(ranks):
        held = set(lay[r * local : r * local + h])
        for j in range(h, local):
            if rng.uniform() < fill:
                e = hot[rng.below(len(hot))]
                if e not in held:
                    lay[r * local + j] = e
                    held.add(e)
    return lay


def golden_case(i):
    """(T, load, layout, EMA steps) of golden case i at E 384, EP 32, 16 slots per rank."""
    rng = _Rng(20261104 + 7919 * i)
    T = (32, 64, 128, 256)[i % 4]
    load = gen_load(rng, E, float(T * 6 * R), zero_frac=0.1 * (i % 2), f32=(i % 3 == 0))
    hot = sorted(range(E), key=lambda e: (-load[e], e))[: 8 + 16 * i]
    return T, load, gen_layout(rng, E, R, M, hot, 0.25 + 0.15 * i), 1 + i


# [gen-end]


def _sha(a, dtype):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()[:16]


# Golden values: the shipped reference modules on golden_case(i) (sha256[:16] of the little-endian float64 / int32
#   bytes).
# rows / tables / cumw_rows: l3_route_solver.weights (l3_core.c, guided), lean_route_device.tables_from_layout,
#   l3_route_weights.cumw_np
# b / cumw_b: l3b_solver.weights_b(load x ema_scale(steps), par_vector(costmodel_b.l3b_params(prior_params(T),
#   settings)))
#   us_T: eplb_gate.gt1_layer_us(form T product model, load, layout, rows weights)
#   us_B: eplb_gate.gt1_layer_us(costmodel_b.gate_model(prior_params(T)), load, layout, B weights)
# (GOLDEN, GOLDEN_PRIOR_SHA: costmodel_b.prior_params(T, local_batch=T // 4, layers=0..60)['sha256'], GOLDEN_EMA:
# eplb_layout.ema_scale(steps) are at the end of this file.)
RUN_PARAMS_SHA = "ffac495598d382f3ec190cdc1759c06628fd8a2e2b7b7863b48426686d158a8c"  # prodB1 run, params.b8.json (T 32)


# ================================================================================================ standalone import
def test_standalone_import_has_no_torch():
    code = (
        "import sys; sys.path.insert(0, %r); import csolvers, shares, costmodel; "
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.') or m.startswith('tensorrt_llm')]; "
        "assert not bad, bad; print('ok')" % str(GENEPLB_DIR)
    )
    done = subprocess.run(
        [sys.executable, "-s", "-c", code], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr


def test_prior_file_is_the_shipped_one():
    raw = (GENEPLB_DIR / "data" / "costmodel_b_prior.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "931939287edf6e0c6a48e5400eff426cd55014445e0f3ec2ef37cb60b53ce732"
    )


# ================================================================================================ solvers
def test_solvers_build_and_solve(solvers, tmp_path):
    lib = csolvers.build_rows_solver(tmp_path / "c")
    assert csolvers.build_rows_solver(tmp_path / "c") is lib  # loaded once
    blib = csolvers.build_b_solver(tmp_path / "c")
    assert blib.l3b_det_exp(0.0) == 1.0 and blib.l3b_det_log(1.0) == 0.0
    assert (
        abs(blib.l3b_det_exp(1.0) - math.e) < 1e-14
        and abs(blib.l3b_det_log(10.0) - math.log(10.0)) < 1e-14
    )
    assert len(list((tmp_path / "c" / "solvers").glob("*.so"))) == 2
    T, load, phys, steps = golden_case(1)
    w = solvers.solve_rows(load, phys, E, R, M)
    ph = np.asarray(phys)
    # shares of every expert sum to 1 over its copies; FREE slots 0
    sums = np.bincount(ph[ph >= 0], weights=w[ph >= 0], minlength=E)
    assert np.allclose(sums, 1.0, atol=1e-12) and np.all(w[ph < 0] == 0.0)
    info = {}
    wb = solvers.solve_b(
        load, phys, shares.ema_scale(steps), M, costmodel.prior_params(T), info=info
    )
    assert wb is not None and info["pick"] in ("C", "B", "A")
    sums = np.bincount(ph[ph >= 0], weights=wb[ph >= 0], minlength=E)
    assert np.allclose(sums, 1.0, atol=1e-9)


def test_solve_b_refuses_like_the_product(solvers):
    T, load, phys, _ = golden_case(0)
    gone = [(-1 if e == 5 else e) for e in phys]  # expert 5 has rows but no slot: rc -2
    load = list(load)
    load[5] = 10.0
    info = {}
    assert (
        solvers.solve_b(load, gone, 1.0, M, costmodel.prior_params(T), info=info) is None
        and info["rc"] == -2
    )
    bad = dict(costmodel.l3b_params(costmodel.prior_params(T)), tau=-1.0)  # rc -3
    assert solvers.solve_b(load, phys, 1.0, M, bad) is None
    with pytest.raises(ValueError):
        solvers.solve_b(load, phys, 0.0, M, costmodel.prior_params(T))


@pytest.mark.parametrize("i", range(6))
def test_golden_shares(solvers, i):
    g = GOLDEN[i]
    T, load, phys, steps = golden_case(i)
    w = solvers.solve_rows(load, phys, E, R, M)
    assert _sha(w, "<f8") == g["rows"]
    tabs = shares.tables_from_phys(phys, E)
    assert _sha(np.concatenate(tabs), "<i4") == g["tables"]
    assert _sha(shares.cumw_from_weights(w, phys, tabs), "<i4") == g["cumw_rows"]
    wb = solvers.solve_b(load, phys, shares.ema_scale(steps), M, costmodel.prior_params(T))
    assert _sha(wb, "<f8") == g["b"]
    assert _sha(shares.cumw_from_weights(wb, phys, tabs), "<i4") == g["cumw_b"]
    assert math.isclose(
        costmodel.layer_us(costmodel.GATE_MODEL_T, load, phys, w, R), g["us_T"], rel_tol=1e-12
    )
    assert math.isclose(
        costmodel.layer_us(costmodel.prior_params(T), load, phys, wb, R), g["us_B"], rel_tol=1e-12
    )


# ================================================================================================ tables / cumw
@pytest.mark.parametrize("i", range(6))
def test_tables_round_trip(i):
    _, _, phys, _ = golden_case(i)
    count, start, slots = shares.tables_from_phys(phys, E)
    assert count.dtype == start.dtype == slots.dtype == np.int32
    assert int(count.sum()) == sum(1 for e in phys if e >= 0) and np.all(slots[count.sum() :] == -1)
    back = [-1] * S
    for e in range(E):
        row = slots[start[e] : start[e] + count[e]]
        assert list(row) == sorted(
            row
        )  # ascending slot order within an expert (makeTables(layout, original(layout)))
        for q in row:
            back[q] = e
    assert back == phys


def _decode_cumw(cw, count, start):
    """Per replicated expert: the copies' shares from the Q32 cumulative words (last copy: the rest)."""
    u = np.asarray(cw, dtype=np.int64) & M32
    out = {}
    for e in np.nonzero(count > 1)[0]:
        c, s = int(count[e]), int(start[e])
        edges = [0] + [int(u[s + k]) for k in range(c - 1)] + [1 << 32]
        out[int(e)] = [(edges[k + 1] - edges[k]) / 2.0**32 for k in range(c)]
    return out


@pytest.mark.parametrize("i", range(6))
def test_cumw_round_trip(i):
    _, _, phys, _ = golden_case(i)
    rng = _Rng(99 + i)
    tabs = shares.tables_from_phys(phys, E)
    count, start, slots = tabs
    w = np.array([rng.uniform() if e >= 0 else 0.0 for e in phys])
    cw = shares.cumw_from_weights(w, phys, tabs)
    assert cw.dtype == np.int32 and cw.shape == (S,)
    rep = count > 1
    for e, got in _decode_cumw(cw, count, start).items():
        ws = np.array([w[q] for q in slots[start[e] : start[e] + count[e]]])
        assert np.allclose(got, ws / ws.sum(), atol=4.0 * 2.0**-32)
        assert cw[start[e] + count[e] - 1] == -1  # the last copy's word is unused (M32)
    used = np.zeros(S, dtype=bool)
    for e in np.nonzero(rep)[0]:
        used[start[e] : start[e] + count[e] - 1] = True
    assert np.all(cw[~used] == -1)
    # even weights -> equal Q32 shares; zero total -> equal shares; negative weights count as 0
    even = _decode_cumw(
        shares.cumw_from_weights(shares.even_weights(phys), phys, tabs), count, start
    )
    assert all(np.allclose(v, 1.0 / len(v), atol=2.0**-32) for v in even.values())
    zero = _decode_cumw(shares.cumw_from_weights(np.zeros(S), phys, tabs), count, start)
    assert all(np.allclose(v, 1.0 / len(v), atol=2.0**-32) for v in zero.values())
    assert np.array_equal(
        shares.cumw_from_weights(np.where(w > 0.5, w, -w), phys, tabs),
        shares.cumw_from_weights(np.where(w > 0.5, w, 0.0), phys, tabs),
    )
    with pytest.raises(ValueError):
        shares.cumw_from_weights(w, [(e + 1) % E if e >= 0 else -1 for e in phys], tabs)


def test_cumw_reference_semantics():
    """Small hand-checked case: expert 0 on slots 0, 2, 3 with weights 1, 2, 1; expert 1 on slot 1 alone."""
    phys = [0, 1, 0, 0]
    cw = shares.cumw_from_weights([1.0, 7.0, 2.0, 1.0], phys, shares.tables_from_phys(phys, 2))
    assert list(cw) == [1 << 30, (3 << 30) - (1 << 32), -1, -1]


# ================================================================================================ cost model / gate
def test_prior_params_and_settings():
    p = costmodel.prior_params(32, local_batch=8, layers=list(range(61)))
    assert p["sha256"] == RUN_PARAMS_SHA
    for T, sha in GOLDEN_PRIOR_SHA.items():
        assert (
            costmodel.prior_params(T, local_batch=T // 4, layers=list(range(61)))["sha256"] == sha
        )
    for T, key, tier, mode in (
        (4, 32, 32, "inline"),
        (32, 32, 32, "inline"),
        (45, 32, 32, "inline"),
        (48, 64, 32, "inline"),
        (64, 64, 64, "inline"),
        (91, 128, 64, "overlap"),
        (128, 128, 128, "overlap"),
        (256, 256, 128, "overlap"),
        (512, 256, 128, "overlap"),
    ):
        s = costmodel.settings_for_T(T)
        assert (s["prior_key"], s["tier"], s["mode"], s["extrapolated"]) == (
            key,
            tier,
            mode,
            T < 32,
        )
        assert costmodel.prior_params(T)["prior_key"] == key
    low = costmodel.prior_params(16)
    assert low["extrapolated"] is True and low["cf"] == costmodel.prior_params(32)["cf"]
    assert "extrapolated" not in costmodel.prior_params(32)
    assert costmodel.l3b_params(costmodel.prior_params(32)) == dict(
        cf=5.43707,
        cl=1.68627,
        cc=0.0082,
        ch=1.85443,
        tau=0.5,
        tile=256.0,
        margin=3.0,
        polish_iters=64.0,
        pricing=1.0,
        disp=8.0,
    )
    assert costmodel.l3b_params(costmodel.prior_params(64))["margin"] == 1.5
    assert "pricing" not in costmodel.l3b_params(costmodel.prior_params(64))
    assert costmodel.is_b(costmodel.prior_params(64)) and not costmodel.is_b(costmodel.GATE_MODEL_T)


def test_ema_scale():
    for steps, want in GOLDEN_EMA.items():
        assert shares.ema_scale(steps) == want
    with pytest.raises(ValueError):
        shares.ema_scale(0)


# ================================================================================================ kernels: host
#   logic (g++)
def _section(text, begin, end):
    i = text.index(begin)
    return text[i : text.index(end, i) + len(end)]


def _spread_limit(sr, rows, rows_ref):
    if sr < 0 or rows_ref <= 0 or sr >= rows_ref or rows > rows_ref or rows <= 0:
        return rows
    limit = -(-(sr * rows) // rows_ref)
    return rows if limit >= rows else limit


def ref_route(
    ids, ep_rank, tables, cumw, wmode, word, rows_ref, experts, slots_total, top_k, ranks
):
    """The route kernels' rule in plain Python: the padding rule, then the native (wmode 0) or the weighted rule per
    1024-element block; ids outside [0, E) and count-0 experts -> the sentinel slot id ``slots_total``."""
    counts, starts, slots = (list(map(int, a)) for a in tables)
    cumw = [int(x) & M32 for x in cumw]
    n = len(ids)
    first = _spread_limit(word, n // top_k, rows_ref) * top_k if word is not None else n
    routed = [
        x if f < first else (f - first + ep_rank * (experts // ranks)) % experts
        for f, x in enumerate(ids)
    ]
    out = []
    for b in range((n + 1023) // 1024):
        exps = [x if 0 <= x < experts else experts for x in routed[b * 1024 : (b + 1) * 1024]]
        arbs = [e if e < experts and counts[e] > 1 else experts for e in exps]
        tot, js = {}, []
        for a in arbs:
            js.append(tot.get(a, 0))
            tot[a] = tot.get(a, 0) + 1
        for e, a, j in zip(exps, arbs, js):
            if e >= experts or counts[e] <= 0:
                out.append(slots_total)
                continue
            c, s = counts[e], starts[e]
            if wmode == 0:
                out.append(slots[s + ((j + b + ep_rank) % c if c > 1 else 0)])
                continue
            k = 0
            if c > 1:
                step = M32 if tot[a] <= 1 else (1 << 32) // tot[a]
                u = (e * 0x632BE5AB + (ep_rank * 64 + b) * 0x9E3779B9 + j * step) & M32
                while k < c - 1 and u >= cumw[s + k]:
                    k += 1
            out.append(slots[s + k])
    return routed, out


_PRELUDE = (
    "#define __host__\n"
    "#define __device__\n"
    "#define __forceinline__ inline\n"
    "#include <cstdio>\n"
    "#include <cstdint>\n"
    "#include <vector>\n"
)
_HARNESS = r"""
int main(int argc,char** argv){
 if(argc<2)return 2;
 FILE* f=fopen(argv[1],"r");int nc=0;if(fscanf(f,"%d",&nc)!=1)return 4;
 std::vector<int> arb(LR_BLOCK),counts(LR_E),starts(LR_E),slots(LR_SLOTS),tot(LR_E),e_buf(LR_BLOCK),rank_buf(LR_BLOCK);
 std::vector<unsigned short> cm(LR_CHUNKS*LR_E);std::vector<uint32_t> cumw_s(LR_SLOTS);
 std::vector<int> mb_e(LR_BLOCK),mb_a(LR_BLOCK),mb_r(LR_BLOCK),mb_tot(LR_E),mb_bef(LR_E);
 std::vector<unsigned char> mb_cnt(MB_WARPS*LR_E);
 for(int c=0;c<nc;++c){
  int ep=0,n=0,wmode=0,haspad=0,word=0,rows_ref=0;
  if(fscanf(f,"%d %d %d %d %d %d",&ep,&n,&wmode,&haspad,&word,&rows_ref)!=6)return 5;
  std::vector<int> cg(LR_E),sg(LR_E),slg(LR_SLOTS),ids(n>0?n:1),o1(n>0?n:1),st(n>0?n:1),o2(n>0?n:1);
  std::vector<uint32_t> cw(LR_SLOTS);
  for(int i=0;i<LR_E;++i)if(fscanf(f,"%d",&cg[i])!=1)return 6;
  for(int i=0;i<LR_E;++i)if(fscanf(f,"%d",&sg[i])!=1)return 6;
  for(int i=0;i<LR_SLOTS;++i)if(fscanf(f,"%d",&slg[i])!=1)return 6;
  for(int i=0;i<LR_SLOTS;++i)if(fscanf(f,"%u",&cw[i])!=1)return 6;
  for(int i=0;i<n;++i)if(fscanf(f,"%d",&ids[i])!=1)return 6;
  int first=haspad?pad_first(word,n/PAD_K,rows_ref):n;
  for(int b=0;b<(n+LR_BLOCK-1)/LR_BLOCK;++b){
   l3pad_emulate_block(ids.data(),n,first,b,ep,cg.data(),sg.data(),slg.data(),cw.data(),wmode,o1.data(),st.data(),arb.data(),cm.data(),
                       counts.data(),starts.data(),slots.data(),cumw_s.data(),tot.data(),e_buf.data(),rank_buf.data());
   l3b_mb_emulate_rule_block(ids.data(),n,first,b,ep,cg.data(),sg.data(),slg.data(),cw.data(),wmode,o2.data(),counts.data(),starts.data(),
                             slots.data(),cumw_s.data(),mb_e.data(),mb_a.data(),mb_r.data(),mb_tot.data(),mb_bef.data(),mb_cnt.data());
  }
  for(int i=0;i<n;++i){printf("%d ",o1[i]);}
  printf("\n");
  for(int i=0;i<n;++i){printf("%d ",o2[i]);}
  printf("\n");
 }
 return 0;
}
"""
_STUBS = r"""
#define __global__
#define __launch_bounds__(x)
#define __shared__ static
#define __align__(x) __attribute__((aligned(x)))
struct geneplb_dim3{unsigned x,y,z;};
static geneplb_dim3 threadIdx,blockIdx,blockDim;
static int __syncthreads_or(int p){return p;}
static void __syncthreads(){}
static unsigned __match_any_sync(unsigned m,int v){return v>=0?m:m;}
static int __popc(unsigned v){return __builtin_popcount(v);}
template<class T> static T atomicAdd(T* p,T v){T o=*p;*p+=v;return o;}
template<class T> static T atomicOr(T* p,T v){T o=*p;*p|=v;return o;}
template<class T> static T atomicCAS(T* p,T c,T v){T o=*p;if(o==c)*p=v;return o;}
"""
# (experts, ranks, slots per rank, top-k): the shipped EP32 / EP16 geometries, a 448-slot EP16, a small EP4
GEOMETRIES = ((384, 32, 16, 6), (384, 16, 32, 6), (384, 16, 28, 6), (256, 4, 68, 6))


@pytest.mark.skipif(not HAS_GXX, reason="g++ needed")
@pytest.mark.parametrize("geo", GEOMETRIES)
def test_route_host_logic_generic_geometry(geo, tmp_path):
    """The kernels' [l3-host] / [l3b-host] logic (the C kernel's and the B kernel's host emulations, the same
    __host__ __device__ functions the device code calls) compiled for each geometry, against ref_route; the device
    kernels compile there too (g++ with CUDA built-in stubs)."""
    experts, ranks, local, top_k = geo
    slots_total = ranks * local
    cu_c = (GENEPLB_DIR / "csrc" / "l3_pad_route_kernels.cu").read_text()
    cu_b = (GENEPLB_DIR / "csrc" / "l3b_route_kernels.cu").read_text()
    region = _section(cu_c, "// [l3-host-begin]", "// [l3-host-end]")
    assert region == _section(cu_b, "// [l3-host-begin]", "// [l3-host-end]")
    l3b = _section(cu_b, "// [l3b-host-begin]", "// [l3b-host-end]")
    defines = (
        f"#define GENEPLB_E {experts}\n#define GENEPLB_SLOTS {slots_total}\n#define GENEPLB_RANKS {ranks}\n"
        f"#define GENEPLB_TOPK {top_k}\n"
    )
    head = _section(cu_c, "#ifndef GENEPLB_RANKS", '(shipped: 16, 32)");\n')
    src = tmp_path / "logic.cpp"
    src.write_text(_PRELUDE + defines + head + region + "\n" + l3b + "\n" + _HARNESS)
    exe = tmp_path / "logic"
    done = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    kernels = (
        _section(
            cu_c,
            "template<bool Sample=false>\n__global__ void __launch_bounds__(LR_BLOCK) l3_pad_route_k(",
            "\n}\n",
        )
        + _section(
            cu_b,
            "template<typename OutT>\n__global__ void __launch_bounds__(MB_TPB) l3b_mb_k(",
            "\n}\n",
        )
        + _section(
            cu_b,
            "template<typename OutT>\n__global__ void __launch_bounds__(MB_TPB) l3b_mbs_k(",
            "\n}\n",
        )
    )
    asserts = _section(
        cu_c, "// (port) generic geometry: LOAD stages", 'static shared memory <= 48 KB");\n'
    ) + _section(cu_b, "static_assert(2*MB_TPB>=LR_SLOTS", 'cover the counts / starts");\n')
    stub = tmp_path / "kernels.cpp"
    stub.write_text(
        _PRELUDE
        + _STUBS
        + defines
        + head
        + region
        + "\n"
        + l3b
        + "\n"
        + asserts
        + kernels
        + "int main(){auto a=&l3_pad_route_k<false>;auto b=&l3b_mb_k<int>;auto c=&l3b_mb_k<int64_t>;"
        "auto d=&l3b_mbs_k<int64_t>;return (int)(sizeof(a)+sizeof(b)+sizeof(c)+sizeof(d))*0;}\n"
    )
    done = subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-fsyntax-only",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            "-Wno-unknown-pragmas",
            str(stub),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    # cases: three layouts x ids families x padding words x both rules
    rng = _Rng(7 + experts + ranks)
    cases = []
    h = experts // ranks
    for li in range(3):
        load = gen_load(rng, experts, 1000.0)
        hot = sorted(range(experts), key=lambda e: (-load[e], e))[: 4 + 20 * li]
        phys = gen_layout(rng, experts, ranks, local, hot, (0.0, 0.5, 1.0)[li])
        tabs = shares.tables_from_phys(phys, experts)
        w = [rng.uniform() if rng.uniform() > 0.2 else 0.0 for _ in range(slots_total)]
        cw = [int(x) & M32 for x in shares.cumw_from_weights(w, phys, tabs)]
        for rows in (1, 171, 400):
            ids = [rng.below(experts) for _ in range(rows * top_k)]
            ids[rng.below(len(ids))] = (-1, experts + 3)[rows % 2]
            for haspad, word, rows_ref in (
                (0, 0, rows),
                (1, 0x7FFFFFFF, rows),
                (1, rows // 3, rows),
                (1, rows // 2, 4 * rows),
            ):
                for wmode in (0, 1):
                    cases.append((rng.below(ranks), ids, wmode, haspad, word, rows_ref, tabs, cw))
    assert h * ranks == experts
    path = tmp_path / "cases.txt"
    with path.open("w") as fh:
        fh.write(f"{len(cases)}\n")
        for ep, ids, wmode, haspad, word, rows_ref, tabs, cw in cases:
            fh.write(f"{ep} {len(ids)} {wmode} {haspad} {word} {rows_ref}\n")
            for row in (*tabs, cw, ids):
                fh.write(" ".join(str(int(x)) for x in row) + "\n")
    run = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.split("\n")
    for c, (ep, ids, wmode, haspad, word, rows_ref, tabs, cw) in enumerate(cases):
        _, want = ref_route(
            ids,
            ep,
            tabs,
            cw,
            wmode,
            word if haspad else None,
            rows_ref,
            experts,
            slots_total,
            top_k,
            ranks,
        )
        assert [int(x) for x in lines[2 * c].split()] == want, f"C host logic, case {c}"
        assert [int(x) for x in lines[2 * c + 1].split()] == want, f"B host logic, case {c}"


# ================================================================================================ CUDA (torch + GPU)
def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device needed")
    return torch, importlib.import_module("_geneplb_numerics_under_test.native")


def _tables(torch, dev, tabs, cw, wmode):
    native = importlib.import_module("_geneplb_numerics_under_test.native")
    t = [torch.tensor(np.asarray(a), dtype=torch.int32, device=dev) for a in (*tabs, cw)]
    return native.LayerTables(*t, torch.tensor([wmode], dtype=torch.int32, device=dev))


@pytest.mark.parametrize("variant", ["C", "B"])
def test_route_kernel_matches_reference(variant, tmp_path):
    torch, native = _cuda()
    dev = torch.device("cuda", torch.cuda.current_device())
    T, load, phys, _ = golden_case(3)
    rng = _Rng(31)
    tabs = shares.tables_from_phys(phys, E)
    w = np.array([rng.uniform() for _ in range(S)])
    cw = shares.cumw_from_weights(w, phys, tabs)
    for ep_rank in (0, 5, 31):
        kern = native.RouteKernel(variant, E, S, R, ep_rank, dev, cache_dir=tmp_path)
        for wmode in (0, 1):
            lt = _tables(torch, dev, tabs, cw, wmode)
            for rows, word, rows_ref in (
                (1, None, 1),
                (171, native.VALID_ALL, 171),
                (512, 300, 512),
                (128, 100, 512),
            ):
                ids = [rng.below(E) for _ in range(rows * 6)]
                ids[0] = -1
                t_ids = torch.tensor(ids, dtype=torch.int32, device=dev).view(rows, 6)
                vr = None if word is None else torch.tensor([word], dtype=torch.int32, device=dev)
                _, want = ref_route(ids, ep_rank, tabs, cw, wmode, word, rows_ref, E, S, 6, R)
                for dtype in (torch.int32, torch.int64):
                    got = kern(t_ids, lt, rows_ref, vr, dtype)
                    assert got.dtype == dtype and tuple(got.shape) == (rows, 6)
                    assert got.view(-1).tolist() == want


def test_route_stage_matches_route_and_staging(tmp_path):
    """B's stage entry == its route + MegaMoE's staging: slots then -1, weights then 0, other columns untouched."""
    torch, native = _cuda()
    dev = torch.device("cuda", torch.cuda.current_device())
    _, _, phys, _ = golden_case(3)
    rng = _Rng(37)
    tabs = shares.tables_from_phys(phys, E)
    cw = shares.cumw_from_weights(np.array([rng.uniform() for _ in range(S)]), phys, tabs)
    lt = _tables(torch, dev, tabs, cw, 1)
    kern = native.RouteKernel("B", E, S, R, 5, dev, cache_dir=tmp_path)
    for rows, bucket, word in (
        (0, 256, None),
        (1, 256, None),
        (171, 256, 168),
        (256, 256, None),
        (300, 1024, 297),
    ):
        ids = torch.randint(0, E, (rows, 6), dtype=torch.int32, device=dev)
        if rows:
            ids[0, 0] = -1
        w_in = torch.rand(rows, 6, dtype=torch.float32, device=dev)
        vr = torch.tensor(
            [native.VALID_ALL if word is None else word], dtype=torch.int32, device=dev
        )
        for dtype in (torch.int64, torch.int32):
            idx = torch.full(
                (bucket, 6), 7, dtype=dtype, device=dev
            )  # stale rows of an earlier launch
            w_out = torch.full((bucket, 8), 5.0, device=dev)
            kern.stage(ids, lt, max(rows, 1), vr, idx, w_in, w_out)
            if rows:
                assert torch.equal(idx[:rows], kern(ids, lt, rows, vr, dtype))
            assert bool((idx[rows:] == -1).all())
            assert torch.equal(w_out[:rows, :6], w_in) and bool((w_out[rows:, :6] == 0).all())
            assert bool((w_out[:, 6:] == 5.0).all())


def test_route_kernel_capturable(tmp_path):
    torch, native = _cuda()
    dev = torch.device("cuda", torch.cuda.current_device())
    _, _, phys, _ = golden_case(2)
    tabs = shares.tables_from_phys(phys, E)
    lt = _tables(
        torch, dev, tabs, shares.cumw_from_weights(shares.even_weights(phys), phys, tabs), 1
    )
    vr = torch.tensor([native.VALID_ALL], dtype=torch.int32, device=dev)
    ids = torch.randint(0, E, (64, 6), dtype=torch.int32, device=dev)
    for variant in ("C", "B"):
        kern = native.RouteKernel(variant, E, S, R, 3, dev, cache_dir=tmp_path)
        want = kern(ids, lt, 64, vr)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            kern(ids, lt, 64, vr)  # warm up on the side stream
            with torch.cuda.graph(graph, stream=stream):
                out = kern(ids, lt, 64, vr)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want)


def test_stats_kernel_matches_reference(tmp_path):
    torch, native = _cuda()
    dev = torch.device("cuda", torch.cuda.current_device())
    L = 5
    st = native.StatsKernel(L, E, R, dev, cache_dir=tmp_path)
    fused = native.StatsKernel(L, E, R, dev, cache_dir=tmp_path)
    rng = _Rng(5)
    enable = torch.ones(1, dtype=torch.int32, device=dev)
    F = np.zeros((L, E), dtype=np.float32)
    for vr_host in (native.VALID_ALL, 7, 0):
        vr = torch.tensor([vr_host], dtype=torch.int32, device=dev)
        rows = [rng.below(40) + 1 for _ in range(L)]
        ids = [
            torch.tensor([rng.below(E) for _ in range(r * 6)], dtype=torch.int32, device=dev).view(
                r, 6
            )
            for r in rows
        ]
        for li in range(L):
            st.record(li, ids[li], vr, enable)
        fused.record_fused([t.data_ptr() for t in ids], rows, vr, enable)
        for li in range(L):
            C = np.bincount(ids[li][: min(vr_host, rows[li])].view(-1).cpu().numpy(), minlength=E)
            F[li] = (F[li] * np.float32(0.95)).astype(np.float32) + C.astype(np.float32)
            assert np.array_equal(st.counts[li].cpu().numpy(), C)
        assert np.array_equal(st.load.cpu().numpy().view(np.int32), F.view(np.int32))
        assert torch.equal(st.load, fused.load) and torch.equal(st.counts, fused.counts)
    assert st.steps.tolist() == [3] * L
    st.reset()
    assert int(st.steps.sum()) == 0 and float(st.load.abs().sum()) == 0.0


# ================================================================================================ golden values
GOLDEN = [
    dict(
        rows="a5c36c30d90d52ed",
        tables="20ca1fd0e4172c7e",
        cumw_rows="d7b70434c1026135",
        b="fbf4e642f5e7fa54",
        cumw_b="2524e490e8187e32",
        us_T=47.780582869209496,
        us_B=73.93440805104854,
    ),  # B pick A (lse), 34 replicas
    dict(
        rows="2d22c617fc6f3d9f",
        tables="ad2b888256f228ab",
        cumw_rows="33ee114009472609",
        b="851c89191a77a39d",
        cumw_b="8ff4fc7fb8185231",
        us_T=58.56004667691016,
        us_B=80.01136622563827,
    ),  # B pick C (lse), 55 replicas
    dict(
        rows="1b06b4d66eb00cbd",
        tables="ccd2fefb48b3dfd2",
        cumw_rows="0fbe87ce81fec820",
        b="87a633ff2a0ba669",
        cumw_b="0e3d40e428161e08",
        us_T=71.7164119801772,
        us_B=96.18557682470106,
    ),  # B pick C (lse), 64 replicas
    dict(
        rows="1e7d90c6fa994a61",
        tables="bbda30f8066df0b2",
        cumw_rows="9c509cb7e7dec646",
        b="4b85a63a28e46ad2",
        cumw_b="dee066776655298b",
        us_T=186.87911464272557,
        us_B=204.3520957516458,
    ),  # B pick B (hot), 89 replicas
    dict(
        rows="7600f920fc651825",
        tables="3e03881737abd316",
        cumw_rows="d0ff1b294b5288d1",
        b="057a592a8f233b32",
        cumw_b="a66fe8598e88c739",
        us_T=53.42540617264983,
        us_B=72.28787193911195,
    ),  # B pick B (raw), 106 replicas
    dict(
        rows="2bd32cf755350e80",
        tables="cae16b3c92ea01c9",
        cumw_rows="f18348616a20d55e",
        b="270b610f1a9fc894",
        cumw_b="fbbc76428cf68965",
        us_T=62.968600970065225,
        us_B=90.55048080914554,
    ),  # B pick C (lse), 124 replicas
]
GOLDEN_PRIOR_SHA = {
    32: "ffac495598d382f3ec190cdc1759c06628fd8a2e2b7b7863b48426686d158a8c",
    48: "5c77d60a39b28de34938cfb59be682dc8d66e2e0fc03383e279668fba8b9aaef",
    64: "2b239aa1af5594876fdbbe6eea49a6133c5dc9abc74563f4e5a8da70ae0dcfd1",
    96: "35fe3febec04602c504482cd41f3fc312d21204a165504d99557f29a0ca1142b",
    128: "7abe229ab988d5b54a1e7764952f68dff2f4efd1907c21f52eb35afbd0681a86",
    256: "68589bdd7e4ebce4cbdc9765595dd5cdff179e643d1751aae7e28231e723afc8",
    512: "5f912479fe9e7d6b587609c5828bbb7cf789a4cf587ee10d0246a2f5ec85220e",
}
GOLDEN_EMA = {
    1: 1.0,
    2: 0.5128205128205131,
    3: 0.35056967572305,
    4: 0.269550860878062,
    5: 0.22102469812861147,
    6: 0.188744693841066,
    7: 0.1657480336566196,
    8: 0.14855328322956285,
    9: 0.13522628851724064,
    10: 0.1246065359345489,
    64: 0.051949354822278894,
    200: 0.05000175269474911,
}
