// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
// GenEPLB's route kernel of C: expert id -> physical slot inside the captured decode CUDA graph, with a per-layer weighted
// replica choice and a padding rule for the dummy rows of a padded graph replay.
//
// One kernel, two rules, chosen per launch by a fixed-address device word (a graph cannot switch kernels after capture):
//   wmode == 0  the native rule, bit for bit: slot = globalSlotIds[start_e + (j + blockIdx + epRank) % c] (TensorRT-LLM's
//               moeComputeRouteKernel, offsetByEpRank = true): the [lean-route-logic] section below, a lean re-implementation
//               of that kernel ("the lean kernel" below);
//   wmode != 0  the weighted rule of the [wroute-logic] section below: copy k of expert e takes the share given by the Q32
//               cumulative weights cumw[start_e + k] (globalSlotIds order of the CURRENT tables), systematic sampling with a fixed
//               Weyl start phase.
// Both rules share LOAD / ARB / RANK / PREFIX (the per-block ordinal j of an element among the replicated experts), so the only
// rule-specific code is the RESOLVE choice (l3_slot) and the block total that PREFIX already computes (wroute::wr_prefix_column =
// lr_prefix_column plus the returned total).  Fast path (no expert with more than one replica): slot = globalSlotIds[start_e]
// under both rules (count 1 -> copy 0).
//
// Padding ([spread-logic] and [pad-route-logic] sections): CUDA graph padding appends dummy requests as the trailing rows of a
// padded replay, and with the router's own choice all of them go to the same six experts.  The kernel l3_pad_route_k = l3_route_k
// with two more arguments (rows_ref, the stats valid_rows word) and a padding-aware LOAD: an element of a row >=
// spread_limit(valid_rows, rows, rows_ref) takes logical expert (j + ep_rank * E / RANKS) % E (j = its index among the launch's
// padding elements) instead of the router's choice; staging stores the id the route used.  Entry `route` has no padding word;
// entry `route_pad` adds rows_ref and the word.  A launch without padding (word >= rows_ref, the VALID_ALL sentinel, a full batch)
// routes bit for bit as `route`.  No new graph node, no new per-replay host write (the word is the stats valid_rows word), no
// staging copy.
//
// Per-layer inputs, all fixed addresses captured into the graph: the three GPU placement tables, the layer's cumw[SLOTS] row and
// its wmode word.  The host writes them only between replays on the execution stream (never while capturing): weights + wmode = 1
// when the layer's tables are committed and idle, wmode = 0 before any replay that may see republished tables, so cumw always
// describes the tables the kernel reads while wmode == 1.  path_counters = {fast, full native, full weighted} launches (block 0 /
// thread 0).
//
// All index logic is in the [l3-host-begin/end] region as __host__ __device__ functions; the host emulations (l3_emulate_block,
// l3pad_emulate_block) run the kernel's phases on the host, and test_geneplb_numerics.py compiles the region with g++ for several
// geometries against its Python reference route.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#ifndef GENEPLB_RANKS
#define GENEPLB_RANKS 16              // EP ranks (native.py passes -DGENEPLB_RANKS)
#endif
// geneplb/native.py builds this file with the geometry as -D macros; the defaults are DeepSeek-V4-Pro's (384 routed experts,
// 512 physical slots, top-6, 62 MoE layers).
#ifndef GENEPLB_E
#define GENEPLB_E 384                 // routed experts
#endif
#ifndef GENEPLB_SLOTS
#define GENEPLB_SLOTS 512             // physical slots of all ranks (ranks x slots per rank); also the invalid-expert sentinel slot id
#endif
#ifndef GENEPLB_TOPK
#define GENEPLB_TOPK 6                // top-k: a row is GENEPLB_TOPK consecutive flat elements
#endif
#ifndef GENEPLB_LAYERS
#define GENEPLB_LAYERS 62             // MoE layers with statistics (route_packed's diagnostic sample slab: the first LAYERS-1)
#endif
static_assert(GENEPLB_RANKS>=1&&GENEPLB_E%GENEPLB_RANKS==0&&GENEPLB_SLOTS%GENEPLB_RANKS==0,
              "supported EP sizes: the ranks divide the experts and the slots (shipped: 16, 32)");
// [l3-host-begin]
namespace {
// [lean-route-logic-begin]
constexpr int LR_E=GENEPLB_E;           // routed experts (metaInfo.expertCount)
constexpr int LR_SLOTS=GENEPLB_SLOTS;   // epSize * slotCountPerRank = 16 x 32; also the invalid-expert sentinel slot id
constexpr int LR_BLOCK=1024;            // flat (token, k) elements per block = moeComputeRouteDevice's 256 threads x 4 items
constexpr int LR_CHUNK=32;              // elements per chunk (one warp of the TRT-LLM arbitration loop)
constexpr int LR_CHUNKS=LR_BLOCK/LR_CHUNK;   // 32 chunks per block
constexpr int LR_CM_WORDS=LR_CHUNKS*LR_E/2;  // per-chunk per-expert counts as uint16: 12288 entries = 6144 uint32 words

// Phase LOAD (thread t of block b): the element's expert id, out-of-range ids and the padded tail mapped to the sentinel
// LR_E (moeComputeRouteKernel: expertIds[i] = tokenIdx < n ? ids[tokenIdx] : expertCount; then <0 || >=expertCount -> expertCount).
__host__ __device__ __forceinline__ int lr_sanitize(int raw){ return (raw<0||raw>=LR_E)?LR_E:raw; }
__host__ __device__ __forceinline__ int lr_expert(const int* ids,int n,int f){ return (f>=n)?LR_E:lr_sanitize(ids[f]); }
// Phase ARB: only experts with more than one replica take part in the arbitration (arbitrateExpertId = count > 1 ? e : E).
__host__ __device__ __forceinline__ int lr_arbitrate_id(int e,const int* counts){ return (e<LR_E&&counts[e]>1)?e:LR_E; }
// Phase RANK: within the element's 32-element chunk, rank = earlier elements with the same arbitration id (TRT-LLM:
// popc(match & lanemask_lt)), count = all elements of the chunk with that id (popc(match)).  Plain loop over the chunk,
// host-compilable; on the device the 32 reads of one warp hit the same shared address (broadcast).
__host__ __device__ __forceinline__ void lr_chunk_rank(const int* arb_chunk,int lane,int* rank,int* count){
 int a=arb_chunk[lane],r=0,c=0;
 for(int p=0;p<LR_CHUNK;++p){int m=(arb_chunk[p]==a)?1:0;c+=m;if(p<lane)r+=m;}
 *rank=r;*count=c;
}
// Phase PREFIX (thread e < LR_E): exclusive prefix over the chunks of expert e's per-chunk counts, in place -- cm[c][e]
// becomes the number of e-elements in chunks 0..c-1 of this block (TRT-LLM: the atomicAdd's returned oldVal for chunk c).
__host__ __device__ __forceinline__ void lr_prefix_column(unsigned short* cm,int e){
 int run=0;
 for(int c=0;c<LR_CHUNKS;++c){int v=cm[c*LR_E+e];cm[c*LR_E+e]=(unsigned short)run;run+=v;}
}
// Phase RESOLVE: the slot.  ordinal = cm[chunk][e] + rank (0 for non-arbitrated experts); TRT-LLM adds baseOffset =
// blockIdx + epRank and takes % count; count == 1 -> replica 0.  Defensive sentinels where TRT-LLM would read undefined
// memory (count <= 0 or a start/offset outside the 512-entry table): our tables never produce them.
__host__ __device__ __forceinline__ int lr_slot(int e,int ordinal,int block,int ep_rank,const int* counts,const int* starts,const int* slots){
 if(e<0||e>=LR_E)return LR_SLOTS;
 int c=counts[e],s=starts[e];
 if(c<=0||s<0||s>=LR_SLOTS)return LR_SLOTS;
 int idx=(c>1)?((ordinal+block+ep_rank)%c):0;
 int off=s+idx;
 if(off<0||off>=LR_SLOTS)return LR_SLOTS;
 return slots[off];
}
// FAST PATH predicate: does any expert have more than one replica?  (On the device the kernel evaluates the same OR with
// __syncthreads_or over the per-thread counts: block-uniform, one barrier.)  Without replicas the arbitration cannot change any
// decision -- TRT-LLM arbitrates only experts with count > 1 and routes count == 1 to replica 0 -- so slot = lr_slot(e, 0, ...).
__host__ __device__ __forceinline__ int lr_any_replica(const int* counts){ for(int e=0;e<LR_E;++e)if(counts[e]>1)return 1; return 0; }
// Host emulation of one block: the phases above for every thread in order (the kernel runs the same calls with
// __syncthreads between the phases).  Writes out[f] for the block's elements f < n.  Buffers: arb[LR_BLOCK],
// cm[LR_CHUNKS*LR_E], counts/starts[LR_E], slots[LR_SLOTS] (the shared copies), e_buf/rank_buf[LR_BLOCK].
// Returns 1 when the block took the fast path (no replicas), 0 for the full path.
__host__ inline int lr_emulate_block(const int* ids,int n,int b,int ep_rank,const int* counts_g,const int* starts_g,const int* slots_g,
 int* out,int* arb,unsigned short* cm,int* counts,int* starts,int* slots,int* e_buf,int* rank_buf){
 for(int t=0;t<LR_BLOCK;++t){                           // LOAD
  e_buf[t]=lr_expert(ids,n,b*LR_BLOCK+t);
  if(t<LR_E){counts[t]=counts_g[t];starts[t]=starts_g[t];}
  if(t<LR_SLOTS)slots[t]=slots_g[t];
 }
 if(!lr_any_replica(counts)){                           // FAST PATH: count == 1 everywhere -> replica 0, no arbitration
  for(int t=0;t<LR_BLOCK;++t){int f=b*LR_BLOCK+t;if(f<n)out[f]=lr_slot(e_buf[t],0,b,ep_rank,counts,starts,slots);}
  return 1;
 }
 for(int i=0;i<LR_CHUNKS*LR_E;++i)cm[i]=0;
 for(int t=0;t<LR_BLOCK;++t)arb[t]=lr_arbitrate_id(e_buf[t],counts);          // ARB
 for(int t=0;t<LR_BLOCK;++t){                                                 // RANK (leader writes the chunk count)
  int c=t/LR_CHUNK,lane=t%LR_CHUNK,r,cnt;
  lr_chunk_rank(arb+c*LR_CHUNK,lane,&r,&cnt);
  rank_buf[t]=r;
  int a=arb[t];
  if(r==0&&a<LR_E)cm[c*LR_E+a]=(unsigned short)cnt;
 }
 for(int e=0;e<LR_E;++e)lr_prefix_column(cm,e);                               // PREFIX
 for(int t=0;t<LR_BLOCK;++t){                                                 // RESOLVE
  int f=b*LR_BLOCK+t;
  if(f>=n)continue;
  int a=arb[t];
  int ordinal=(a<LR_E)?((int)cm[(t/LR_CHUNK)*LR_E+a]+rank_buf[t]):0;
  out[f]=lr_slot(e_buf[t],ordinal,b,ep_rank,counts,starts,slots);
 }
 return 0;
}
// [lean-route-logic-end]
}  // namespace
namespace wroute {
// [wroute-logic-begin]
constexpr int WR_E = GENEPLB_E;        // routed experts
constexpr int WR_SLOTS = GENEPLB_SLOTS; // physical slots of all ranks (e.g. 32 ranks x 16 slots); also the sentinel
constexpr int WR_BLOCK = 1024;         // flat elements per block (moeComputeRouteDevice: 256 threads x 4 items)
constexpr int WR_CHUNK = 32;
constexpr int WR_CHUNKS = WR_BLOCK / WR_CHUNK;
// Start phase: a Weyl sequence over the group index g = ep_rank * WR_GROUP_STRIDE + block (consecutive groups' starts are
// spread by the golden ratio, so the per-group remainders cancel over the 32 ranks), offset per expert.  Constants fixed in
// design on synthetic launches only, never on trajectories.
constexpr uint32_t WR_PHI_G = 0x9E3779B9u, WR_PHI_E = 0x632BE5ABu;
constexpr int WR_GROUP_STRIDE = 64;    // blocks per launch at decode <= 3 (3072 elements); prefill launches may exceed it

__host__ __device__ __forceinline__ int wr_expert(const int* ids, int n, int f) {
  if (f >= n) return WR_E;
  int raw = ids[f];
  return (raw < 0 || raw >= WR_E) ? WR_E : raw;
}
// arbitration id: only replicated experts take part (as natively)
__host__ __device__ __forceinline__ int wr_arb(int e, const int* counts) { return (e < WR_E && counts[e] > 1) ? e : WR_E; }
// rank of lane within its chunk among equal arbitration ids, and the chunk count of that id
__host__ __device__ __forceinline__ void wr_chunk_rank(const int* arb_chunk, int lane, int* rank, int* count) {
  int a = arb_chunk[lane], r = 0, c = 0;
  for (int p = 0; p < WR_CHUNK; ++p) { int m = (arb_chunk[p] == a) ? 1 : 0; c += m; if (p < lane) r += m; }
  *rank = r; *count = c;
}
// exclusive prefix over chunks of expert e's per-chunk counts (in place); returns the block total n_{e,b}
__host__ __device__ __forceinline__ int wr_prefix_column(unsigned short* cm, int e) {
  int run = 0;
  for (int c = 0; c < WR_CHUNKS; ++c) { int v = cm[c * WR_E + e]; cm[c * WR_E + e] = (unsigned short)run; run += v; }
  return run;
}
__host__ __device__ __forceinline__ uint32_t wr_phase(int e, int ep_rank, int block) {
  return (uint32_t)e * WR_PHI_E + (uint32_t)(ep_rank * WR_GROUP_STRIDE + block) * WR_PHI_G;
}
// step = floor(2^32 / n) for n >= 1 (n == 1 -> 2^32 - 1, any u_0 = phase works)
__host__ __device__ __forceinline__ uint32_t wr_step(int n) {
  return n <= 1 ? 0xFFFFFFFFu : (uint32_t)(((uint64_t)1 << 32) / (uint64_t)n);
}
// the slot of an element: e, its ordinal j in the block, the block total n of e
__host__ __device__ __forceinline__ int wr_slot(int e, int j, uint32_t step, int block, int ep_rank, const int* counts,
                                                const int* starts, const int* slots, const uint32_t* cumw) {
  if (e < 0 || e >= WR_E) return WR_SLOTS;
  int c = counts[e], s = starts[e];
  if (c <= 0 || s < 0 || s + c > WR_SLOTS) return WR_SLOTS;
  int k = 0;
  if (c > 1) {
    uint32_t u = wr_phase(e, ep_rank, block) + (uint32_t)j * step;   // mod 2^32
    while (k < c - 1 && u >= cumw[s + k]) ++k;
  }
  return slots[s + k];
}
#ifndef __CUDACC__
// Host emulation of one launch (all blocks, phases in order).  Scratch buffers sized by the caller.
inline void wr_emulate_launch(const int* ids, int n, int ep_rank, const int* counts, const int* starts, const int* slots,
                              const uint32_t* cumw, int* out, int* arb, unsigned short* cm, int* e_buf, int* rank_buf, int* tot) {
  int blocks = (n + WR_BLOCK - 1) / WR_BLOCK;
  for (int b = 0; b < blocks; ++b) {
    for (int t = 0; t < WR_BLOCK; ++t) { e_buf[t] = wr_expert(ids, n, b * WR_BLOCK + t); arb[t] = wr_arb(e_buf[t], counts); }
    for (int i = 0; i < WR_CHUNKS * WR_E; ++i) cm[i] = 0;
    for (int t = 0; t < WR_BLOCK; ++t) {
      int c = t / WR_CHUNK, lane = t % WR_CHUNK, r, cnt;
      wr_chunk_rank(arb + c * WR_CHUNK, lane, &r, &cnt);
      rank_buf[t] = r;
      if (r == 0 && arb[t] < WR_E) cm[c * WR_E + arb[t]] = (unsigned short)cnt;
    }
    for (int e = 0; e < WR_E; ++e) tot[e] = wr_prefix_column(cm, e);
    for (int t = 0; t < WR_BLOCK; ++t) {
      int f = b * WR_BLOCK + t;
      if (f >= n) continue;
      int a = arb[t];
      int j = (a < WR_E) ? ((int)cm[(t / WR_CHUNK) * WR_E + a] + rank_buf[t]) : 0;
      uint32_t step = (a < WR_E) ? wr_step(tot[a]) : 0u;
      out[f] = wr_slot(e_buf[t], j, step, b, ep_rank, counts, starts, slots, cumw);
    }
  }
}
#endif
// [wroute-logic-end]
}  // namespace wroute
namespace {
// [l3-route-logic-begin]
static_assert(LR_E==wroute::WR_E&&LR_SLOTS==wroute::WR_SLOTS&&LR_BLOCK==wroute::WR_BLOCK&&LR_CHUNK==wroute::WR_CHUNK,
              "the lean and the weighted rule share one block geometry");
// RESOLVE: wmode 0 -> the native rule (lean's lr_slot); otherwise the weighted rule with the block total of the element's
// arbitration id (n_{e,b}; only replicated experts are arbitrated, a non-arbitrated element has c == 1 and takes copy 0 either way).
__host__ __device__ __forceinline__ int l3_slot(int wmode,int e,int a,int ordinal,int total,int block,int ep_rank,const int* counts,
  const int* starts,const int* slots,const uint32_t* cumw){
 if(wmode==0)return lr_slot(e,ordinal,block,ep_rank,counts,starts,slots);
 return wroute::wr_slot(e,ordinal,(a<LR_E)?wroute::wr_step(total):0u,block,ep_rank,counts,starts,slots,cumw);
}
// Host emulation of one block of l3_route_k (the kernel runs the same calls with __syncthreads between the phases).  Buffers as
// lr_emulate_block plus cumw[LR_SLOTS] (the shared copy) and tot[LR_E].  Returns 1 when the block took the fast path.
__host__ inline int l3_emulate_block(const int* ids,int n,int b,int ep_rank,const int* counts_g,const int* starts_g,const int* slots_g,
 const uint32_t* cumw_g,int wmode,int* out,int* arb,unsigned short* cm,int* counts,int* starts,int* slots,uint32_t* cumw,int* tot,
 int* e_buf,int* rank_buf){
 for(int t=0;t<LR_BLOCK;++t){                           // LOAD
  e_buf[t]=lr_expert(ids,n,b*LR_BLOCK+t);
  if(t<LR_E){counts[t]=counts_g[t];starts[t]=starts_g[t];}
  if(t<LR_SLOTS){slots[t]=slots_g[t];cumw[t]=cumw_g[t];}
 }
 if(!lr_any_replica(counts)){                           // FAST PATH (both rules): count == 1 everywhere -> copy 0
  for(int t=0;t<LR_BLOCK;++t){int f=b*LR_BLOCK+t;if(f<n)out[f]=lr_slot(e_buf[t],0,b,ep_rank,counts,starts,slots);}
  return 1;
 }
 for(int i=0;i<LR_CHUNKS*LR_E;++i)cm[i]=0;
 for(int t=0;t<LR_BLOCK;++t)arb[t]=lr_arbitrate_id(e_buf[t],counts);          // ARB
 for(int t=0;t<LR_BLOCK;++t){                                                 // RANK
  int c=t/LR_CHUNK,lane=t%LR_CHUNK,r,cnt;
  lr_chunk_rank(arb+c*LR_CHUNK,lane,&r,&cnt);
  rank_buf[t]=r;
  int a=arb[t];
  if(r==0&&a<LR_E)cm[c*LR_E+a]=(unsigned short)cnt;
 }
 for(int e=0;e<LR_E;++e)tot[e]=wroute::wr_prefix_column(cm,e);              // PREFIX (+ block totals)
 for(int t=0;t<LR_BLOCK;++t){                                                 // RESOLVE
  int f=b*LR_BLOCK+t;
  if(f>=n)continue;
  int a=arb[t];
  int ordinal=(a<LR_E)?((int)cm[(t/LR_CHUNK)*LR_E+a]+rank_buf[t]):0;
  out[f]=l3_slot(wmode,e_buf[t],a,ordinal,(a<LR_E)?tot[a]:0,b,ep_rank,counts,starts,slots,cumw);
 }
 return 0;
}
// [l3-route-logic-end]
}  // namespace
namespace {
// The dummy-row spread's index logic of stats_kernels.cu, byte for byte.  Here `sr` is the stats valid_rows word: real rows of
// the main pass = real generation requests x (draft + 1), written by the host before every replay.
// [spread-logic-begin]
// First row to rewrite for one invocation: `sr` = spread_rows word (real rows of the MAIN pass, whose row count is
// `rows_ref` = rows of layer 0's first invocation in this captured forward), `rows` = rows of THIS invocation. The
// MTP draft re-invocation of a layer has fewer rows (one or (draft+1) per request, same request order, dummies still
// trailing), so the limit scales: ceil(sr * rows / rows_ref) -- ceil so a real row is never rewritten. Returns `rows`
// (nothing to do) for the sentinel, a negative word, a full batch, or an invocation larger than the reference.
__host__ __device__ __forceinline__ int spread_limit(int sr,int rows,int rows_ref){
 if(sr<0||rows_ref<=0||sr>=rows_ref||rows>rows_ref||rows<=0)return rows;
 long long num=(long long)sr*rows;
 long long limit=(num+rows_ref-1)/rows_ref;
 return limit>=rows?rows:(int)limit;
}
// Flat index i = r*K + k -> expert (r*K + k) % E: six consecutive, hence distinct, experts per row (no within-row
// duplicate for the route kernels), uniform over the logical experts across rows.
__host__ __device__ __forceinline__ int spread_value(int i,int e){ return i%e; }
// [spread-logic-end]
// [pad-route-logic-begin]
// The padding rows of a padded Graph replay (CUDA Graph padding appends dummy requests, input token id 1, as the TRAILING rows; with
// the router's own choice every one of them goes to the same six experts) take, inside the route kernel, a rule-based expert:
//   first padding element of a launch = spread_limit(valid_rows, rows, rows_ref) x PAD_K   (= rows x PAD_K: nothing is padding)
//   the j-th padding element of rank r -> logical expert (j + r x PAD_PHASE) % LR_E, PAD_PHASE = LR_E / RANKS (one rank's share)
// and the unchanged rule (native or weighted, fast path or full) maps that expert to a slot exactly as for a real element.
//   - a launch without padding routes bit for bit as l3_route_kernels.cu (the LOAD returns lr_sanitize(raw) for every element);
//   - padding elements are trailing (and spread_limit rounds up), so no real element is rewritten and a real element's ordinal --
//     the earlier elements of its expert in its block -- is unchanged;
//   - a padding row = PAD_K consecutive experts starting at a multiple of PAD_K (LR_E and PAD_PHASE are multiples): distinct experts;
//   - the phase rotates each rank's start by one rank's share of the experts, so the union of the ranks' padding is even over the
//     experts for any fill (an unphased (r x K + k) % E would send every rank's first padding rows to the same experts).
constexpr int PAD_K=GENEPLB_TOPK;            // top-k: a row is PAD_K consecutive flat elements
constexpr int PAD_PHASE=LR_E/GENEPLB_RANKS;  // 24 at EP16, 12 at EP32
// (port) the shipped geometries also have LR_E and PAD_PHASE multiples of PAD_K (a padding row never wraps the expert range); a
// generic one (e.g. 256 experts, top-6) may wrap: PAD_K consecutive values mod LR_E are still distinct experts for PAD_K <= LR_E.
static_assert(PAD_K>=1&&PAD_K<=LR_E&&PAD_PHASE*GENEPLB_RANKS==LR_E,"a padding row is PAD_K distinct experts; the rank phases tile the experts");
__host__ __device__ __forceinline__ int pad_first(int vr,int rows,int rows_ref){ return spread_limit(vr,rows,rows_ref)*PAD_K; }
__host__ __device__ __forceinline__ int pad_expert(int f,int first,int ep_rank){ return spread_value(f-first+ep_rank*PAD_PHASE,LR_E); }
// LOAD of flat element f of a launch of n elements whose padding starts at `first` (n: none): the logical expert the rule routes, and
// in *staged the id the kernel stages for the verification (the raw id of a real element, the pad expert of a padding element).
__host__ __device__ __forceinline__ int pad_load(const int* ids,int n,int f,int first,int ep_rank,int* staged){
 int raw=(f<n)?ids[f]:LR_E;
 if(f<n&&f>=first){int e=pad_expert(f,first,ep_rank);*staged=e;return e;}
 *staged=raw;
 return lr_sanitize(raw);
}
// Host emulation of one block of l3_pad_route_k: l3_emulate_block with pad_load as its LOAD; staged[f] for the block's elements f < n.
// Returns 1 when the block took the fast path.
__host__ inline int l3pad_emulate_block(const int* ids,int n,int first,int b,int ep_rank,const int* counts_g,const int* starts_g,
 const int* slots_g,const uint32_t* cumw_g,int wmode,int* out,int* staged,int* arb,unsigned short* cm,int* counts,int* starts,int* slots,
 uint32_t* cumw,int* tot,int* e_buf,int* rank_buf){
 for(int t=0;t<LR_BLOCK;++t){                           // LOAD (padding elements: the pad expert)
  int f=b*LR_BLOCK+t,s=0;
  e_buf[t]=pad_load(ids,n,f,first,ep_rank,&s);
  if(f<n)staged[f]=s;
  if(t<LR_E){counts[t]=counts_g[t];starts[t]=starts_g[t];}
  if(t<LR_SLOTS){slots[t]=slots_g[t];cumw[t]=cumw_g[t];}
 }
 if(!lr_any_replica(counts)){                           // FAST PATH (both rules): count == 1 everywhere -> copy 0
  for(int t=0;t<LR_BLOCK;++t){int f=b*LR_BLOCK+t;if(f<n)out[f]=lr_slot(e_buf[t],0,b,ep_rank,counts,starts,slots);}
  return 1;
 }
 for(int i=0;i<LR_CHUNKS*LR_E;++i)cm[i]=0;
 for(int t=0;t<LR_BLOCK;++t)arb[t]=lr_arbitrate_id(e_buf[t],counts);          // ARB
 for(int t=0;t<LR_BLOCK;++t){                                                 // RANK
  int c=t/LR_CHUNK,lane=t%LR_CHUNK,r,cnt;
  lr_chunk_rank(arb+c*LR_CHUNK,lane,&r,&cnt);
  rank_buf[t]=r;
  int a=arb[t];
  if(r==0&&a<LR_E)cm[c*LR_E+a]=(unsigned short)cnt;
 }
 for(int e=0;e<LR_E;++e)tot[e]=wroute::wr_prefix_column(cm,e);              // PREFIX (+ block totals)
 for(int t=0;t<LR_BLOCK;++t){                                                 // RESOLVE
  int f=b*LR_BLOCK+t;
  if(f>=n)continue;
  int a=arb[t];
  int ordinal=(a<LR_E)?((int)cm[(t/LR_CHUNK)*LR_E+a]+rank_buf[t]):0;
  out[f]=l3_slot(wmode,e_buf[t],a,ordinal,(a<LR_E)?tot[a]:0,b,ep_rank,counts,starts,slots,cumw);
 }
 return 0;
}
// [pad-route-logic-end]
}  // namespace
// [l3-host-end]

namespace {
constexpr int K=GENEPLB_TOPK;            // top-k of DeepSeek-V4-Pro
constexpr int MAX_ROWS=1<<16;            // host cap on rows per launch
constexpr int MISMATCH_WORDS=8;          // [count, first_ordinal, first_index, expected, got, elements_compared, launches_compared, 0]
constexpr int PATH_WORDS=3;              // {fast, full native, full weighted} launches
// (port) generic geometry: LOAD stages one table entry per thread; the kernel's static shared memory stays within 48 KB
static_assert(LR_E<=LR_SLOTS&&LR_SLOTS<=LR_BLOCK,"l3_pad_route_k: one thread per counts / starts / slots / cumw entry");
static_assert(4*LR_BLOCK+2*LR_CHUNKS*LR_E+3*4*LR_E+2*4*LR_SLOTS+4<=48*1024,"l3_pad_route_k: static shared memory <= 48 KB");

void ck(const torch::Tensor&t,torch::ScalarType d){
 TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==d&&t.get_device()==c10::cuda::current_device(),
  "current-device contiguous tensor required");}

// One thread per flat element; blockIdx.x = TRT-LLM's block (1024 elements).  LOAD stages the three tables and the layer's cumw row
// into shared memory in the same round trip as the element's id and reads the layer's wmode word (block-uniform); one barrier
// (__syncthreads_or over the replica counts) picks FAST or FULL; FULL is the lean kernel's ARB / RANK / PREFIX with the block totals
// kept, and RESOLVE takes the rule wmode names.  Staging as the lean kernel (verify_enable).
// l3_route_k + (rows_ref, valid_rows): the LOAD reads the padding word once per thread (one address, launch-uniform
// `first`) and gives a padding element its pad expert (pad_load); valid_rows == nullptr (entry `route`): first = n, no padding.  The
// staged id is the id the route used (verify_submit replays TRT-LLM's op on it); nothing else differs from l3_route_k.
template<bool Sample=false>
__global__ void __launch_bounds__(LR_BLOCK) l3_pad_route_k(const int* __restrict__ ids,int n,int ep_rank,
  const int* __restrict__ counts_g,const int* __restrict__ starts_g,const int* __restrict__ slots_g,
  const int* __restrict__ cumw_g,const int* __restrict__ wmode_g,int* __restrict__ out,
  const int* __restrict__ verify_enable,int* __restrict__ stage_ids,int* __restrict__ stage_slots,int* __restrict__ path_counters,
  int rows_ref,const int* __restrict__ valid_rows,int* sample=nullptr,const int* sample_steps=nullptr,
  int sample_layer=0,int sample_capacity=0,int sample_blocks=0,bool rewrite_dummy=true){
 __shared__ int arb[LR_BLOCK];
 __shared__ __align__(16) unsigned short cm[LR_CHUNKS*LR_E];
 __shared__ int counts[LR_E];
 __shared__ int starts[LR_E];
 __shared__ int tot[LR_E];
 __shared__ int slots[LR_SLOTS];
 __shared__ uint32_t cumw[LR_SLOTS];
 int t=threadIdx.x,b=blockIdx.x,f=b*LR_BLOCK+t;
 int first=(valid_rows!=nullptr)?pad_first(*valid_rows,n/PAD_K,rows_ref):n;   // first padding element (n: none)
 int real_first=first;
 if constexpr(Sample){if(!rewrite_dummy)first=n;}
 __shared__ int sample_cell;
 if constexpr(Sample){
  if(t==0){
   sample_cell=-1;
   if(sample[0]!=0){
    int step=sample_steps[sample_layer];long long slot=(long long)step-sample[2];
    if(sample[0]!=1||sample[1]<=0||sample[2]<0||*valid_rows<0||*valid_rows>rows_ref)atomicOr(sample+3,4);
    else if(b>=sample_blocks)atomicOr(sample+3,1);
    else if(slot<0||slot>=sample_capacity){} // Explicit bounded subwindow: outside it, route normally without recording.
    else{
     int cell=4+((slot*(GENEPLB_LAYERS-1)+sample_layer)*sample_blocks+b)*521;
     if(atomicCAS(sample+cell+1,-1,step)!=-1)atomicOr(sample+3,2);
     else{
      sample[cell]=sample[1];sample[cell+2]=n/PAD_K;sample[cell+3]=rows_ref;
      sample[cell+4]=real_first;sample[cell+5]=*wmode_g;sample[cell+6]=b;
      sample[cell+7]=ep_rank;sample[cell+8]=rewrite_dummy?1:0;sample_cell=cell;
     }
    }
   }
  }
 }
 int raw;                                                     // the staged id (the id the route used)
 int e=pad_load(ids,n,f,first,ep_rank,&raw);                  // LOAD (as the lean kernel; a padding element: its pad expert) ...
 int cval=0;
 if(t<LR_E){cval=counts_g[t];counts[t]=cval;starts[t]=starts_g[t];}
 if(t<LR_SLOTS){slots[t]=slots_g[t];cumw[t]=(uint32_t)cumw_g[t];}   // ... plus the layer's cumw row (uint32 bits in int32 storage)
 int ve=*verify_enable;
 int wmode=*wmode_g;                                          // block-uniform: one word per layer
 int replicas=__syncthreads_or(t<LR_E&&cval>1);
 if constexpr(Sample){
  // Existing LOAD barrier publishes the cell reservation. No extra launch/barrier.
  if(sample_cell>=0){
   reinterpret_cast<uint16_t*>(sample+sample_cell+9)[t]=(uint16_t)e;
   if(f<n&&e>=LR_E)atomicOr(sample+3,8);
  }
 }
 if(!replicas){                                               // FAST PATH: identical under both rules
  int slot=lr_slot(e,0,b,ep_rank,counts,starts,slots);
  if(f<n){
   out[f]=slot;
   if(ve!=0){stage_ids[f]=raw;stage_slots[f]=slot;}
  }
  if(t==0&&b==0)atomicAdd(path_counters,1);
  return;
 }
 if(t==0&&b==0)atomicAdd(path_counters+(wmode!=0?2:1),1);
 for(int i=t;i<LR_CM_WORDS;i+=LR_BLOCK)reinterpret_cast<unsigned*>(cm)[i]=0u;
 arb[t]=lr_arbitrate_id(e,counts);
 __syncthreads();
 int c=t/LR_CHUNK,lane=t%LR_CHUNK,r,cnt;                      // RANK
 lr_chunk_rank(arb+c*LR_CHUNK,lane,&r,&cnt);
 int a=arb[t];
 if(r==0&&a<LR_E)cm[c*LR_E+a]=(unsigned short)cnt;
 __syncthreads();
 if(t<LR_E)tot[t]=wroute::wr_prefix_column(cm,t);            // PREFIX (+ the block total of expert t)
 __syncthreads();
 int ordinal=(a<LR_E)?((int)cm[c*LR_E+a]+r):0;                // RESOLVE
 int slot=l3_slot(wmode,e,a,ordinal,(a<LR_E)?tot[a]:0,b,ep_rank,counts,starts,slots,cumw);
 if(f<n){
  out[f]=slot;
  if(ve!=0){stage_ids[f]=raw;stage_slots[f]=slot;}
 }
}

// As the lean kernel's compare_k.
__global__ void compare_k(const int* __restrict__ expected,const int* __restrict__ got,int n,int ordinal,int* __restrict__ words){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i==0){atomicAdd(words+5,n);atomicAdd(words+6,1);}
 if(i>=n)return;
 int a=expected[i],b=got[i];
 if(a!=b){
  int old=atomicAdd(words,1);
  if(old==0){words[1]=ordinal;words[2]=i;words[3]=a;words[4]=b;}
 }
}
} // namespace

// Route `ids` (int32 [rows, 6]) with the three tables at the given device addresses, the layer's cumw row (int32 [512]) and its
// wmode word (int32 [1]).  Current stream, capturable: every pointer is fixed at capture time.
// one launcher for both entries; valid_rows == nullptr (entry `route`) = no padding word = no padding rule.
static torch::Tensor launch_route(torch::Tensor ids,int64_t ep_rank,int64_t counts_addr,int64_t starts_addr,int64_t slots_addr,
  torch::Tensor cumw,torch::Tensor wmode,torch::Tensor verify_enable,torch::Tensor stage_ids,torch::Tensor stage_slots,torch::Tensor path_counters,
  int64_t rows_ref,const int* valid_rows,int* sample=nullptr,const int* sample_steps=nullptr,
  int sample_layer=0,int sample_capacity=0,int sample_blocks=0,bool rewrite_dummy=true){
 ck(ids,torch::kInt32);ck(cumw,torch::kInt32);ck(wmode,torch::kInt32);ck(verify_enable,torch::kInt32);ck(stage_ids,torch::kInt32);
 ck(stage_slots,torch::kInt32);ck(path_counters,torch::kInt32);
 TORCH_CHECK(path_counters.numel()==PATH_WORDS,"path_counters are three int32 words {fast, full native, full weighted}");
 TORCH_CHECK(ids.dim()==2&&ids.size(1)==K&&ids.size(0)>=0&&ids.size(0)<=MAX_ROWS,"ids must be int32 [rows<=65536, 6]");
 TORCH_CHECK(ep_rank>=0&&ep_rank<GENEPLB_RANKS,"ep_rank must be in 0..ranks-1");
 for(int64_t p:{counts_addr,starts_addr,slots_addr})TORCH_CHECK(p>0&&(p%4)==0,"table address must be a non-null int32 device address");
 TORCH_CHECK(cumw.numel()==LR_SLOTS,"cumw is the layer's 512-entry Q32 cumulative weight row");
 TORCH_CHECK(wmode.numel()==1&&verify_enable.numel()==1,"wmode / verify_enable are single int32 words");
 int64_t n=ids.size(0)*K;
 TORCH_CHECK(stage_ids.dim()==1&&stage_slots.dim()==1&&stage_ids.numel()>=n&&stage_slots.numel()>=n,"staging rows must cover this launch's elements");
 auto out=torch::empty_like(ids);
 if(n==0)return out;
 int blocks=(int)((n+LR_BLOCK-1)/LR_BLOCK);
 if(sample){
  l3_pad_route_k<true><<<blocks,LR_BLOCK,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),(int)n,(int)ep_rank,
   reinterpret_cast<const int*>(counts_addr),reinterpret_cast<const int*>(starts_addr),reinterpret_cast<const int*>(slots_addr),
   cumw.data_ptr<int>(),wmode.data_ptr<int>(),out.data_ptr<int>(),verify_enable.data_ptr<int>(),stage_ids.data_ptr<int>(),
   stage_slots.data_ptr<int>(),path_counters.data_ptr<int>(),(int)rows_ref,valid_rows,sample,sample_steps,
   sample_layer,sample_capacity,sample_blocks,rewrite_dummy);
 }else{
 l3_pad_route_k<<<blocks,LR_BLOCK,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),(int)n,(int)ep_rank,
  reinterpret_cast<const int*>(counts_addr),reinterpret_cast<const int*>(starts_addr),reinterpret_cast<const int*>(slots_addr),
  cumw.data_ptr<int>(),wmode.data_ptr<int>(),out.data_ptr<int>(),verify_enable.data_ptr<int>(),stage_ids.data_ptr<int>(),
  stage_slots.data_ptr<int>(),path_counters.data_ptr<int>(),(int)rows_ref,valid_rows);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 return out;
}

torch::Tensor route_cuda(torch::Tensor ids,int64_t ep_rank,int64_t counts_addr,int64_t starts_addr,int64_t slots_addr,
  torch::Tensor cumw,torch::Tensor wmode,torch::Tensor verify_enable,torch::Tensor stage_ids,torch::Tensor stage_slots,torch::Tensor path_counters){
 return launch_route(ids,ep_rank,counts_addr,starts_addr,slots_addr,cumw,wmode,verify_enable,stage_ids,stage_slots,path_counters,
                     ids.dim()==2?ids.size(0):0,nullptr);
}

// as `route`, plus the main-pass row count of the captured forward (rows_ref >= this launch's rows; the MTP draft
// re-invocations have fewer) and the padding word (int32 [1], the stats valid_rows word: real rows of the main pass).
torch::Tensor route_pad_cuda(torch::Tensor ids,int64_t ep_rank,int64_t counts_addr,int64_t starts_addr,int64_t slots_addr,
  torch::Tensor cumw,torch::Tensor wmode,torch::Tensor verify_enable,torch::Tensor stage_ids,torch::Tensor stage_slots,torch::Tensor path_counters,
  int64_t rows_ref,torch::Tensor valid_rows){
 ck(valid_rows,torch::kInt32);
 TORCH_CHECK(valid_rows.numel()==1,"valid_rows is a single int32 word");
 TORCH_CHECK(ids.dim()==2&&rows_ref>0&&rows_ref<=MAX_ROWS&&ids.size(0)<=rows_ref,"rows_ref must be in (0, 65536] and cover this launch's rows");
 return launch_route(ids,ep_rank,counts_addr,starts_addr,slots_addr,cumw,wmode,verify_enable,stage_ids,stage_slots,path_counters,
                     rows_ref,valid_rows.data_ptr<int>());
}

// Private diagnostic primitive: no product settings/capture caller enables it yet.
torch::Tensor route_packed_cuda(torch::Tensor ids,int64_t ep_rank,int64_t counts_addr,int64_t starts_addr,int64_t slots_addr,
 torch::Tensor cumw,torch::Tensor wmode,torch::Tensor verify_enable,torch::Tensor stage_ids,torch::Tensor stage_slots,
 torch::Tensor path_counters,int64_t rows_ref,torch::Tensor valid_rows,torch::Tensor steps,torch::Tensor sample,
 int64_t layer,int64_t capacity,int64_t max_blocks,bool rewrite_dummy){
 ck(valid_rows,torch::kInt32);ck(steps,torch::kInt32);ck(sample,torch::kInt32);
 TORCH_CHECK(valid_rows.numel()==1&&steps.numel()==GENEPLB_LAYERS,"sampling uses original valid-row word and62 stats steps");
 TORCH_CHECK(layer>=0&&layer<GENEPLB_LAYERS-1,"main_first4N layers0..60 only; MTP is not qualified");
 TORCH_CHECK(capacity>=1&&capacity<=4096&&max_blocks>=1&&max_blocks<=384,"bounded sample/block capacities");
 TORCH_CHECK(sample.dim()==1&&sample.numel()==4+capacity*(GENEPLB_LAYERS-1)*max_blocks*521&&sample.numel()<=0x7fffffff,
  "exact bounded packed-ID slab ABI required");
 TORCH_CHECK(ids.dim()==2&&ids.size(0)>0&&rows_ref==ids.size(0),"main source rows must equal reference; zero rows unsupported");
 TORCH_CHECK(ids.size(0)*K<=max_blocks*1024,"captured source exceeds declared block capacity");
 auto first=reinterpret_cast<uintptr_t>(sample.data_ptr());
 auto last=first+sample.numel()*sample.element_size();
 auto disjoint=[&](uintptr_t begin,uintptr_t bytes){TORCH_CHECK(last<=begin||begin+bytes<=first,"sample overlaps route/state memory");};
 for(auto const&t:{ids,cumw,wmode,verify_enable,stage_ids,stage_slots,path_counters,valid_rows,steps})
  disjoint(reinterpret_cast<uintptr_t>(t.data_ptr()),t.numel()*t.element_size());
 disjoint(static_cast<uintptr_t>(counts_addr),LR_E*sizeof(int));
 disjoint(static_cast<uintptr_t>(starts_addr),LR_E*sizeof(int));
 disjoint(static_cast<uintptr_t>(slots_addr),LR_SLOTS*sizeof(int));
 return launch_route(ids,ep_rank,counts_addr,starts_addr,slots_addr,cumw,wmode,verify_enable,stage_ids,stage_slots,path_counters,
  rows_ref,valid_rows.data_ptr<int>(),sample.data_ptr<int>(),steps.data_ptr<int>(),(int)layer,(int)capacity,(int)max_blocks,rewrite_dummy);
}

void compare_cuda(torch::Tensor expected,torch::Tensor got,int64_t ordinal,torch::Tensor words){
 ck(expected,torch::kInt32);ck(got,torch::kInt32);ck(words,torch::kInt32);
 TORCH_CHECK(expected.numel()==got.numel()&&expected.numel()<=MAX_ROWS*K,"compare operands must have equal length");
 TORCH_CHECK(words.numel()==MISMATCH_WORDS,"mismatch words");
 int n=(int)expected.numel();
 if(n==0)return;
 compare_k<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(expected.data_ptr<int>(),got.data_ptr<int>(),n,(int)ordinal,words.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("route",&route_cuda);m.def("compare",&compare_cuda);
 m.def("route_packed",&route_packed_cuda);
 m.def("route_pad",&route_pad_cuda);m.attr("PAD_K")=PAD_K;m.attr("PAD_PHASE")=PAD_PHASE;
 m.attr("EXPERTS")=LR_E;m.attr("SLOTS")=LR_SLOTS;m.attr("TOP_K")=K;m.attr("BLOCK")=LR_BLOCK;m.attr("CHUNK")=LR_CHUNK;
 m.attr("MAX_ROWS")=MAX_ROWS;m.attr("MISMATCH_WORDS")=MISMATCH_WORDS;m.attr("PATH_WORDS")=PATH_WORDS;m.attr("RANKS")=GENEPLB_RANKS;
 m.attr("WR_PHI_G")=(int64_t)wroute::WR_PHI_G;m.attr("WR_PHI_E")=(int64_t)wroute::WR_PHI_E;m.attr("WR_GROUP_STRIDE")=wroute::WR_GROUP_STRIDE;
}
