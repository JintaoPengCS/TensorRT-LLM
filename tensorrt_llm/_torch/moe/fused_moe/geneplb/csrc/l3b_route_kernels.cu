// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
// GenEPLB's route kernels of cost model B: the logical -> physical remap of C's kernel (l3_pad_route_kernels.cu, l3_pad_route_k)
// without what B does not read (no verification staging, no path counters), int32 or int64 output.
//   [l3-host-begin/end]  l3_pad_route_kernels.cu's region byte for byte (test_geneplb_numerics.py checks it): the native and
//                        weighted rules, the pad rule and their host emulations;
//   [l3b-host-begin/end] the mb kernel's arithmetic (lb_step32 == wr_step, lb_spread_limit == spread_limit for every input), its RESOLVE
//                        (lb_slot), the stage store (lb_stage_store) and the host emulations l3b_mb_emulate_rule_block / l3b_stage_emulate,
//                        equal to l3pad_emulate_block bit for bit (test_geneplb_numerics.py compares them for several geometries);
//   l3b_mb_k<OutT>       the same slot for every element under the same tables, cumw, wmode and
//                        padding word, from many small blocks: a launch's elements in tiles of MB_TPB (256), one block each; a block
//                        recounts its whole rule block (LR_BLOCK elements, MB_TILES per thread, chunk-aligned) into shared histograms --
//                        the rule block's totals and the counts before its tile -- plus the per-chunk counts of its own tile, then
//                        resolves its own elements (ordinal = before + earlier own chunks + rank in the chunk). Every global word a thread
//                        reads is loaded before the first use; no verification staging and no path counters (nothing reads them); no
//                        communication between blocks, no scratch, two block barriers; int64 output and the padding rule fused;
//   route_b(...)         its entry into the caller's buffer `out` (int32 | int64, as many elements as ids): the caller allocates it on the
//                        main stream before the overlap's fork; pad selects the padding word (route_pad's semantics) or none (route's);
//   l3b_mbs_k<OutT>      l3b_mb_k's text with its two stores replaced by lb_stage_store and one early exit: the same slots, written
//                        straight into MegaMoE's staging of the selected
//                        bucket -- topk_idx_local[0 : bucket] (the slots, -1 for the rows past the routed ones) and topk_weights[0 :
//                        bucket, 0 : 6] (the gate's fp32 weights, 0 past them; row stride ld, the columns past 6 untouched) -- what
//                        mega_moe_cute_dsl._stage_inputs' index fill, weight copy, weight-tail zero and index copy would leave there;
//   route_b_stage(...)   its entry: grid ceil(bucket x 6 / MB_TPB); a tile past the routed elements only writes the tail.
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
// 512 physical slots, top-6).
#ifndef GENEPLB_E
#define GENEPLB_E 384                 // routed experts
#endif
#ifndef GENEPLB_SLOTS
#define GENEPLB_SLOTS 512             // physical slots of all ranks (ranks x slots per rank); also the invalid-expert sentinel slot id
#endif
#ifndef GENEPLB_TOPK
#define GENEPLB_TOPK 6                // top-k: a row is GENEPLB_TOPK consecutive flat elements
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

// [l3b-host-begin]
namespace {
// [l3b-logic-begin]
// lb_step32(n) == wroute::wr_step(n) for every int n, by 32-bit divisions only: with 2^32 = q n + r (0 <= r < n),
// floor((2^32 - 1) / n) is q - 1 exactly when r == 0, and then (2^32 - 1) mod n == n - 1; otherwise it is q and the remainder r - 1.
__host__ __device__ __forceinline__ uint32_t lb_step32(int n){
 if(n<=1)return 0xFFFFFFFFu;
 uint32_t un=(uint32_t)n,q=0xFFFFFFFFu/un;
 return q+((0xFFFFFFFFu-q*un)==un-1u?1u:0u);
}
// lb_spread_limit == spread_limit for every input: the same branches; the quotient by a 32-bit division whenever the dividend fits (at
// decode always: rows_ref <= MAX_ROWS), else by the 64-bit one.  Every operand is non-negative past the first test.
__host__ __device__ __forceinline__ int lb_spread_limit(int sr,int rows,int rows_ref){
 if(sr<0||rows_ref<=0||sr>=rows_ref||rows>rows_ref||rows<=0)return rows;
 unsigned long long num=(unsigned long long)sr*(unsigned long long)rows+(unsigned long long)(rows_ref-1);
 unsigned long long limit=(num<=0xFFFFFFFFull)?(unsigned long long)((uint32_t)num/(uint32_t)rows_ref):num/(unsigned long long)rows_ref;
 return limit>=(unsigned long long)rows?rows:(int)limit;
}
__host__ __device__ __forceinline__ int lb_pad_first(int vr,int rows,int rows_ref){ return lb_spread_limit(vr,rows,rows_ref)*PAD_K; }
// route_mb (l3b_mb_k): tiles of MB_TPB elements (one block each, MB_TILES per rule block); thread t of a block handles element
// t + MB_TPB i of its rule block for i < MB_TILES -- warp w holds chunk w + MB_WARPS i, a whole chunk -- and owns item q (its tile).
constexpr int MB_TPB=256;
constexpr int MB_TILES=LR_BLOCK/MB_TPB;
constexpr int MB_WARPS=MB_TPB/32;
static_assert(MB_TPB%32==0&&LR_BLOCK%MB_TPB==0&&MB_TILES*MB_TPB==LR_BLOCK,"tiles are whole chunks of one rule block");
// RESOLVE of l3b_mb_k / l3b_mbs_k: l3_slot with the step by lb_step32 (the same value)
__host__ __device__ __forceinline__ int lb_slot(int wmode,int e,int a,int ordinal,int total,int block,int ep_rank,const int* counts,
  const int* starts,const int* slots,const uint32_t* cumw){
 if(wmode==0)return lr_slot(e,ordinal,block,ep_rank,counts,starts,slots);
 return wroute::wr_slot(e,ordinal,(a<LR_E)?lb_step32(total):0u,block,ep_rank,counts,starts,slots,cumw);
}
// stage (l3b_mbs_k): flat element f of a bucket of `total` = bucket x PAD_K elements, n of them routed -> the slot (f < n) or -1 (the
// tail) into idx_out[f], and the gate's weight w_in[f] (f < n) or 0 (the tail) into row f / PAD_K, column f % PAD_K of w_out (row stride
// ld >= PAD_K; the columns past PAD_K are never written).  The kernel's store and the host emulation's (l3b_stage_emulate).
template<typename OutT>
__host__ __device__ __forceinline__ void lb_stage_store(int f,int n,int total,int ld,int slot,const float* w_in,OutT* idx_out,float* w_out){
 if(f<n){idx_out[f]=(OutT)slot;w_out[(f/PAD_K)*ld+f%PAD_K]=w_in[f];}
 else if(f<total){idx_out[f]=(OutT)(-1);w_out[(f/PAD_K)*ld+f%PAD_K]=0.0f;}
}
#ifndef __CUDACC__
// Host emulation of l3b_mb_k's blocks in rule block b: every tile the launch has there (base + q MB_TPB < n), each from its own recount.
// Buffers: counts/starts[LR_E], slots/cumw[LR_SLOTS], e_buf/a_buf/r_buf[LR_BLOCK] (the rule block's elements), tot/bef[LR_E],
// cnt[MB_WARPS*LR_E].  Returns 1 for the fast path.
inline int l3b_mb_emulate_rule_block(const int* ids,int n,int first,int b,int ep_rank,const int* counts_g,const int* starts_g,
 const int* slots_g,const uint32_t* cumw_g,int wmode,int* out,int* counts,int* starts,int* slots,uint32_t* cumw,int* e_buf,int* a_buf,
 int* r_buf,int* tot,int* bef,unsigned char* cnt){
 const int base=b*LR_BLOCK;
 int replicas=0;
 for(int i=0;i<LR_E;++i){counts[i]=counts_g[i];starts[i]=starts_g[i];replicas|=(counts[i]>1);}
 for(int i=0;i<LR_SLOTS;++i){slots[i]=slots_g[i];cumw[i]=cumw_g[i];}
 for(int x=0;x<LR_BLOCK;++x){int s=0;e_buf[x]=pad_load(ids,n,base+x,first,ep_rank,&s);}   // LOAD (every block: its rule block)
 for(int q=0;q<MB_TILES&&base+q*MB_TPB<n;++q){                                            // the launch's blocks in this rule block
  if(!replicas){                                                                         // FAST PATH: as C's kernel
   for(int t=0;t<MB_TPB;++t){int f=base+q*MB_TPB+t;if(f<n)out[f]=lr_slot(e_buf[q*MB_TPB+t],0,b,ep_rank,counts,starts,slots);}
   continue;
  }
  for(int i=0;i<LR_E;++i){tot[i]=0;bef[i]=0;}
  for(int i=0;i<MB_WARPS*LR_E;++i)cnt[i]=0;
  for(int it=0;it<MB_TILES;++it){                                                         // COUNT: item it of every thread
   for(int w=0;w<MB_WARPS;++w){
    const int c0=(it*MB_WARPS+w)*LR_CHUNK;                                                // warp w's chunk w + MB_WARPS it
    for(int l=0;l<LR_CHUNK;++l)a_buf[c0+l]=lr_arbitrate_id(e_buf[c0+l],counts);
    for(int l=0;l<LR_CHUNK;++l){
     int r,k;
     lr_chunk_rank(a_buf+c0,l,&r,&k);                                                     // the kernel: __match_any_sync
     r_buf[c0+l]=r;
     const int a=a_buf[c0+l];
     if(r==0&&a<LR_E){tot[a]+=k;if(it<q)bef[a]+=k;else if(it==q)cnt[w*LR_E+a]=(unsigned char)k;}
    }
   }
  }
  for(int t=0;t<MB_TPB;++t){                                                              // RESOLVE: thread t's own element (item q)
   const int x=q*MB_TPB+t,f=base+x,w=t/LR_CHUNK;
   if(f>=n)continue;
   int a=a_buf[x],ordinal=0,total=0;
   if(a<LR_E){ordinal=bef[a]+r_buf[x];for(int v=0;v<w;++v)ordinal+=cnt[v*LR_E+a];total=tot[a];}
   out[f]=lb_slot(wmode,e_buf[x],a,ordinal,total,b,ep_rank,counts,starts,slots,cumw);
  }
 }
 return replicas?0:1;
}
// Host emulation of l3b_mbs_k (route_b_stage): l3b_mb_emulate_rule_block's slots for the launch's n elements (slots_out[n]), then every
// element of the bucket (total = bucket x PAD_K) through lb_stage_store, the kernel's own store.  Buffers as l3b_mb_emulate_rule_block's.
template<typename OutT>
inline void l3b_stage_emulate(const int* ids,int n,int first,int ep_rank,const int* counts_g,const int* starts_g,const int* slots_g,
 const uint32_t* cumw_g,int wmode,const float* w_in,int total,int ld,OutT* idx_out,float* w_out,int* slots_out,int* counts,int* starts,
 int* slots,uint32_t* cumw,int* e_buf,int* a_buf,int* r_buf,int* tot,int* bef,unsigned char* cnt){
 const int blocks=(n+LR_BLOCK-1)/LR_BLOCK;
 for(int b=0;b<blocks;++b)
  l3b_mb_emulate_rule_block(ids,n,first,b,ep_rank,counts_g,starts_g,slots_g,cumw_g,wmode,slots_out,counts,starts,slots,cumw,e_buf,a_buf,
                            r_buf,tot,bef,cnt);
 for(int f=0;f<total;++f)lb_stage_store<OutT>(f,n,total,ld,(f<n)?slots_out[f]:0,w_in,idx_out,w_out);
}
#endif
// [l3b-logic-end]
}  // namespace
// [l3b-host-end]

namespace {
constexpr int K=GENEPLB_TOPK;            // top-k of DeepSeek-V4-Pro
constexpr int MAX_ROWS=1<<16;            // host cap on rows per launch
static_assert(LR_CHUNKS*LR_CHUNK==LR_BLOCK&&LR_CHUNK==32&&LR_BLOCK==1024,"a chunk is one warp, a block 32 warps");
static_assert(2*MB_TPB>=LR_SLOTS&&(MB_WARPS*LR_E)%4==0,"l3b_mb_k: two table words per thread cover the slots, its counts zero as words");
static_assert(LR_E<=LR_SLOTS,"(port) generic geometry: the same two table words per thread cover the counts / starts");

void ck(const torch::Tensor&t,torch::ScalarType d){
 TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==d&&t.get_device()==c10::cuda::current_device(),
  "current-device contiguous tensor required");}

// route_b: ceil(n / MB_TPB) blocks of MB_TPB threads; block = tile q of rule block b.
template<typename OutT>
__global__ void __launch_bounds__(MB_TPB) l3b_mb_k(const int* __restrict__ ids,int n,int ep_rank,
  const int* __restrict__ counts_g,const int* __restrict__ starts_g,const int* __restrict__ slots_g,
  const int* __restrict__ cumw_g,const int* __restrict__ wmode_g,OutT* __restrict__ out,int rows_ref,const int* __restrict__ valid_rows){
 __shared__ int counts[LR_E];
 __shared__ int starts[LR_E];
 __shared__ int slots[LR_SLOTS];
 __shared__ uint32_t cumw[LR_SLOTS];
 __shared__ int tot[LR_E];                                    // the rule block's elements per arbitration id
 __shared__ int bef[LR_E];                                    // ... in the chunks before this tile
 __shared__ __align__(16) unsigned char cnt[MB_WARPS*LR_E];   // ... in each chunk of this tile (<= 32)
 const int t=threadIdx.x,lane=t&31,w=t>>5;
 const int b=blockIdx.x/MB_TILES,q=blockIdx.x%MB_TILES,base=b*LR_BLOCK;
 int vr=(valid_rows!=nullptr)?*valid_rows:0;                  // LOAD: every global word, issued before the first use
 int raw[MB_TILES];
#pragma unroll
 for(int i=0;i<MB_TILES;++i){int f=base+t+MB_TPB*i;raw[i]=(f<n)?ids[f]:LR_E;}
 int cv[2],sv[2],slv[2],cwv[2];
#pragma unroll
 for(int k=0;k<2;++k){
  int x=t+MB_TPB*k;
  cv[k]=(x<LR_E)?counts_g[x]:0;sv[k]=(x<LR_E)?starts_g[x]:0;
  slv[k]=(x<LR_SLOTS)?slots_g[x]:0;cwv[k]=(x<LR_SLOTS)?cumw_g[x]:0;
 }
 int wmode=*wmode_g;                                          // block-uniform: one word per layer
 int first=(valid_rows!=nullptr)?lb_pad_first(vr,n/PAD_K,rows_ref):n;
 int e[MB_TILES];
#pragma unroll
 for(int i=0;i<MB_TILES;++i){int f=base+t+MB_TPB*i;e[i]=(f<n&&f>=first)?pad_expert(f,first,ep_rank):lr_sanitize(raw[i]);}
 int rep=0;
#pragma unroll
 for(int k=0;k<2;++k){
  int x=t+MB_TPB*k;
  if(x<LR_E){counts[x]=cv[k];starts[x]=sv[k];tot[x]=0;bef[x]=0;rep|=(cv[k]>1)?1:0;}
  if(x<LR_SLOTS){slots[x]=slv[k];cumw[x]=(uint32_t)cwv[k];}
 }
 for(int i=t;i<MB_WARPS*LR_E/4;i+=MB_TPB)reinterpret_cast<unsigned*>(cnt)[i]=0u;
 int e_own=e[0];
#pragma unroll
 for(int i=1;i<MB_TILES;++i)if(q==i)e_own=e[i];
 const int f_own=base+q*MB_TPB+t;
 int replicas=__syncthreads_or(rep);                          // barrier A: the tables, the zeroed counts
 if(!replicas){                                               // FAST PATH: as C's kernel
  int slot=lr_slot(e_own,0,b,ep_rank,counts,starts,slots);
  if(f_own<n)out[f_own]=(OutT)slot;
  return;
 }
 int a_own=LR_E,r_own=0;
#pragma unroll
 for(int i=0;i<MB_TILES;++i){                                 // COUNT: this warp's chunk w + MB_WARPS i of the rule block
  int a=lr_arbitrate_id(e[i],counts);
  unsigned peers=__match_any_sync(0xffffffffu,a);
  int r=__popc(peers&((1u<<lane)-1u));
  if(r==0&&a<LR_E){
   int k=__popc(peers);
   atomicAdd(&tot[a],k);
   if(i<q)atomicAdd(&bef[a],k);
   else if(i==q)cnt[w*LR_E+a]=(unsigned char)k;
  }
  if(i==q){a_own=a;r_own=r;}
 }
 __syncthreads();                                             // barrier B: the counts
 int ordinal=0,total=0;                                       // RESOLVE
 if(a_own<LR_E){
  ordinal=bef[a_own]+r_own;
  for(int v=0;v<w;++v)ordinal+=cnt[v*LR_E+a_own];
  total=tot[a_own];
 }
 int slot=lb_slot(wmode,e_own,a_own,ordinal,total,b,ep_rank,counts,starts,slots,cumw);
 if(f_own<n)out[f_own]=(OutT)slot;
}

// route_b_stage: l3b_mb_k into MegaMoE's staging of the selected bucket; ceil(total / MB_TPB) blocks.
template<typename OutT>
__global__ void __launch_bounds__(MB_TPB) l3b_mbs_k(const int* __restrict__ ids,int n,int ep_rank,
  const int* __restrict__ counts_g,const int* __restrict__ starts_g,const int* __restrict__ slots_g,
  const int* __restrict__ cumw_g,const int* __restrict__ wmode_g,OutT* __restrict__ out,int rows_ref,const int* __restrict__ valid_rows,int bucket_n,int ld,
  const float* __restrict__ w_in,float* __restrict__ w_out){
 __shared__ int counts[LR_E];
 __shared__ int starts[LR_E];
 __shared__ int slots[LR_SLOTS];
 __shared__ uint32_t cumw[LR_SLOTS];
 __shared__ int tot[LR_E];                                    // the rule block's elements per arbitration id
 __shared__ int bef[LR_E];                                    // ... in the chunks before this tile
 __shared__ __align__(16) unsigned char cnt[MB_WARPS*LR_E];   // ... in each chunk of this tile (<= 32)
 const int t=threadIdx.x,lane=t&31,w=t>>5;
 const int b=blockIdx.x/MB_TILES,q=blockIdx.x%MB_TILES,base=b*LR_BLOCK;
 if(base+q*MB_TPB>=n){                                        // STAGE: a tile past the routed elements: the tail only
  lb_stage_store<OutT>(base+q*MB_TPB+t,n,bucket_n,ld,0,w_in,out,w_out);
  return;
 }
 int vr=(valid_rows!=nullptr)?*valid_rows:0;                  // LOAD: every global word, issued before the first use
 int raw[MB_TILES];
#pragma unroll
 for(int i=0;i<MB_TILES;++i){int f=base+t+MB_TPB*i;raw[i]=(f<n)?ids[f]:LR_E;}
 int cv[2],sv[2],slv[2],cwv[2];
#pragma unroll
 for(int k=0;k<2;++k){
  int x=t+MB_TPB*k;
  cv[k]=(x<LR_E)?counts_g[x]:0;sv[k]=(x<LR_E)?starts_g[x]:0;
  slv[k]=(x<LR_SLOTS)?slots_g[x]:0;cwv[k]=(x<LR_SLOTS)?cumw_g[x]:0;
 }
 int wmode=*wmode_g;                                          // block-uniform: one word per layer
 int first=(valid_rows!=nullptr)?lb_pad_first(vr,n/PAD_K,rows_ref):n;
 int e[MB_TILES];
#pragma unroll
 for(int i=0;i<MB_TILES;++i){int f=base+t+MB_TPB*i;e[i]=(f<n&&f>=first)?pad_expert(f,first,ep_rank):lr_sanitize(raw[i]);}
 int rep=0;
#pragma unroll
 for(int k=0;k<2;++k){
  int x=t+MB_TPB*k;
  if(x<LR_E){counts[x]=cv[k];starts[x]=sv[k];tot[x]=0;bef[x]=0;rep|=(cv[k]>1)?1:0;}
  if(x<LR_SLOTS){slots[x]=slv[k];cumw[x]=(uint32_t)cwv[k];}
 }
 for(int i=t;i<MB_WARPS*LR_E/4;i+=MB_TPB)reinterpret_cast<unsigned*>(cnt)[i]=0u;
 int e_own=e[0];
#pragma unroll
 for(int i=1;i<MB_TILES;++i)if(q==i)e_own=e[i];
 const int f_own=base+q*MB_TPB+t;
 int replicas=__syncthreads_or(rep);                          // barrier A: the tables, the zeroed counts
 if(!replicas){                                               // FAST PATH: as C's kernel
  int slot=lr_slot(e_own,0,b,ep_rank,counts,starts,slots);
  lb_stage_store<OutT>(f_own,n,bucket_n,ld,slot,w_in,out,w_out);
  return;
 }
 int a_own=LR_E,r_own=0;
#pragma unroll
 for(int i=0;i<MB_TILES;++i){                                 // COUNT: this warp's chunk w + MB_WARPS i of the rule block
  int a=lr_arbitrate_id(e[i],counts);
  unsigned peers=__match_any_sync(0xffffffffu,a);
  int r=__popc(peers&((1u<<lane)-1u));
  if(r==0&&a<LR_E){
   int k=__popc(peers);
   atomicAdd(&tot[a],k);
   if(i<q)atomicAdd(&bef[a],k);
   else if(i==q)cnt[w*LR_E+a]=(unsigned char)k;
  }
  if(i==q){a_own=a;r_own=r;}
 }
 __syncthreads();                                             // barrier B: the counts
 int ordinal=0,total=0;                                       // RESOLVE
 if(a_own<LR_E){
  ordinal=bef[a_own]+r_own;
  for(int v=0;v<w;++v)ordinal+=cnt[v*LR_E+a_own];
  total=tot[a_own];
 }
 int slot=lb_slot(wmode,e_own,a_own,ordinal,total,b,ep_rank,counts,starts,slots,cumw);
 lb_stage_store<OutT>(f_own,n,bucket_n,ld,slot,w_in,out,w_out);
}
} // namespace

// One route launch of l3b_mb_k into the caller's buffer `out` (int32 | int64, as many elements as ids, on the current device), current
// stream, capturable (every pointer is fixed at capture time).  pad: the padding word valid_rows applies with rows_ref of the captured
// forward (entry route_pad's semantics), else none (entry route's: rows_ref unused).
template<typename OutT>
static void launch_b(torch::Tensor ids,int ep_rank,const int* cg,const int* sg,const int* slg,torch::Tensor cumw,torch::Tensor wmode,
  int rows_ref,const int* vr,OutT* out,int n){
 auto stream=at::cuda::getCurrentCUDAStream();
 l3b_mb_k<OutT><<<(n+MB_TPB-1)/MB_TPB,MB_TPB,0,stream>>>(ids.data_ptr<int>(),n,ep_rank,cg,sg,slg,cumw.data_ptr<int>(),
  wmode.data_ptr<int>(),out,rows_ref,vr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void route_b_cuda(torch::Tensor ids,int64_t ep_rank,int64_t counts_addr,int64_t starts_addr,int64_t slots_addr,torch::Tensor cumw,
  torch::Tensor wmode,int64_t rows_ref,torch::Tensor valid_rows,bool pad,torch::Tensor out){
 ck(ids,torch::kInt32);ck(cumw,torch::kInt32);ck(wmode,torch::kInt32);ck(valid_rows,torch::kInt32);
 TORCH_CHECK(out.is_cuda()&&out.is_contiguous()&&out.get_device()==c10::cuda::current_device()&&
             (out.scalar_type()==torch::kInt32||out.scalar_type()==torch::kInt64),"out: a current-device contiguous int32 / int64 tensor");
 TORCH_CHECK(ids.dim()==2&&ids.size(1)==K&&ids.size(0)>=0&&ids.size(0)<=MAX_ROWS,"ids must be int32 [rows<=65536, 6]");
 TORCH_CHECK(ep_rank>=0&&ep_rank<GENEPLB_RANKS,"ep_rank must be in 0..ranks-1");
 for(int64_t p:{counts_addr,starts_addr,slots_addr})TORCH_CHECK(p>0&&(p%4)==0,"table address must be a non-null int32 device address");
 TORCH_CHECK(cumw.numel()==LR_SLOTS,"cumw is the layer's 512-entry Q32 cumulative weight row");
 TORCH_CHECK(wmode.numel()==1&&valid_rows.numel()==1,"wmode / valid_rows are single int32 words");
 int64_t n=ids.size(0)*K;
 TORCH_CHECK(out.numel()==n,"out must have as many elements as ids");
 if(pad)TORCH_CHECK(rows_ref>0&&rows_ref<=MAX_ROWS&&ids.size(0)<=rows_ref,"rows_ref must be in (0, 65536] and cover this launch's rows");
 if(n==0)return;
 const int* cg=reinterpret_cast<const int*>(counts_addr);
 const int* sg=reinterpret_cast<const int*>(starts_addr);
 const int* slg=reinterpret_cast<const int*>(slots_addr);
 const int* vr=pad?valid_rows.data_ptr<int>():nullptr;
 int rr=pad?(int)rows_ref:(int)ids.size(0);
 if(out.scalar_type()==torch::kInt64)
  launch_b<int64_t>(ids,(int)ep_rank,cg,sg,slg,cumw,wmode,rr,vr,out.data_ptr<int64_t>(),(int)n);
 else
  launch_b<int>(ids,(int)ep_rank,cg,sg,slg,cumw,wmode,rr,vr,out.data_ptr<int>(),(int)n);
}

// The stage entry: l3b_mbs_k into the bucket's staging, current stream, capturable.  idx_out: MegaMoE's
// topk_idx_local of the bucket (contiguous [bucket, 6], int64 | int32); w_in: the gate's fp32 weights [rows, 6]; w_out: the bucket's
// topk_weights rows (fp32 [bucket, >= 6], unit column stride, row stride ld = w_out.stride(0)).  pad / rows_ref / valid_rows as route_b.
template<typename OutT>
static void launch_stage(torch::Tensor ids,int ep_rank,const int* cg,const int* sg,const int* slg,torch::Tensor cumw,torch::Tensor wmode,
  int rows_ref,const int* vr,OutT* out,int n,int total,int ld,const float* w_in,float* w_out){
 auto stream=at::cuda::getCurrentCUDAStream();
 l3b_mbs_k<OutT><<<(total+MB_TPB-1)/MB_TPB,MB_TPB,0,stream>>>(n>0?ids.data_ptr<int>():nullptr,n,ep_rank,cg,sg,slg,cumw.data_ptr<int>(),
  wmode.data_ptr<int>(),out,rows_ref,vr,total,ld,w_in,w_out);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void route_b_stage_cuda(torch::Tensor ids,int64_t ep_rank,int64_t counts_addr,int64_t starts_addr,int64_t slots_addr,torch::Tensor cumw,
  torch::Tensor wmode,int64_t rows_ref,torch::Tensor valid_rows,bool pad,torch::Tensor idx_out,torch::Tensor w_in,torch::Tensor w_out){
 ck(ids,torch::kInt32);ck(cumw,torch::kInt32);ck(wmode,torch::kInt32);ck(valid_rows,torch::kInt32);ck(w_in,torch::kFloat);
 TORCH_CHECK(ids.dim()==2&&ids.size(1)==K&&ids.size(0)>=0&&ids.size(0)<=MAX_ROWS,"ids must be int32 [rows<=65536, 6]");
 TORCH_CHECK(idx_out.is_cuda()&&idx_out.is_contiguous()&&idx_out.get_device()==c10::cuda::current_device()&&
             (idx_out.scalar_type()==torch::kInt64||idx_out.scalar_type()==torch::kInt32)&&idx_out.dim()==2&&idx_out.size(1)==K&&
             idx_out.size(0)>=ids.size(0)&&idx_out.size(0)<=MAX_ROWS,"idx_out: the bucket's contiguous int64 / int32 [bucket >= rows, 6] slots");
 TORCH_CHECK(w_in.dim()==2&&w_in.size(0)==ids.size(0)&&w_in.size(1)==K,"w_in: the gate's contiguous fp32 [rows, 6] weights");
 TORCH_CHECK(w_out.is_cuda()&&w_out.scalar_type()==torch::kFloat&&w_out.get_device()==c10::cuda::current_device()&&w_out.dim()==2&&
             w_out.size(0)==idx_out.size(0)&&w_out.size(1)>=K&&w_out.stride(1)==1&&w_out.stride(0)>=K&&w_out.stride(0)<=4096,
             "w_out: the bucket's fp32 weight rows [bucket, >= 6], unit column stride, row stride in [6, 4096]");
 TORCH_CHECK(ep_rank>=0&&ep_rank<GENEPLB_RANKS,"ep_rank must be in 0..ranks-1");
 for(int64_t p:{counts_addr,starts_addr,slots_addr})TORCH_CHECK(p>0&&(p%4)==0,"table address must be a non-null int32 device address");
 TORCH_CHECK(cumw.numel()==LR_SLOTS,"cumw is the layer's 512-entry Q32 cumulative weight row");
 TORCH_CHECK(wmode.numel()==1&&valid_rows.numel()==1,"wmode / valid_rows are single int32 words");
 if(pad)TORCH_CHECK(rows_ref>0&&rows_ref<=MAX_ROWS&&ids.size(0)<=rows_ref,"rows_ref must be in (0, 65536] and cover this launch's rows");
 int64_t n=ids.size(0)*K,total=idx_out.size(0)*K;
 if(total==0)return;
 const int* cg=reinterpret_cast<const int*>(counts_addr);
 const int* sg=reinterpret_cast<const int*>(starts_addr);
 const int* slg=reinterpret_cast<const int*>(slots_addr);
 const int* vr=pad?valid_rows.data_ptr<int>():nullptr;
 int rr=pad?(int)rows_ref:(int)ids.size(0);
 const float* wi=n>0?w_in.data_ptr<float>():nullptr;
 if(idx_out.scalar_type()==torch::kInt64)
  launch_stage<int64_t>(ids,(int)ep_rank,cg,sg,slg,cumw,wmode,rr,vr,idx_out.data_ptr<int64_t>(),(int)n,(int)total,(int)w_out.stride(0),wi,
                        w_out.data_ptr<float>());
 else
  launch_stage<int>(ids,(int)ep_rank,cg,sg,slg,cumw,wmode,rr,vr,idx_out.data_ptr<int>(),(int)n,(int)total,(int)w_out.stride(0),wi,
                    w_out.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("route_b",&route_b_cuda);
 m.def("route_b_stage",&route_b_stage_cuda);
 m.attr("EXPERTS")=LR_E;m.attr("SLOTS")=LR_SLOTS;m.attr("TOP_K")=K;m.attr("BLOCK")=LR_BLOCK;m.attr("CHUNK")=LR_CHUNK;
 m.attr("MAX_ROWS")=MAX_ROWS;m.attr("RANKS")=GENEPLB_RANKS;
 m.attr("PAD_K")=PAD_K;m.attr("PAD_PHASE")=PAD_PHASE;m.attr("MB_TPB")=MB_TPB;m.attr("STAGE")=1;
 m.attr("WR_PHI_G")=(int64_t)wroute::WR_PHI_G;m.attr("WR_PHI_E")=(int64_t)wroute::WR_PHI_E;m.attr("WR_GROUP_STRIDE")=wroute::WR_GROUP_STRIDE;
}
