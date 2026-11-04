// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
// GenEPLB's device statistics collector for all MoE layers (62 for DeepSeek-V4-Pro).
//
// Per launch (one MoE layer, one Graph replay):
//   C[l][e]  = number of (row,k) entries of this rank's routing tensor equal to e
//   F[l][e]  = fl32( fl32(0.95f * F[l][e]) + C[l][e] )      (two roundings, no FMA)
//   steps[l] += 1
// Storage is per-layer, not per-call: 62 x 384 int32 + 62 x 384 float32.
// All buffers have fixed addresses; no allocation, no host read, no sync, so
// the launch can be recorded into a CUDA Graph and replays every step.
// `enable`, `valid_rows` and `spread_rows` are read from device memory (not kernel args) so
// the host can flip them between replays without recapture.
//
// Dummy-row spread: CUDA Graph padding appends dummy requests (input token id 1,
// cuda_graph_runner._get_padded_batch / resource_manager.add_dummy_requests) as the TRAILING rows of a padded replay;
// every dummy row routes to the same top-6 experts, a hot spot the placement does not cover. The
// spread kernel rewrites those rows of the LOGICAL routing tensor in place (per layer, before the stats record and
// before the route to physical slots) so their routing is uniform over the logical experts. It reads its own
// device word `spread_rows` (real rows of the main pass; SPREAD_NONE = no rewrite) so the stats kernel's
// `valid_rows` semantics are untouched.
//
// A torch extension, the binding included in this file (single source).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
// geneplb/native.py builds this file with the geometry as -D macros; the defaults are DeepSeek-V4-Pro's.
#ifndef GENEPLB_LAYERS
#define GENEPLB_LAYERS 62             // MoE layers with statistics
#endif
#ifndef GENEPLB_E
#define GENEPLB_E 384                 // routed experts
#endif
#ifndef GENEPLB_SLOTS
#define GENEPLB_SLOTS 512             // physical slots of all ranks (exported as SLOTS / LOCAL_SLOTS only)
#endif
#ifndef GENEPLB_TOPK
#define GENEPLB_TOPK 6                // top-k
#endif
namespace {
constexpr int L=GENEPLB_LAYERS;     // MoE layers 0..61; the MTP module MoE (layer_idx 62) is excluded
constexpr int E=GENEPLB_E;          // routed experts
constexpr int K=GENEPLB_TOPK;       // top-k of DeepSeek-V4-Pro
#ifndef GENEPLB_RANKS
#define GENEPLB_RANKS 16              // EP ranks (native.py passes -DGENEPLB_RANKS)
#endif
constexpr int R=GENEPLB_RANKS;      // EP ranks (16 or 32)
constexpr int LOCAL_SLOTS=GENEPLB_SLOTS/R; // physical slots per rank (512 slots: 32 at EP16, 16 at EP32)
constexpr int SLOTS=R*LOCAL_SLOTS;  // 512
constexpr int MAX_ROWS=1<<16;       // host cap on rows per launch
constexpr int THREADS=1024;         // one block per launch; 3072 ids -> 3 per thread
constexpr float LAMBDA=0.95f;       // native decayFactor (moeLoadBalanceCommon.h:72)
constexpr int VALID_ALL=0x7fffffff; // valid_rows sentinel: count every row
constexpr int SPREAD_NONE=VALID_ALL;  // spread_rows sentinel: rewrite nothing (full batch / before the first padded replay)
constexpr int ST_BAD_EXPERT=1, ST_PAD_SEEN=2, ST_BAD_LAYER=4, ST_BAD_ROWS=8;
constexpr int STATUS_WORDS=L+1;     // status[L] is the global word (bad layer argument)
constexpr int RING_STEPS=64;        // per-step rank load: ring of per-replay [L][R] destination-rank token sums
constexpr int SNAP_LOAD=0, SNAP_COUNTS=L*E, SNAP_STEPS=2*L*E, SNAP_STATUS=2*L*E+L, PACKET=2*L*E+L+STATUS_WORDS;

void ck(const torch::Tensor&t,torch::ScalarType d){
 TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==d&&t.get_device()==c10::cuda::current_device(),
  "current-device contiguous tensor required");}

// One block per layer. Shared histogram like native statisticKernel (moeLoadBalanceKernels.cu:186-213),
// then C and F written by the same block: no global atomics, deterministic. Shared by the per-layer
// kernel (v1: one launch per layer per replay) and the fused kernel (v3: one launch per forward, 62 blocks).
__device__ __forceinline__ void count_block(int layer,const int* __restrict__ ids,int rows,int k,
  const int* __restrict__ valid_rows,const int* __restrict__ enable,
  int* __restrict__ counts,float* __restrict__ load,int* __restrict__ steps,int* __restrict__ status){
 __shared__ int hist[E]; __shared__ int flags;
 if(*enable==0)return;                                  // block-uniform early exit (kill switch)
 if(layer<0||layer>=L){if(!threadIdx.x)atomicOr(status+L,ST_BAD_LAYER);return;}
 for(int e=threadIdx.x;e<E;e+=blockDim.x)hist[e]=0;
 if(!threadIdx.x)flags=0;
 __syncthreads();
 int vr=*valid_rows,limit=rows,f=0;
 if(vr<0){limit=0;f|=ST_BAD_ROWS;}else if(vr<rows)limit=vr;   // padded tail rows are skipped
 int n=limit*k;
 for(int i=threadIdx.x;i<n;i+=blockDim.x){
  int e=ids[i];
  if(e>=0&&e<E)atomicAdd_block(hist+e,1);
  else if(e==-1)f|=ST_PAD_SEEN;                        // v3 tolerated -1 as padding sentinel; gate_forward never emits it
  else f|=ST_BAD_EXPERT;
 }
 if(f)atomicOr_block(&flags,f);
 __syncthreads();
 int*c=counts+layer*E;float*F=load+layer*E;
 for(int e=threadIdx.x;e<E;e+=blockDim.x){
  int v=hist[e];c[e]=v;
  // Explicit round-to-nearest intrinsics: nvcc must not contract into an FMA,
  // otherwise the CPU reference (two roundings) cannot be byte-exact.
  F[e]=__fadd_rn(__fmul_rn(F[e],LAMBDA),(float)v);
 }
 if(!threadIdx.x){steps[layer]+=1;if(flags)atomicOr(status+layer,flags);}
}

__global__ void __launch_bounds__(THREADS) count_decay_k(int layer,const int* __restrict__ ids,int rows,int k,
  const int* __restrict__ valid_rows,const int* __restrict__ enable,
  int* __restrict__ counts,float* __restrict__ load,int* __restrict__ steps,int* __restrict__ status){
 count_block(layer,ids,rows,k,valid_rows,enable,counts,load,steps,status);
}

// v3 fused statistics: the 62 per-layer id sources (device pointers + row counts) travel as ONE kernel
// argument (62*8 + 62*4 = 744 B < 4 KB), fixed at capture time like every other node argument; block b
// histograms layer b. Sources are either the layers' token_selected_experts tensors (variant fused_ptr, the
// hook keeps them referenced until this launch) or the per-layer staging copies (variant fused_copy).
constexpr int STAGE_ROWS=2048;      // staging rows per layer for fused_copy (62 x 2048 x 6 int32 = 3 MB)
struct Sources { const int* ids[L]; int rows[L]; };
static_assert(sizeof(Sources)<=4096&&R>=1&&R<=THREADS,"(port) generic geometry: the fused sources are one kernel argument (<= 4 KB); rank_sums_k has one thread per rank");

__global__ void __launch_bounds__(THREADS) fused_count_decay_k(Sources s,int k,
  const int* __restrict__ valid_rows,const int* __restrict__ enable,
  int* __restrict__ counts,float* __restrict__ load,int* __restrict__ steps,int* __restrict__ status){
 int layer=blockIdx.x;
 if(layer>=L)return;
 count_block(layer,s.ids[layer],s.rows[layer],k,valid_rows,enable,counts,load,steps,status);
}

// Dummy-row spread: the pure index logic first (host + device, between the markers), then the kernel.
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

// One block per launch; rows*K <= 3072 ids at 512 rows -> 3 per thread. NOT gated by `enable`: the stats kill
// switch (stats_after_decision) must not stop the spread, which has to run on every replay.
__global__ void __launch_bounds__(THREADS) spread_dummy_rows_k(int* __restrict__ ids,int rows,int rows_ref,
  const int* __restrict__ spread_rows){
 int limit=spread_limit(*spread_rows,rows,rows_ref);       // block-uniform
 if(limit>=rows)return;
 int n=rows*K;
 for(int i=limit*K+threadIdx.x;i<n;i+=blockDim.x)ids[i]=spread_value(i,E);
}

// Per-step rank load: right after the fused stats kernel of a replay (stream order), block l sums this replay's
// per-expert counts C[l][e] (this rank's rows, logical experts) by the destination rank of each expert's single copy (rank_map, written
// by the host at the window start; -1 = unknown, not counted) into ring[(steps[l]-1) % RING_STEPS][l][16]. count_block incremented
// steps[l] for this replay before this kernel runs, and reset zeroes steps at the window start, so a window's replays land in slots
// 0..window-1. Gated by `enable` exactly like count_block: outside drift windows nothing is read or written. Summed over the 16 source
// ranks on the host, slot s gives the tokens every rank processed at step s -- the direct per-step imbalance.
__global__ void __launch_bounds__(THREADS) rank_sums_k(const int* __restrict__ counts,const int* __restrict__ rank_map,
  const int* __restrict__ steps,const int* __restrict__ enable,int* __restrict__ ring){
 if(*enable==0)return;                                  // block-uniform early exit (kill switch / outside windows)
 int layer=blockIdx.x;
 if(layer>=L)return;
 __shared__ int sums[R];
 if(threadIdx.x<R)sums[threadIdx.x]=0;
 __syncthreads();
 for(int e=threadIdx.x;e<E;e+=blockDim.x){
  int r=rank_map[layer*E+e];
  if(r>=0&&r<R)atomicAdd_block(sums+r,counts[layer*E+e]);
 }
 __syncthreads();
 int slot=(steps[layer]-1)%RING_STEPS; if(slot<0)slot+=RING_STEPS;
 if(threadIdx.x<R)ring[(slot*L+layer)*R+threadIdx.x]=sums[threadIdx.x];
}

__global__ void reset_k(int* counts,float* load,int* steps,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<L*E){counts[i]=0;load[i]=0.f;}
 if(i<L)steps[i]=0;
 if(i<STATUS_WORDS)status[i]=0;
}

// Copies F bits, C, steps, status into one int32 packet (single NCCL all_gather).
__global__ void snapshot_k(const int* __restrict__ counts,const float* __restrict__ load,
  const int* __restrict__ steps,const int* __restrict__ status,int* __restrict__ packet){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<L*E){packet[SNAP_LOAD+i]=__float_as_int(load[i]);packet[SNAP_COUNTS+i]=counts[i];}
 if(i<L)packet[SNAP_STEPS+i]=steps[i];
 if(i<STATUS_WORDS)packet[SNAP_STATUS+i]=status[i];
}
} // namespace

void record_cuda(int64_t layer,torch::Tensor ids,torch::Tensor valid_rows,torch::Tensor enable,
  torch::Tensor counts,torch::Tensor load,torch::Tensor steps,torch::Tensor status){
 for(auto&t:{ids,valid_rows,enable,counts,steps,status})ck(t,torch::kInt32);
 ck(load,torch::kFloat32);
 TORCH_CHECK(layer>=0&&layer<L,"layer must be in [0,62)");
 TORCH_CHECK(ids.dim()==2&&ids.size(1)==K&&ids.size(0)>=0&&ids.size(0)<=MAX_ROWS,"ids must be int32 [rows<=65536, 6]");
 TORCH_CHECK(valid_rows.numel()==1&&enable.numel()==1,"valid_rows/enable are single int32 words");
 TORCH_CHECK(counts.sizes()==torch::IntArrayRef({L,E})&&load.sizes()==torch::IntArrayRef({L,E})
  &&steps.numel()==L&&status.numel()==STATUS_WORDS,"stats buffer shapes");
 count_decay_k<<<1,THREADS,0,at::cuda::getCurrentCUDAStream()>>>((int)layer,ids.data_ptr<int>(),(int)ids.size(0),K,
  valid_rows.data_ptr<int>(),enable.data_ptr<int>(),counts.data_ptr<int>(),load.data_ptr<float>(),
  steps.data_ptr<int>(),status.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// v3: one launch for all 62 layers. ptrs/rows come from Python (data_ptr() of the held ids tensors, or the
// staging slices); they are kernel arguments, so a captured node replays the same sources.
void record_fused_cuda(std::vector<int64_t> ptrs,std::vector<int64_t> rows,torch::Tensor valid_rows,torch::Tensor enable,
  torch::Tensor counts,torch::Tensor load,torch::Tensor steps,torch::Tensor status){
 for(auto&t:{valid_rows,enable,counts,steps,status})ck(t,torch::kInt32);
 ck(load,torch::kFloat32);
 TORCH_CHECK(ptrs.size()==(size_t)L&&rows.size()==(size_t)L,"fused sources need exactly 62 pointers and row counts");
 TORCH_CHECK(valid_rows.numel()==1&&enable.numel()==1,"valid_rows/enable are single int32 words");
 TORCH_CHECK(counts.sizes()==torch::IntArrayRef({L,E})&&load.sizes()==torch::IntArrayRef({L,E})
  &&steps.numel()==L&&status.numel()==STATUS_WORDS,"stats buffer shapes");
 Sources s;
 for(int l=0;l<L;++l){
  TORCH_CHECK(ptrs[l]>0&&(ptrs[l]%4)==0,"fused source pointer must be a non-null int32 device address");
  TORCH_CHECK(rows[l]>=0&&rows[l]<=MAX_ROWS,"fused source rows must be in [0, 65536]");
  s.ids[l]=reinterpret_cast<const int*>(ptrs[l]);s.rows[l]=(int)rows[l];
 }
 fused_count_decay_k<<<L,THREADS,0,at::cuda::getCurrentCUDAStream()>>>(s,K,
  valid_rows.data_ptr<int>(),enable.data_ptr<int>(),counts.data_ptr<int>(),load.data_ptr<float>(),
  steps.data_ptr<int>(),status.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Dummy-row spread: rewrite the trailing dummy rows of this invocation's logical ids in place (current stream, capturable:
// the only non-argument input is the fixed-address `spread_rows` word). rows_ref = rows of the main pass.
void spread_cuda(torch::Tensor ids,int64_t rows_ref,torch::Tensor spread_rows){
 ck(ids,torch::kInt32);ck(spread_rows,torch::kInt32);
 TORCH_CHECK(ids.dim()==2&&ids.size(1)==K&&ids.size(0)>=0&&ids.size(0)<=MAX_ROWS,"ids must be int32 [rows<=65536, 6]");
 TORCH_CHECK(spread_rows.numel()==1,"spread_rows is a single int32 word");
 TORCH_CHECK(rows_ref>0&&rows_ref<=MAX_ROWS&&ids.size(0)<=rows_ref,"rows_ref must be in (0, 65536] and cover this invocation's rows");
 if(ids.size(0)==0)return;
 spread_dummy_rows_k<<<1,THREADS,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),(int)ids.size(0),(int)rows_ref,
  spread_rows.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Per-step rank sums of the replay just counted (current stream, capturable: fixed addresses, no arguments but the tensors).
void rank_sums_cuda(torch::Tensor counts,torch::Tensor rank_map,torch::Tensor steps,torch::Tensor enable,torch::Tensor ring){
 for(auto&t:{counts,rank_map,steps,enable,ring})ck(t,torch::kInt32);
 TORCH_CHECK(counts.sizes()==torch::IntArrayRef({L,E})&&rank_map.sizes()==torch::IntArrayRef({L,E})&&steps.numel()==L&&enable.numel()==1
  &&ring.sizes()==torch::IntArrayRef({RING_STEPS,L,R}),"rank sums buffer shapes");
 rank_sums_k<<<L,THREADS,0,at::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int>(),rank_map.data_ptr<int>(),steps.data_ptr<int>(),
  enable.data_ptr<int>(),ring.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reset_cuda(torch::Tensor counts,torch::Tensor load,torch::Tensor steps,torch::Tensor status){
 ck(counts,torch::kInt32);ck(load,torch::kFloat32);ck(steps,torch::kInt32);ck(status,torch::kInt32);
 TORCH_CHECK(counts.numel()==L*E&&load.numel()==L*E&&steps.numel()==L&&status.numel()==STATUS_WORDS,"reset shapes");
 reset_k<<<(L*E+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int>(),load.data_ptr<float>(),
  steps.data_ptr<int>(),status.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void snapshot_cuda(torch::Tensor counts,torch::Tensor load,torch::Tensor steps,torch::Tensor status,torch::Tensor packet){
 ck(counts,torch::kInt32);ck(load,torch::kFloat32);ck(steps,torch::kInt32);ck(status,torch::kInt32);ck(packet,torch::kInt32);
 TORCH_CHECK(counts.numel()==L*E&&load.numel()==L*E&&steps.numel()==L&&status.numel()==STATUS_WORDS
  &&packet.numel()==PACKET,"snapshot shapes");
 snapshot_k<<<(L*E+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int>(),load.data_ptr<float>(),
  steps.data_ptr<int>(),status.data_ptr<int>(),packet.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("record",&record_cuda);m.def("record_fused",&record_fused_cuda);m.def("reset",&reset_cuda);m.def("snapshot",&snapshot_cuda);
 m.def("spread",&spread_cuda);   // dummy-row spread
 m.def("rank_sums",&rank_sums_cuda);m.attr("RING_STEPS")=RING_STEPS;   // per-step rank load
 m.attr("LAYERS")=L;m.attr("EXPERTS")=E;m.attr("TOP_K")=K;m.attr("RANKS")=R;m.attr("LOCAL_SLOTS")=LOCAL_SLOTS;m.attr("SLOTS")=SLOTS;
 m.attr("MAX_ROWS")=MAX_ROWS;m.attr("STAGE_ROWS")=STAGE_ROWS;m.attr("STATUS_WORDS")=STATUS_WORDS;m.attr("PACKET_WORDS")=PACKET;m.attr("VALID_ALL")=VALID_ALL;
 m.attr("SPREAD_NONE")=SPREAD_NONE;
 m.attr("SNAP_LOAD")=SNAP_LOAD;m.attr("SNAP_COUNTS")=SNAP_COUNTS;m.attr("SNAP_STEPS")=SNAP_STEPS;m.attr("SNAP_STATUS")=SNAP_STATUS;
 m.attr("ST_BAD_EXPERT")=ST_BAD_EXPERT;m.attr("ST_PAD_SEEN")=ST_PAD_SEEN;m.attr("ST_BAD_LAYER")=ST_BAD_LAYER;m.attr("ST_BAD_ROWS")=ST_BAD_ROWS;
}
