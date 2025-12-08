/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedLayernormKernels/layernorm_param.h"
#include "tensorrt_llm/kernels/fusedLayernormKernels/ws_layernorm.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace torch_ext
{

// Fused Add + RMSNorm + FP4 Quantization kernel
// input: [M, N] - input tensor (fp16/bf16)
// residual: [M, N] - residual tensor (fp16/bf16)
// gamma: [N] - RMSNorm weight (fp16/bf16)
// sf_scale: [1] - optional scale factor for FP4 quantization (float)
// use_rms_norm: bool - if true use RMSNorm, else use LayerNorm
// Returns:
//   normed_output: [M, N/8] - FP4 quantized normalized output (uint32_t, packed)
//   output: [M, N] - pre-norm output (input + residual), same dtype as input
//   sf_out: scale factors for FP4 (uint8_t), swizzled layout
//
// NOTE: This kernel requires SM90 (Hopper) or SM100 (Blackwell) GPU architecture.
// NOTE: Hidden dimension N must be >= 2048 and <= 16384.
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_add_rms_norm_quant(at::Tensor const& input,
    at::Tensor const& residual, at::Tensor const& gamma, std::optional<at::Tensor> const& sf_scale, bool use_rms_norm)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    CHECK_TH_CUDA(residual);
    CHECK_CONTIGUOUS(residual);
    CHECK_TH_CUDA(gamma);
    CHECK_CONTIGUOUS(gamma);

    // Check GPU architecture - kernel requires SM90+ (Hopper/Blackwell)
    auto const device = input.get_device();
    cudaDeviceProp props;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    TORCH_CHECK(props.major >= 9,
        "fused_add_rms_norm_quant requires SM90 (Hopper) or newer GPU architecture. "
        "Current device: sm_",
        props.major, props.minor);

    auto const& inputShape = input.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank == 2, "input should be 2D tensor [M, N].");
    TORCH_CHECK(residual.sizes() == inputShape, "residual shape must match input shape.");

    int64_t const m = inputShape[0];
    int64_t const n = inputShape[1];

    TORCH_CHECK(gamma.sizes()[0] == n, "gamma size must match hidden dimension N.");
    TORCH_CHECK(n >= 2048, "Hidden dimension N must be >= 2048 (kernel constraint).");
    TORCH_CHECK(n <= 16384, "Hidden dimension N must be <= 16384.");
    TORCH_CHECK(n % 16 == 0, "Hidden dimension N must be divisible by 16 for FP4 quantization.");

    // Validate sf_scale if provided
    float* sfScalePtr = nullptr;
    if (sf_scale.has_value())
    {
        CHECK_INPUT(sf_scale.value(), torch::kFloat32);
        sfScalePtr = sf_scale.value().data_ptr<float>();
    }

    // Allocate output tensors
    // normed_output: FP4 packed output [M, N/8] as uint32_t (8 FP4 values packed per uint32)
    auto normed_output = at::detail::empty_cuda({m, n / 8}, torch::kInt32, input.device(), std::nullopt);

    // output: pre-norm output (input + residual) [M, N], same dtype as input
    auto output = at::detail::empty_cuda({m, n}, input.scalar_type(), input.device(), std::nullopt);

    // sf_out: scale factors for FP4, swizzled layout
    // sfVecSize = 16 for FP4 quantization (16 FP4 values share one scale factor)
    int64_t const sfVecSize = 16;
    int64_t const sfSize = tensorrt_llm::computeSwizzledLayoutSFSize(m, n / sfVecSize);
    auto sf_out = at::detail::empty_cuda({sfSize}, SF_DTYPE, input.device(), std::nullopt);

    // Get number of SMs for persistent kernel
    static int const multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    // Allocate counters for warp-specialized kernel using PyTorch allocator
    // This avoids cudaMalloc/cudaFree overhead and ensures proper cleanup
    auto counters_tensor
        = at::detail::empty_cuda({static_cast<int64_t>(sizeof(tensorrt_llm::kernels::WarpSpecializedCounters))},
            torch::kByte, input.device(), std::nullopt);
    counters_tensor.zero_();
    auto* counters
        = reinterpret_cast<tensorrt_llm::kernels::WarpSpecializedCounters*>(counters_tensor.mutable_data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(device);

#define LAUNCH_FUSED_ADD_RMS_NORM_QUANT(T)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        using Param = tensorrt_llm::kernels::GeneralFP4AddBiasResidualPreLayerNormParam<T>;                            \
        tensorrt_llm::kernels::WarpSpecializedParam<Param> param;                                                      \
        param.normed_output = reinterpret_cast<uint32_t*>(normed_output.data_ptr());                                   \
        param.output = reinterpret_cast<T*>(output.data_ptr());                                                        \
        param.input = const_cast<T*>(reinterpret_cast<T const*>(input.data_ptr()));                                    \
        param.sf_scale = sfScalePtr;                                                                                   \
        param.sf_out = reinterpret_cast<uint32_t*>(sf_out.data_ptr());                                                 \
        param.residual = reinterpret_cast<T const*>(residual.data_ptr());                                              \
        param.bias = nullptr;                                                                                          \
        param.gamma = reinterpret_cast<T const*>(gamma.data_ptr());                                                    \
        param.beta = nullptr;                                                                                          \
        param.m = static_cast<int>(m);                                                                                 \
        param.n = static_cast<int>(n);                                                                                 \
        param.layernorm_eps = 1e-5f;                                                                                   \
        param.stream = stream;                                                                                         \
        param.counters = counters;                                                                                     \
        tensorrt_llm::kernels::invokeWSLayerNorm<Param>(param, use_rms_norm, multiProcessorCount);                     \
    } while (0)

    if (input.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_FUSED_ADD_RMS_NORM_QUANT(half);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_FUSED_ADD_RMS_NORM_QUANT(__nv_bfloat16);
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for fused_add_rms_norm_quant with bf16 input.");
#endif
    }
    else
    {
        C10_THROW_ERROR(
            NotImplementedError, "fused_add_rms_norm_quant only supports input tensor with dtypes fp16/bf16.");
    }

#undef LAUNCH_FUSED_ADD_RMS_NORM_QUANT

    // No explicit sync needed - kernel runs asynchronously on the stream
    // counters_tensor will be freed when it goes out of scope (after stream sync at Python level)

    return std::make_tuple(normed_output, output, sf_out);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_add_rms_norm_quant(Tensor input, Tensor residual, Tensor gamma, "
        "Tensor? sf_scale, bool use_rms_norm=True) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_add_rms_norm_quant", &torch_ext::fused_add_rms_norm_quant);
}
