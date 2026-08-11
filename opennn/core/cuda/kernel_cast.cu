//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D T Y P E   C A S T   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_cast.cuh"

__global__ void cast_fp32_to_bf16_kernel(const int n_vec,
                                         const int n,
                                         const float* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    const float4* __restrict__ const src4 = reinterpret_cast<const float4*>(src);
    __nv_bfloat162* __restrict__ const dst2 = reinterpret_cast<__nv_bfloat162*>(dst);

    for (Index i = tid; i < n_vec; i += stride)
    {
        const float4 in = src4[i];
        dst2[i * 2 + 0] = __floats2bfloat162_rn(in.x, in.y);
        dst2[i * 2 + 1] = __floats2bfloat162_rn(in.z, in.w);
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
        dst[i] = __float2bfloat16(src[i]);
}

void cast_fp32_to_bf16(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream)
{
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    launch_vec4_on(stream, n, are_float4_aligned(src) && is_bfloat162_aligned(dst),
                   cast_fp32_to_bf16_kernel, src, dst);
}

__global__ void cast_bf16_to_fp32_kernel(const int n,
                                         const __nv_bfloat16* __restrict__ src,
                                         float* __restrict__ dst)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;
    for (Index i = tid; i < n; i += stride)
        dst[i] = __bfloat162float(src[i]);
}

void cast_bf16_to_fp32(const Index n, const __nv_bfloat16* src, float* dst)
{
    launch_elementwise(n, cast_bf16_to_fp32_kernel, src, dst);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
