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

// Four elements per thread (one 8-byte load, one float4 store) when both
// pointers allow it, a scalar tail otherwise - the widening twin of the kernel
// above. This is the per-step cast of every convolution weight gradient in
// BF16 training (cuDNN's wgrad stores BF16 for these shapes), 53 launches over
// ResNet-50's 25M parameters.
__global__ void cast_bf16_to_fp32_kernel(const int n_vec,
                                         const int n,
                                         const __nv_bfloat16* __restrict__ src,
                                         float* __restrict__ dst)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    const __nv_bfloat162* __restrict__ const src2 = reinterpret_cast<const __nv_bfloat162*>(src);
    float4* __restrict__ const dst4 = reinterpret_cast<float4*>(dst);

    for (Index i = tid; i < n_vec; i += stride)
    {
        const uint2 raw = reinterpret_cast<const uint2*>(src2)[i];
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        dst4[i] = make_float4(lo.x, lo.y, hi.x, hi.y);
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
        dst[i] = __bfloat162float(src[i]);
}

void cast_bf16_to_fp32(const Index n, const __nv_bfloat16* src, float* dst)
{
    const bool aligned = are_float4_aligned(dst)
        && (reinterpret_cast<std::uintptr_t>(src) & 0x7) == 0;
    launch_vec4_on(opennn::device::get_compute_stream(), n, aligned,
                   cast_bf16_to_fp32_kernel, src, dst);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
