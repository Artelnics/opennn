// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/network/layers/kernel_upsampling.cuh"

__global__ void upsampling_forward_kernel(
    const int n,
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int channels, const int scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        Index b; int oh, ow, c;
        nhwc_decompose(i, channels, out_w, out_h, b, oh, ow, c);

        const Index source = ((b * in_h + oh / scale) * in_w + ow / scale) * channels + c;
        dst[i] = src[source];
    }
}

__global__ void upsampling_backward_kernel(
    const int n,
    const float* __restrict__ out_delta,
    float* __restrict__ in_delta,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int channels, const int scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        Index b; int ih, iw, c;
        nhwc_decompose(i, channels, in_w, in_h, b, ih, iw, c);

        const Index first_row = b * out_h + Index(ih) * scale;
        const Index first_col = Index(iw) * scale;

        float sum = 0.0f;
        for (int u = 0; u < scale; ++u)
        {
            const Index row_start = ((first_row + u) * out_w + first_col) * channels + c;
            for (int v = 0; v < scale; ++v)
                sum += out_delta[row_start + Index(v) * channels];
        }

        in_delta[i] = sum;
    }
}

void upsampling_forward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                             const float* src, float* dst)
{
    const int n = checked_int(Index(batch) * Index(in_h * scale)
                            * Index(in_w * scale) * Index(channels));
    launch_elementwise_strided(n, upsampling_forward_kernel,
                       src, dst, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}

void upsampling_backward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                              const float* out_delta, float* in_delta)
{
    const int n = checked_int(Index(batch) * Index(in_h) * Index(in_w) * Index(channels));
    launch_elementwise_strided(n, upsampling_backward_kernel,
                       out_delta, in_delta, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}
