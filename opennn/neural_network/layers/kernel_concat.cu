//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_concat.cuh"

template<bool Scatter>
__global__ void concat_slice_kernel(
    const int n,
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int H, const int W,
    const int slice_ch, const int total_ch, const int ch_offset)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        Index b; int h, w, c;
        nhwc_decompose(i, slice_ch, W, H, b, h, w, c);
        const Index strided = ((b * H + h) * W + w) * total_ch + ch_offset + c;
        if constexpr (Scatter) dst[strided] = src[i];
        else                   dst[i] = src[strided];
    }
}

void concat_forward_slice_cuda(const int batch, const int H, const int W,
                               const int slice_ch, const int total_ch, const int ch_offset,
                               const float* src, float* dst)
{
    launch_elementwise_strided(Index(batch) * H * W * slice_ch, concat_slice_kernel<true>,
                       src, dst, H, W, slice_ch, total_ch, ch_offset);
}

void concat_backward_slice_cuda(const int batch, const int H, const int W,
                                const int slice_ch, const int total_ch, const int ch_offset,
                                const float* out_delta, float* in_delta)
{
    launch_elementwise_strided(Index(batch) * H * W * slice_ch, concat_slice_kernel<false>,
                       out_delta, in_delta, H, W, slice_ch, total_ch, ch_offset);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
