//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_concat.cuh"

template<typename T, bool Scatter>
__global__ void concat_slice_kernel(
    const int n,
    const T* __restrict__ src,
    T* __restrict__ dst,
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

template<typename T, bool Scatter>
void slice_channels_cuda(const int batch, const int H, const int W,
                         const int slice_ch, const int total_ch, const int ch_offset,
                         const T* src, T* dst)
{
    launch_elementwise_strided(Index(batch) * H * W * slice_ch, concat_slice_kernel<T, Scatter>,
                               src, dst, H, W, slice_ch, total_ch, ch_offset);
}

#define INSTANTIATE(T) \
    template void slice_channels_cuda<T, true>(const int, const int, const int, const int, const int, const int, const T*, T*); \
    template void slice_channels_cuda<T, false>(const int, const int, const int, const int, const int, const int, const T*, T*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

void concat_forward_slice_cuda(const int batch, const int H, const int W,
                               const int slice_ch, const int total_ch, const int ch_offset,
                               const float* src, float* dst)
{
    slice_channels_cuda<float, true>(batch, H, W, slice_ch, total_ch, ch_offset, src, dst);
}

void concat_backward_slice_cuda(const int batch, const int H, const int W,
                                const int slice_ch, const int total_ch, const int ch_offset,
                                const float* out_delta, float* in_delta)
{
    slice_channels_cuda<float, false>(batch, H, W, slice_ch, total_ch, ch_offset, out_delta, in_delta);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
