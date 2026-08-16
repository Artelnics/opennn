#ifndef KERNEL_CONCAT_CUH
#define KERNEL_CONCAT_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// Moves the channel slice [ch_offset, ch_offset + slice_ch) of a (batch, H, W,
// total_ch) tensor to or from a compact (batch, H, W, slice_ch) one: Scatter
// writes compact -> strided, otherwise strided -> compact.
template<typename T, bool Scatter>
void slice_channels_cuda(int batch, int H, int W,
                         int slice_ch, int total_ch, int ch_offset,
                         const T* src, T* dst);

void concat_forward_slice_cuda(int batch, int H, int W,
                               int slice_ch, int total_ch, int ch_offset,
                               const float* src, float* dst);
void concat_backward_slice_cuda(int batch, int H, int W,
                                int slice_ch, int total_ch, int ch_offset,
                                const float* out_delta, float* in_delta);

#endif

#endif
