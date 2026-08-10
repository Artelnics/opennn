#ifndef KERNEL_CONCAT_CUH
#define KERNEL_CONCAT_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void concat_forward_slice_cuda(int batch, int H, int W,
                               int slice_ch, int total_ch, int ch_offset,
                               const float* src, float* dst);
void concat_backward_slice_cuda(int batch, int H, int W,
                                int slice_ch, int total_ch, int ch_offset,
                                const float* out_delta, float* in_delta);

#endif

#endif
