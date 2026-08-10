#ifndef KERNEL_UPSAMPLE_CUH
#define KERNEL_UPSAMPLE_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void upsample_forward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                           const float* src, float* dst);
void upsample_backward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                            const float* out_delta, float* in_delta);

#endif

#endif
