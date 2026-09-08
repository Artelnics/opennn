#ifndef KERNEL_UPSAMPLING_CUH
#define KERNEL_UPSAMPLING_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void upsampling_forward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                             const float* src, float* dst);
void upsampling_backward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                              const float* out_delta, float* in_delta);

#endif

#endif
