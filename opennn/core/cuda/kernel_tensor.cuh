#ifndef KERNEL_TENSOR_CUH
#define KERNEL_TENSOR_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename T>
void transpose_2d_cuda(const Index rows, const Index cols,
                       const T* src, T* dst);

template<typename T>
void bias_grad_sum_cuda(const Index batch, const Index features,
                        const T* delta, float* bias_grad);

#endif

#endif
