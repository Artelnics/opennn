#ifndef KERNEL_POOL3D_CUH
#define KERNEL_POOL3D_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F,
                                 const int* valid_lengths);

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_grad, const float* indices, const int S, const int F);

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F,
                                     const int* valid_lengths);

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_grad, const int S, const int F,
                                      const int* valid_lengths);

#endif

#endif
