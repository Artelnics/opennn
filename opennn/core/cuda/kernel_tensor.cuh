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

// Forward of a dense layer with a single output: one value per row, so the
// GEMM degenerates to a row-wise dot product against one weight vector. cuBLAS
// dispatches a general GEMV there and moves the activation at roughly half the
// bandwidth a streaming reduction reaches, which matters because the operation
// is entirely limited by reading the input. Requires the feature count to fill
// whole 16-byte vectors; callers check that and keep cuBLAS otherwise.
template<typename T>
void linear_forward_single_output_cuda(const Index rows, const Index features,
                                       const T* input, const T* weights,
                                       const T* bias, T* output);

#endif

#endif
