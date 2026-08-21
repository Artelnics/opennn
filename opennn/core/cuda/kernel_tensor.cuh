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
//
// `activation` is an ActivationFunction value applied to each result before it
// is stored, or Identity for none. A single-output layer produces one element
// per row, so an activation of its own would be a launch that reads and writes
// one number per row; here it is a register operation.
template<typename T>
void linear_forward_single_output_cuda(const Index rows, const Index features,
                                       const T* input, const T* weights,
                                       const T* bias, const int activation, T* output);

// Backward of that layer: the input delta, the weight gradient and the bias
// gradient in one pass over the input, where cuBLAS reads it twice through two
// GEMVs. input_delta may be null (nothing consumes it), bias_gradient may be
// null (the layer has no bias). Returns false without launching anything when
// the feature count does not divide into whole 16-byte vectors per lane, which
// is the caller's signal to keep cuBLAS. With fuse_input_relu the input
// delta is also masked by the derivative of the ReLU that produced the
// input, which costs nothing here: that ReLU's output is what the pass
// already reads to build the weight gradient.
template<typename T>
bool linear_backward_single_output_cuda(const Index rows, const Index features,
                                        const T* output_delta, const T* input,
                                        const T* weights, T* input_delta,
                                        bool fuse_input_relu,
                                        float* weight_gradient, float* bias_gradient);

#endif

#endif
