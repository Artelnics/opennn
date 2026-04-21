#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "../../opennn/eigen/unsupported/Eigen/CXX11/Tensor"

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>

// System includes

#include <iostream>
#include <stdio.h>
#include <string>
#include <time.h>

using namespace std;
using namespace Eigen;

typedef float type;

// Optimizer

void adam_update_cuda(const Index, float*, float*, float*, const float*, const float, const float, const float, const float, const float, const float);

void sgd_update_cuda(const Index, float*, float*, const float*, const float, const float, const bool);

// Errors

template<typename T>
void binary_cross_entropy_cuda(const Index, float*, const T*, const T*, const float);

template<typename T>
void binary_cross_entropy_gradient_cuda(const Index, T*, const T*, const T*, const float, const float);

template<typename T>
void multiple_cross_entropy_cuda(const Index, float*, const T*, const T*, const float);

template<typename T>
void multiple_cross_entropy_gradient_cuda(const Index, T*, const T*, const T*, const float);

template<typename T>
void weighted_squared_error_cuda(const Index, float*, const T*, const T*, const float, const float);

template<typename T>
void weighted_squared_error_gradient_cuda(const Index, T*, const T*, const T*, const float, const float, const float);

template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index, const int, const T*, const float*, float*, float*, const float);

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index, const int, const T*, const float*, T*, const float);

// Regularization

template<typename T>
void l1_gradient_cuda(const Index, T*, const T*, const float);

// Bias add — FP32 bias broadcast onto dtype-T output (replaces cudnnAddTensor
// when bias and output dtypes differ, e.g. FP32 bias + BF16 activation output).

template<typename T>
void add_bias_cuda(const Index, T*, const float*, const int);

// Addition

template<typename T>
void addition_cuda(const Index, const T*, const T*, T*);

// Embedding

template<typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding);

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_gradients, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

// MultiHead Attention

template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D);

template<typename T>
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D);

template<typename T>
void attention_masks_cuda(const int batch_size, const int heads_number, const int query_sequence_length,
                          const int source_sequence_length, const int embedding_dimension,
                          const T* source_input, T* attention_weights, T* padding_mask,
                          const bool use_causal_mask);

// Pooling 3D

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F);

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_grad, const float* indices, const int S, const int F);

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F);

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_grad, const int S, const int F);

// Normalization Layer

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta);

#endif // KERNEL_CUH
