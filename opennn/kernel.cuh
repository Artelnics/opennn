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

// Cast FP32 source buffer to a BF16 destination buffer of the same length.
// Used to refresh the BF16 working copy of network parameters after each
// Adam step (master FP32 → working BF16 mirror).
void cast_fp32_to_bf16_cuda(const Index n, const float* src, __nv_bfloat16* dst);

// (input - target) into a FP32 buffer; input dtype templated, target/output FP32.
// Used by the loss helpers to bridge BF16 activations and FP32 targets without
// going through cuDNN OpTensor (which doesn't accept mixed dtypes).
template<typename TIn>
void diff_to_fp32_cuda(const Index n, const TIn* input, const float* target, float* output);

// scale * (input - target), with input/output dtypes independent. Used for the
// gradient of squared-error losses where output (input_delta) follows the
// activation dtype while target stays FP32.
template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            float scale, TOut* output);

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
void cross_entropy_3d_multiple_forward_cuda(const Index, const int, const T*, const float*, float*, float*, float*, const float);

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index, const int, const T*, const float*, T*, const float);

// Regularization

template<typename T>
void l1_gradient_cuda(const Index, T*, const T*, const float);

// Bounding

template<typename TIn, typename TOut>
void bounding_cuda(const Index n, const int features, const TIn* input, const float* lower, const float* upper, TOut* output);

// Scaling / Unscaling. Mixed-dtype: input is typically FP32 from the Batch (or
// BF16 from the previous layer for unscale), output may follow the activation
// dtype downstream or stay FP32 for the network's final output.

template<typename TIn, typename TOut>
void scale_cuda(const Index n, const int features,
                const TIn* input,
                const float* minimums, const float* maximums,
                const float* means, const float* standard_deviations,
                const float* scalers,
                float min_range, float max_range,
                TOut* output);

template<typename TIn, typename TOut>
void unscale_cuda(const Index n, const int features,
                  const TIn* input,
                  const float* minimums, const float* maximums,
                  const float* means, const float* standard_deviations,
                  const float* scalers,
                  float min_range, float max_range,
                  TOut* output);

// Embedding

template<typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding);

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

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
