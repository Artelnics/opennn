#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cstdint>

#include "../eigen/Eigen/Core"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

using Eigen::Index;

// Optimizer

// `params_bf16` (last arg, default nullptr): when non-null, the kernel writes
// the freshly-updated FP32 master into the BF16 mirror in the same pass —
// avoids a separate cast_fp32_to_bf16_cuda call over the whole parameter set.
void adam_update_cuda(const Index, float*, float*, float*, const float*,
                      const float, const float, const float, const float,
                      const float, const float,
                      __nv_bfloat16* params_bf16 = nullptr);

void sgd_update_cuda(const Index, float*, float*, const float*,
                     const float, const float, const bool,
                     __nv_bfloat16* params_bf16 = nullptr);

void clip_gradient_norm_cuda(const Index n, float* gradient, const float* squared_norm, const float max_norm, const float eps);

// Cast FP32 source buffer to a BF16 destination buffer of the same length.
// Used to initialize the BF16 mirror of network parameters in
// NeuralNetwork::copy_parameters_device, and by Batch::copy_device_async to
// convert FP32 inputs (pinned host) to BF16 inputs (device). The post-step
// refresh of the mirror during Adam/SGD is folded into the optimizer kernels
// themselves (see params_bf16 above), not into this cast.
// When `stream` is null the host wrapper falls back to Backend::get_compute_stream().
void cast_fp32_to_bf16_cuda(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream = nullptr);
void cast_bf16_to_fp32_cuda(const Index n, const __nv_bfloat16* src, float* dst);

template<typename TIn>
void diff_to_fp32_cuda(const Index n, const TIn* input, const float* target, float* output);

template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            float scale, TOut* output);

// Errors

template<typename T>
void binary_cross_entropy_cuda(const Index, float*, const float*, const T*, const float);

template<typename T>
void binary_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float, const float);

template<typename T>
void multiple_cross_entropy_cuda(const Index, float*, const float*, const T*, const float);

template<typename T>
void multiple_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float);

template<typename T>
void weighted_squared_error_cuda(const Index, float*, const float*, const T*, const float, const float);

template<typename T>
void weighted_squared_error_gradient_cuda(const Index, T*, const float*, const T*, const float, const float, const float);

template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index, const int, const T*, const float*, float*, float*, float*, const float);

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index, const int, const T*, const float*, T*, const float);

template<typename T>
void cross_entropy_3d_multiple_backward_device_count_cuda(const Index, const int, const T*, const float*, T*, const float*);

void accumulate_scaled_metric_cuda(const float*, float, float*);

void accumulate_cross_entropy_3d_metrics_cuda(const float*, float*, float*);

// Regularization

template<typename T>
void l1_gradient_cuda(const Index, T*, const T*, const float);

// Bounding

template<typename TIn, typename TOut>
void bounding_cuda(const Index n, const int features, const TIn* input, const float* lower, const float* upper, TOut* output);

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
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

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

template<typename T>
void attention_sequence_lengths_cuda(const int batch_size,
                                     const int query_sequence_length,
                                     const int source_sequence_length,
                                     const int embedding_dimension,
                                     const T* source_input,
                                     int32_t* query_lengths,
                                     int32_t* source_lengths);

// Pooling 3D

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F);

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_grad, const float* indices, const int S, const int F);

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F);

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_grad, const int S, const int F);

// Dropout

template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long seed);

template<typename T>
void dropout_backward_cuda(const Index n, const T* output_delta, T* input_delta, const uint8_t* mask, const float rate);

// Activation

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function);

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function);

// Normalization Layer

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta);

// Recurrent helpers: gather/scatter a (batch, features) slice of a row-major
// (batch, time, features) tensor at a given timestep t. Used by RecurrentOp
// to materialise the per-step matrices that cuBLAS / activation kernels need.
template<typename T>
void gather_time_slice_cuda(const Index batch, const Index time_steps,
                            const Index features, const Index t,
                            const T* src, T* dst);

template<typename T>
void scatter_time_slice_cuda(const Index batch, const Index time_steps,
                             const Index features, const Index t,
                             const T* src, T* dst);

// Bias-broadcast + activation + (optional) derivative in one pass per step.
// activation_id matches ActivationOp::Function ordinal.
template<typename T>
void rnn_step_bias_activation_cuda(const Index batch, const Index out_features,
                                   T* hidden, const T* bias,
                                   T* derivs_or_null, const int activation_id);

// One-shot forward recurrent step: z = X[t] @ W_in + h[t-1] @ W_rec + b ; h = σ(z).
// Replaces the {2 cuBLAS gemms + bias_activation_kernel} sequence with a single
// kernel launch. One thread block per batch row; one thread per output feature.
// Requires out_features ≤ 1024 (CUDA block size limit). Uses shared memory for
// X[t] and h[t-1] row to avoid repeated global-memory reads from the inner loops.
//
// prev_hidden may be null at t=0 (recurrent term skipped).
// derivs_or_null receives σ'(z) when non-null (training mode).
template<typename T>
void rnn_step_fused_forward_cuda(const Index batch,
                                 const Index in_features,
                                 const Index out_features,
                                 const T* step_input,
                                 const T* prev_hidden,
                                 const T* W_in,
                                 const T* W_rec,
                                 const T* bias,
                                 T* step_hidden,
                                 T* derivs_or_null,
                                 const int activation_id);

// δz = δh ⊙ σ'(z): elementwise multiply (in-place into dst).
template<typename T>
void rnn_elementwise_multiply_cuda(const Index n, T* dst, const T* a);

// bias_grad[f] += sum over batch of delta[b, f]. bias_grad always FP32.
template<typename T>
void rnn_accumulate_bias_grad_cuda(const Index batch, const Index features,
                                   const T* delta, float* bias_grad);

// Fused backward "pre-gemm" kernel (Phase 3.E backward).
//
// For each (batch, out_feature):
//   δh   = first_iter ? output_delta[b, j] : next_carry[b, j]
//   δz   = δh * activation_derivatives[b, t, j]
//   delta[b, j] = δz
//   atomicAdd(bias_grad[j], δz)
//
// Replaces the per-step sequence:
//   copy(delta_src, delta) + gather(activation_derivatives) +
//   elementwise_multiply(delta *= step_derivs) + bias_grad_accumulate
// with a single kernel launch. bias_grad is always FP32 (Adam compute dtype),
// while delta uses the recurrent compute dtype (T).
template<typename T>
void rnn_step_fused_backward_pre_cuda(const Index batch,
                                      const Index out_features,
                                      const Index time_steps,
                                      const Index t,
                                      const bool first_iter,
                                      const T* output_delta,
                                      const T* next_carry,
                                      const T* activation_derivatives,
                                      T* delta,
                                      float* bias_grad);

#endif // KERNEL_CUH
