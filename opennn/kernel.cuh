#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#if defined(__CUDACC__) && defined(_MSC_VER)
__host__ __device__ inline float arg(float x) noexcept
{
    return x < 0.0f ? 3.14159265358979323846f : 0.0f;
}

__host__ __device__ inline double arg(double x) noexcept
{
    return x < 0.0 ? 3.14159265358979323846 : 0.0;
}

template<typename Integer, typename = std::enable_if_t<std::is_integral_v<Integer>>>
__host__ __device__ inline double arg(Integer x) noexcept
{
    if constexpr (std::is_signed_v<Integer>)
        return x < 0 ? 3.14159265358979323846 : 0.0;
    else
        return 0.0;
}
#endif

#include <Eigen/Core>

using Eigen::Index;


void adam_update_cuda(const Index, float*, float*, float*, const float*,
                      const float, const float, const float, const float,
                      const float, const float,
                      __nv_bfloat16* params_bf16 = nullptr);

void sgd_update_cuda(const Index, float*, float*, const float*,
                     const float, const float, const bool,
                     __nv_bfloat16* params_bf16 = nullptr);

void sgd_update_capturable_cuda(
    const Index n, float* parameters, float* velocity, const float* gradients,
    const float* learning_rate_device, const float momentum, const bool nesterov,
    __nv_bfloat16* params_bf16 = nullptr, cudaStream_t stream = nullptr);

void set_scalar_device_cuda(float* dst, const float value, cudaStream_t stream = nullptr);

void adam_update_capturable_cuda(
    const Index n, float* parameters, float* m, float* v, const float* gradients,
    const float beta_1, const float beta_2,
    const float learning_rate, const float epsilon,
    int* step_device, float* effective_lr_device, float* effective_eps_device,
    __nv_bfloat16* params_bf16 = nullptr, cudaStream_t stream = nullptr);

void clip_gradient_norm_cuda(const Index n, float* gradient, const float* squared_norm, const float max_norm, const float eps);

void cast_fp32_to_bf16_cuda(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream = nullptr);
void cast_bf16_to_fp32_cuda(const Index n, const __nv_bfloat16* src, float* dst);

void gather_rows_cuda(const float* matrix, const int* row_indices, float* out,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream = nullptr);

void gather_rows_bf16_cuda(const float* matrix, const int* row_indices, __nv_bfloat16* out,
                           const Index n_rows, const Index n_cols,
                           const Index matrix_cols, const Index col_offset,
                           cudaStream_t stream = nullptr);

template<typename TIn>
void diff_to_fp32_cuda(const Index n, const TIn* input, const float* target, float* output);

template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            float scale, TOut* output);


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


template<typename T>
void l1_gradient_cuda(const Index, T*, const T*, const float);


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


template<typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);


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


template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F);

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_grad, const float* indices, const int S, const int F);

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F);

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_grad, const int S, const int F);


template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long seed);

template<typename T>
void dropout_backward_cuda(const Index n, const T* output_delta, T* input_delta, const uint8_t* mask, const float rate);


template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function);

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function);


template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta);

template<typename T>
void gather_time_slice_cuda(const Index batch, const Index time_steps,
                            const Index features, const Index t,
                            const T* src, T* dst);

template<typename T>
void scatter_time_slice_cuda(const Index batch, const Index time_steps,
                             const Index features, const Index t,
                             const T* src, T* dst);

template<typename T>
void transpose_2d_cuda(const Index rows, const Index cols,
                       const T* src, T* dst);

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

template<typename T>
void rnn_elementwise_multiply_cuda(const Index n, T* dst, const T* a);

template<typename T>
void rnn_accumulate_bias_grad_cuda(const Index batch, const Index features,
                                   const T* delta, float* bias_grad);

template<typename T>
void rnn_step_fused_backward_pre_cuda(const Index batch,
                                      const Index out_features,
                                      const Index time_steps,
                                      const Index t,
                                      const bool first_iter,
                                      const T* output_delta,
                                      const T* next_carry,
                                      const T* activation_derivatives,
                                      T* delta);

#endif // KERNEL_CUH
