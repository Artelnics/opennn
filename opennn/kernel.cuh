#ifndef KERNEL_CUH
#define KERNEL_CUH

#ifdef OPENNN_HAS_CUDA

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
                      __nv_bfloat16* parameters_bf16_mirror = nullptr);

void sgd_update_cuda(const Index, float*, float*, const float*,
                     const float, const float, const bool,
                     __nv_bfloat16* parameters_bf16_mirror = nullptr);

void sgd_update_capturable_cuda(
    const Index n, float* parameters, float* velocity, const float* gradients,
    const float* learning_rate_device, const float momentum, const bool nesterov,
    __nv_bfloat16* parameters_bf16_mirror = nullptr, cudaStream_t stream = nullptr);

void set_scalar_device_cuda(float* dst, const float value, cudaStream_t stream = nullptr);

void adam_update_capturable_cuda(
    const Index n, float* parameters, float* m, float* v, const float* gradients,
    const float beta_1, const float beta_2,
    const float learning_rate, const float epsilon,
    int* step_device, float* effective_lr_device, float* effective_eps_device,
    __nv_bfloat16* parameters_bf16_mirror = nullptr, cudaStream_t stream = nullptr);

void clip_gradient_norm_cuda(const Index n, float* gradient, const float* squared_norm, const float max_norm, const float eps);

void cast_fp32_to_bf16(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream = nullptr);
void cast_bf16_to_fp32(const Index n, const __nv_bfloat16* src, float* dst);

void gather_rows_cuda(const float* matrix, const int* row_indices, float* out,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream = nullptr);

void gather_rows_bf16_cuda(const float* matrix, const int* row_indices, __nv_bfloat16* out,
                           const Index n_rows, const Index n_cols,
                           const Index matrix_cols, const Index col_offset,
                           cudaStream_t stream = nullptr);

void scatter_time_slice_fill_cuda(const Index batch, const Index time_steps,
                                  const Index features, const Index t,
                                  const float* src, float* dst);

inline constexpr int RNN_COPY_MAX_REGIONS = 16;

struct RnnCopySpec
{
    const float* src = nullptr;
    float*       dst = nullptr;
    int rows = 0;
    int cols = 0;
    int transpose = 0;
};

void rnn_copy_regions_cuda(const RnnCopySpec* specs, int count,
                           cudaStream_t stream = nullptr);

void gather_window_rows_cuda(const float* matrix, const int* start_rows, float* out,
                             const Index batch, const Index past, const Index features,
                             const Index matrix_cols, const Index matrix_rows,
                             const Index col_offset, cudaStream_t stream = nullptr);

void gather_window_targets_cuda(const float* matrix, const int* start_rows, float* out,
                                const Index batch, const Index past, const Index future,
                                const Index target_cols, const bool multi_target,
                                const Index matrix_cols, const Index matrix_rows,
                                const Index col_offset, cudaStream_t stream = nullptr);

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
void categorical_cross_entropy_cuda(const Index, float*, const float*, const T*, const float);

template<typename T>
void categorical_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float);

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
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, float* positional_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);


template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D);

template<typename T>
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D);

// Fused padding/causal mask + row softmax over the attention scores: one pass
// over the score tensor instead of a mask pass plus a separate softmax pass.
template<typename T>
void attention_masked_softmax_cuda(const int batch_size, const int heads_number, const int query_sequence_length,
                          const int source_sequence_length, const int embedding_dimension,
                          const T* source_input, T* attention_weights, T* padding_mask,
                          const bool use_causal_mask, const bool zero_padded_queries);

template<typename T>
void attention_length_masked_softmax_cuda(const int batch_size, const int heads_number, const int query_sequence_length,
                                const int source_sequence_length, const int* host_lengths,
                                T* attention_weights, T* padding_mask, const bool use_causal_mask,
                                const bool zero_padded_queries);

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
void first_token_3d_forward_cuda(const int B, const int S, const int F, const T* in, T* out);

template<typename T>
void first_token_3d_backward_cuda(const int B, const int S, const int F, const T* delta, T* in_gradient);


template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long seed);

template<typename T>
void dropout_backward_cuda(const Index n, const T* output_delta, T* input_delta, const uint8_t* mask, const float rate);


template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function);

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function);


// Fused NHWC batchnorm inference: y = gamma*(x-mean)/sqrt(var+eps)+beta, plus
// optional residual add and ReLU in the same pass. residual may be null.
template<typename T>
void batchnorm_inference_cuda(const Index total, const Index channels,
                              const T* x, const T* residual,
                              const float* gamma, const float* beta,
                              const float* mean, const float* variance,
                              const float epsilon, const bool apply_relu, T* y);

// Folds inference-time batchnorm into the convolution parameters:
// W'[k,...] = W[k,...]*gamma[k]/sqrt(var[k]+eps), b'[k] = beta[k]-mean[k]*scale.
// kernel_size is R*S*C. transpose writes W' as [RSC, kernels] for the 1x1 GEMM path.
void conv_bn_fold_cuda(const Index kernels, const Index kernel_size,
                       const float* weights,
                       const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       const float epsilon, const bool transpose,
                       float* folded_weights, float* folded_bias);

// y = a + b, optionally clamped at zero (the residual tail of a folded conv+BN block).
void add_relu_cuda(const Index total, const float* a, const float* b,
                   const bool apply_relu, float* y);

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

// Fused residual-add + layernorm: S = X + R, writes S to `sum` and LayerNorm(S) to Y.
template<typename T>
void layernorm_add_forward_cuda(const int N, const int D, const T* X, const T* R, T* sum, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta);

// RMSNorm: Y = weight * X / sqrt(mean(X^2) + eps). `inv_rms` receives 1/rms per
// row (needed for backward); null skips the stash.
template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps);

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight);

// RoPE: rotate the first rotary_dim channels of each head (head_dim block) of a
// (rows, model_dim) tensor by its sequence position (row % seq + offset).
template<typename T>
void rope_forward_cuda(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* in, T* out, const float* cos, const float* sin);

template<typename T>
void rope_backward_cuda(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* dout, T* din, const float* cos, const float* sin);

// SwiGLU: out = silu(gate) * up (element-wise). Backward writes gate/up grads.
template<typename T>
void swiglu_forward_cuda(const int n, const T* gate, const T* up, T* out);

template<typename T>
void swiglu_backward_cuda(const int n, const T* dout, const T* gate, const T* up, T* dgate, T* dup);

// Grouped-query causal attention. Q/K/V/O laid out [batch, seq, heads*head_dim];
// keys/values have n_kv_heads. The general kernel runs one thread per query with
// an online softmax. Single-token decode (batch 1, query_seq 1, causal) instead
// uses a split-KV kernel — warps hold flash-style partials for a key subset,
// shared across the query heads of each kv head, merged by a combine pass — when
// `decode_partials` provides its scratch (grouped_attention_decode_scratch_floats
// fp32 values). `kv_length_device`, when non-null, overrides the host valid-key
// count so a captured graph replays correctly as the KV cache grows.
inline constexpr int GROUPED_ATTENTION_DECODE_SPLITS = 128;

template<typename T>
void grouped_attention_cuda(const int batch, const int query_seq, const int key_seq,
                            const int n_query_heads, const int n_kv_heads, const int head_dim,
                            const float scale, const int query_position_offset, const bool causal,
                            const int* kv_length_device, float* decode_partials,
                            const T* Q, const T* K, const T* V, T* O);

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

// Batch-parallel column-sum of delta (batch x features) into fp32 bias_grad.
// Caller must zero bias_grad first (atomicAdds). Scales to large batches.
template<typename T>
void bias_grad_sum_cuda(const Index batch, const Index features,
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

// YOLO DetectionOperator

// Apply sigmoid(xy), exp(wh)*anchor, sigmoid(obj), softmax|sigmoid(classes)
// per box across the (batch, grid, grid, boxes_per_cell) tile.
// class_activation: 0 = softmax, 1 = sigmoid (mirrors DetectionOperator::ClassActivation).
// anchors layout: flat [aw0, ah0, aw1, ah1, ...] of length 2*boxes_per_cell.
void detection_forward_cuda(const Index batch_size,
                            const Index grid_size,
                            const Index boxes_per_cell,
                            const Index classes_number,
                            const Index channels,
                            const int class_activation,
                            const float* anchors,
                            const float* input,
                            float* output);

// Chain rule through the same. For (x, y, obj) and sigmoid classes the
// gate is d_sig = out * (1-out). For (w, h) the gate is d_exp = out. For
// softmax classes the per-box Jacobian collapses to out * (delta - <delta, out>).
void detection_backward_cuda(const Index batch_size,
                             const Index grid_size,
                             const Index boxes_per_cell,
                             const Index classes_number,
                             const Index channels,
                             const int class_activation,
                             const float* output,
                             const float* output_delta,
                             float* input_delta);

// YOLOv8-style anchor-free DetectionV8Operator
// All 4+C channels are sigmoid-gated: sigmoid(tx/ty/tw/th/cls...).
void detection_v8_forward_cuda(Index batch_size,
                               Index grid_size,
                               Index grid_width,
                               Index classes_number,
                               const float* input,
                               float* output);

void detection_v8_backward_cuda(Index batch_size,
                                Index grid_size,
                                Index grid_width,
                                Index classes_number,
                                const float* output,
                                const float* output_delta,
                                float* input_delta);

// Nearest-neighbor upsample (NHWC).
void upsample_forward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                           const float* src, float* dst);
void upsample_backward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                            const float* out_delta, float* in_delta);

// Channel concatenation (NHWC) — one call per input slice.
void concat_forward_slice_cuda(int batch, int H, int W,
                               int slice_ch, int total_ch, int ch_offset,
                               const float* src, float* dst);
void concat_backward_slice_cuda(int batch, int H, int W,
                                int slice_ch, int total_ch, int ch_offset,
                                const float* out_delta, float* in_delta);

// GIoU YOLO loss — one thread per (batch * grid * grid * box).
// error_accumulator must be pre-zeroed on device; result is added atomically.
void yolo_error_cuda(const float* output, const float* target, float* error_accumulator,
                     int batch, int grid, int boxes_per_cell, int values_per_box,
                     int classes_number, int sigmoid_classes,
                     float lambda_giou, float lambda_noobj, float lambda_class,
                     float focal_gamma, float obj_focal_gamma);

// GIoU YOLO gradient — delta is zeroed inside, then filled per-box.
void yolo_gradient_cuda(const float* output, const float* target, float* delta,
                        int batch, int grid, int boxes_per_cell, int values_per_box,
                        int classes_number, int sigmoid_classes, float inv_batch,
                        float lambda_giou, float lambda_noobj, float lambda_class,
                        float focal_gamma, float obj_focal_gamma);

#endif // OPENNN_HAS_CUDA

#endif // KERNEL_CUH
