//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"
#include "tensor_types.h"
#include "enum_map.h"

namespace opennn
{

enum class ActivationFunction { Identity, Sigmoid, Tanh, ReLU, Softmax, LeakyReLU, GELU, GELUTanh, SiLU };

// Negative-side slope for LeakyReLU. 0.1 matches the Darknet/YOLO default.
// The CUDA kernels keep their own mirrored copy in kernel_common.cuh, like the
// activation ids above.
inline constexpr float LEAKY_RELU_SLOPE = 0.1f;

const EnumMap<ActivationFunction>& activation_function_map();
const string& activation_function_to_string(ActivationFunction);
ActivationFunction activation_function_from_string(const string&);

bool activation_needs_input(ActivationFunction function);

inline float activation_forward_value(ActivationFunction function, float x)
{
    using enum ActivationFunction;
    switch (function)
    {
    case Identity:  return x;
    case Sigmoid:   return 1.0f / (1.0f + exp(-x));
    case Tanh:      return tanh(x);
    case ReLU:      return max(0.0f, x);
    case LeakyReLU: return x >= 0.0f ? x : x * LEAKY_RELU_SLOPE;
    case GELU:      return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    case GELUTanh:
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
    case SiLU:      return x / (1.0f + exp(-x));   // x * sigmoid(x) (Swish)
    case Softmax:   break;
    }

    throw runtime_error("activation_forward_value: Softmax must be handled separately.");
}

inline float activation_derivative_from_output_value(ActivationFunction function, float y)
{
    using enum ActivationFunction;
    switch (function)
    {
    case Identity:  return 1.0f;
    case Sigmoid:   return y * (1.0f - y);
    case Tanh:      return 1.0f - y * y;
    case ReLU:      return y > 0.0f ? 1.0f : 0.0f;
    case LeakyReLU: return y >= 0.0f ? 1.0f : LEAKY_RELU_SLOPE;
    case Softmax:   break;
    case GELU:
    case GELUTanh:
    case SiLU:      break;   // derivative needs the pre-activation input, not y
    }

    throw runtime_error("activation_derivative_from_output_value: Softmax/GELU/GELUTanh/SiLU must be handled separately.");
}

VectorR activation_forward_values(ActivationFunction, const VectorR&);
MatrixR activation_forward_values(ActivationFunction, const MatrixR&);
VectorR activation_derivative_from_output_values(ActivationFunction, const VectorR&);
MatrixR activation_derivative_from_output_values(ActivationFunction, const MatrixR&);
MatrixR activation_derivative_from_output_values(ActivationFunction, const MatrixMap&);

void bound(const TensorView&, const TensorView&, const TensorView&, TensorView&);

void scale(const TensorView&,
           const TensorView&, const TensorView&,
           const TensorView&, const TensorView&,
           const TensorView&,
           float, float,
           TensorView&);

void unscale(const TensorView&,
             const TensorView&, const TensorView&,
             const TensorView&, const TensorView&,
             const TensorView&,
             float, float,
             TensorView&);

void copy(const TensorView&, TensorView&);

void add(const TensorView&, const TensorView&, TensorView&);

void multiply(const TensorView&, bool, const TensorView&, bool, TensorView&, float alpha = 1.0f, float beta = 0.0f);

void softmax(TensorView&);

void activation_forward(TensorView&, ActivationFunction);
void activation_backward(const TensorView&, TensorView&, ActivationFunction);

void dropout_forward(TensorView&, Buffer&, float);
void dropout_backward(TensorView&, const Buffer&, float);

void linear_forward(const TensorView&, const TensorView&, const TensorView&,
                    TensorView&, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS,
                    TensorView* pre_activation = nullptr);
void linear_backward(const TensorView&, const TensorView&, const TensorView&,
                     const TensorView&, const TensorView&,
                     TensorView&, bool accumulate_input_delta = false);

void layer_normalization_forward(const TensorView&, const TensorView&, const TensorView&,
                        TensorView&, TensorView&,
                        TensorView&, TensorView&);
void layer_normalization_add_forward(const TensorView&, const TensorView&,
                            const TensorView&, const TensorView&,
                            TensorView&, TensorView&,
                            TensorView&, TensorView&, TensorView&);
void layer_normalization_backward(const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         TensorView&);

void rms_normalization_forward(const TensorView&, const TensorView&,
                      TensorView&, TensorView&, TensorView&, float);
void rms_normalization_backward(const TensorView&, const TensorView&,
                       const TensorView&, const TensorView&, const TensorView&,
                       const TensorView&, TensorView&);

// Rotary position embedding (RoPE). Fills [positions, rotary_dim] cos/sin tables
// (HF "rotate_half" convention: emb = cat(freqs, freqs)); rotary_forward rotates
// each head's first rotary_dim channels of a (batch, sequence, heads*head_dim)
// tensor by its sequence position. rotary_backward applies the inverse rotation.
void rotary_build_tables(TensorView&, TensorView&, Index sequence_length, Index rotary_dim, float base);
void rotary_forward(const TensorView&, const TensorView&, const TensorView&,
                    TensorView&, Index head_dim, Index rotary_dim, Index position_offset);
void rotary_backward(const TensorView&, const TensorView&, const TensorView&,
                     TensorView&, Index head_dim, Index rotary_dim, Index position_offset);

// SwiGLU gated activation: output = silu(gate) * up (element-wise).
void swiglu_forward(const TensorView&, const TensorView&, TensorView&);
void swiglu_backward(const TensorView&, const TensorView&, const TensorView&,
                     TensorView&, TensorView&);

// Grouped-query causal attention, laid out [batch, seq, heads*head_dim]; each
// key/value head is shared by n_query_heads/n_kv_heads query heads and head_dim
// is decoupled from the model width. query_position_offset is the absolute
// position of the first query (the KV-cache length when decoding), so query i
// attends keys 0..(offset+i); query_seq != key_seq is the KV-cache case.
// `decode_partials` (device scratch of grouped_attention_decode_scratch_floats
// fp32 values) enables the split-KV single-token decode kernel on GPU; when
// `position_device` is non-null it holds the cached-token count before this
// token on device (valid keys = *position_device + 1, CUDA-graph replay),
// otherwise the count comes from query_position_offset + 1.
void grouped_attention_forward(const TensorView& query, const TensorView& key, const TensorView& value,
                               TensorView& output, Index n_query_heads, Index n_kv_heads, Index head_dim,
                               bool causal, float scale, Index query_position_offset = 0,
                               float* decode_partials = nullptr, const int* position_device = nullptr);

// fp32 element count of the split-KV decode scratch for the shape above.
Index grouped_attention_decode_scratch_floats(Index n_query_heads, Index head_dim);

// Fused per-head QK-Norm + RoPE + KV-cache append for one decoded token (GPU
// only). `qkv_row` is the fused [q | k | v] projection row; rotated q goes to
// `q_out`, rotated k and raw v are appended to the caches at *position_device
// (cached tokens before this token, read on device for CUDA-graph replay).
// Empty norm weights skip QK-Norm. rotary_dim == head_dim.
void qk_rope_cache_append(const TensorView& qkv_row, const TensorView& q_norm_weight,
                          const TensorView& k_norm_weight, const TensorView& cos_table,
                          const TensorView& sin_table, TensorView& q_out,
                          TensorView& key_cache, TensorView& value_cache,
                          Index n_query_heads, Index n_kv_heads, Index head_dim,
                          float epsilon, const int* position_device);

// QK-Norm: RMSNorm applied independently to each head's head_dim vector of a
// (batch, seq, heads*head_dim) tensor, with a per-channel weight of size head_dim.
void qk_norm_forward(const TensorView& input, const TensorView& weight, TensorView& output,
                     Index head_dim, float epsilon);

// Tied language-model head: raw logits = input @ embed_weight^T, straight from
// the shared token-embedding matrix [vocabulary, hidden]; no softmax.
void tied_lm_head_forward(const TensorView& input, const TensorView& embed_weight, TensorView& output);

void embedding_lookup_forward(const TensorView&, const TensorView&,
                              const TensorView&, TensorView&,
                              Index, Index, Index,
                              bool, bool);
void embedding_lookup_backward(const TensorView&, const TensorView&,
                               const TensorView&, const TensorView&,
                               Index, Index, Index,
                               bool);

void max_pooling_3d_forward(const TensorView&, TensorView&, TensorView&, bool);
void average_pooling_3d_forward(const TensorView&, TensorView&);
void max_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void average_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void first_token_3d_forward(const TensorView&, TensorView&);
void first_token_3d_backward(const TensorView&, TensorView&);

void compute_token_valid_lengths(const TensorView&, Index, vector<Index>&);

void pooling_2d_forward(const TensorView&, TensorView&, TensorView&,
                        Index, Index, Index,
                        Index, Index,
                        Index, Index,
                        Index, Index,
                        bool);
void pooling_2d_backward(const TensorView&, const TensorView&,
                         TensorView&,
                         Index, Index, Index,
                         Index, Index,
                         Index, Index,
                         Index, Index,
                         bool);

void split_heads(const TensorView&, TensorView&);
void merge_heads(const TensorView&, TensorView&);

MatrixR append_rows(const MatrixR&, const MatrixR&);
MatrixR append_columns(const MatrixR&, const MatrixR&);
VectorR slice_rows(const VectorR&, const vector<Index>&);
MatrixR slice_rows(const MatrixR&, const vector<Index>&);
VectorI get_nearest_points(const MatrixR&, const VectorR&, int = 1);
MatrixR calculate_distances(const MatrixR&);
vector<Index> filter_selected_indices_by_column(const MatrixR&, const vector<Index>&, Index, float, float);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
