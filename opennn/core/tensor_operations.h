//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/enum_map.h"

namespace opennn
{

inline constexpr float INV_SQRT_2      = 0.70710678118654752440f;
inline constexpr float INV_SQRT_2_PI   = 0.39894228040143267794f;
inline constexpr float SQRT_2_OVER_PI  = 0.7978845608028654f;
inline constexpr float GELU_TANH_CUBIC = 0.044715f;

inline float gelu_value(float x)
{
    return 0.5f * x * (1.0f + erff(x * INV_SQRT_2));
}

inline float gelu_tanh_value(float x)
{
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_TANH_CUBIC * x * x * x)));
}

inline float silu_value(float x)
{
    return x / (1.0f + exp(-x));
}

inline float gelu_derivative(float x)
{
    const float cdf = 0.5f * (1.0f + erff(x * INV_SQRT_2));
    const float pdf = INV_SQRT_2_PI * expf(-0.5f * x * x);
    return cdf + x * pdf;
}

inline float gelu_tanh_derivative(float x)
{
    const float x2 = x * x;
    const float u = SQRT_2_OVER_PI * (x + GELU_TANH_CUBIC * x * x2);
    const float t = tanhf(u);
    const float du = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_TANH_CUBIC * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

inline float silu_derivative(float x)
{
    const float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

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
    case GELU:      return gelu_value(x);
    case GELUTanh:  return gelu_tanh_value(x);
    case SiLU:      return silu_value(x);
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
    case SiLU:      break;
    }

    throw runtime_error("activation_derivative_from_output_value: Softmax/GELU/GELUTanh/SiLU must be handled separately.");
}

template<typename TensorType>
typename TensorType::PlainObject activation_forward_values(ActivationFunction function, const TensorType& values)
{
    return values.unaryExpr([function](float value) { return activation_forward_value(function, value); });
}

template<typename TensorType>
typename TensorType::PlainObject activation_derivative_from_output_values(ActivationFunction function, const TensorType& values)
{
    return values.unaryExpr([function](float value) { return activation_derivative_from_output_value(function, value); });
}

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
                    TensorView* pre_activation = nullptr,
                    const TensorView& weight_scale = {});
void linear_backward(const TensorView&, const TensorView&, const TensorView&,
                     const TensorView&, const TensorView&,
                     TensorView&, bool accumulate_input_delta = false,
                     const TensorView* drelu_mask = nullptr);



void rotary_build_tables(TensorView&, TensorView&, Index sequence_length, Index rotary_dim, float base);
void rotary_forward(const TensorView&, const TensorView&, const TensorView&,
                    TensorView&, Index head_dim, Index rotary_dim, Index position_offset);
void rotary_backward(const TensorView&, const TensorView&, const TensorView&,
                     TensorView&, Index head_dim, Index rotary_dim, Index position_offset);

void swiglu_forward(const TensorView&, const TensorView&, TensorView&);
void swiglu_backward(const TensorView&, const TensorView&, const TensorView&,
                     TensorView&, TensorView&);

void grouped_attention_forward(const TensorView& query, const TensorView& key, const TensorView& value,
                               TensorView& output, Index n_query_heads, Index n_kv_heads, Index head_dim,
                               bool causal, float scale, Index query_position_offset = 0,
                               float* decode_partials = nullptr, const int* position_device = nullptr);

Index grouped_attention_decode_scratch_floats(Index n_query_heads, Index head_dim);

void qk_rope_cache_append(const TensorView& qkv_row, const TensorView& q_norm_weight,
                          const TensorView& k_norm_weight, const TensorView& cos_table,
                          const TensorView& sin_table, TensorView& q_out,
                          TensorView& key_cache, TensorView& value_cache,
                          Index n_query_heads, Index n_kv_heads, Index head_dim,
                          float epsilon, const int* position_device);

void sample_logits_row(const TensorView& logits_row, float temperature, Index top_k, float top_p,
                       unsigned long long seed, unsigned long long step,
                       void* candidates_scratch, int* id_device, float* token_device);

Index sample_logits_scratch_floats();

void qk_norm_forward(const TensorView& input, const TensorView& weight, TensorView& output,
                     Index head_dim, float epsilon);

void tied_lm_head_forward(const TensorView& input, const TensorView& embed_weight, TensorView& output,
                          const TensorView& weight_scale = {});

void embedding_lookup_forward(const TensorView&, const TensorView&,
                              const TensorView&, TensorView&,
                              Index, Index, Index,
                              bool, bool,
                              const TensorView& weight_scale = {});
void embedding_lookup_backward(const TensorView&, const TensorView&,
                               const TensorView&, const TensorView&,
                               Index, Index, Index,
                               bool);


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
VectorI get_nearest_points(const MatrixR&, const VectorR&, int = 1);
MatrixR calculate_distances(const MatrixR&);
vector<Index> filter_selected_indices_by_column(const MatrixR&, const vector<Index>&, Index, float, float);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
