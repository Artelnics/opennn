//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O U P E D   Q U E R Y   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

// Attention, rotary embedding and QK normalisation. This layer is the
// only caller; the tests reach them through this header.
void rotary_build_tables(TensorView&, TensorView&, Index sequence_length, Index rotary_dim, float base);
void rotary_forward(const TensorView&, const TensorView&, const TensorView&,
                    TensorView&, Index head_dim, Index rotary_dim, Index position_offset);
void rotary_backward(const TensorView&, const TensorView&, const TensorView&,
                     TensorView&, Index head_dim, Index rotary_dim, Index position_offset);
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
void qk_norm_forward(const TensorView& input, const TensorView& weight, TensorView& output,
                     Index head_dim, float epsilon);

struct GroupedQueryAttentionOperator : Operator
{
    Index sequence_length = 0;
    Index hidden          = 0;
    Index q_heads         = 0;
    Index kv_heads        = 0;
    Index head_dim        = 0;
    float rope_theta      = 1000000.0f;
    float rms_epsilon     = 1.0e-6f;

    bool use_qk_norm = true;

    TensorView q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    TensorView q_scale, k_scale, v_scale, o_scale, qkv_scale;

    bool qkv_fused = false;

    void set(Index new_sequence_length, Index new_hidden,
             Index new_q_heads, Index new_kv_heads, Index new_head_dim,
             float new_rope_theta, float new_rms_epsilon, bool new_use_qk_norm);

    Index q_dim()  const { return q_heads  * head_dim; }
    Index kv_dim() const { return kv_heads * head_dim; }

    vector<TensorSpec> parameter_specs() const override;
    vector<SlotQuantization> parameter_quantization() const override;
    void link_parameters(span<const TensorView>) override;
    void link_parameter_scales(span<const TensorView>) override;
    void set_parameters_random() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void forward_gpu(TensorView& input, TensorView& output, Index batch, Index past,
                     Index query_capacity, const int* position_device);

    Buffer kv_key, kv_value;
    Index cache_capacity = 0;
    Type cache_dtype = Type::FP32;
};

class GroupedQueryAttention final : public Layer
{
public:

    GroupedQueryAttention(const Shape& = Shape({0, 0}),
                          Index q_heads = 0, Index kv_heads = 0, Index head_dim = 0,
                          float rope_theta = 1000000.0f, float rms_epsilon = 1.0e-6f,
                          bool use_qk_norm = true,
                          const string& = "grouped_query_attention");

    Shape get_input_shape() const noexcept override { return { sequence_length, hidden }; }
    Shape get_output_shape() const override { return { sequence_length, hidden }; }

    Index get_sequence_length() const { return sequence_length; }
    Index get_hidden() const { return hidden; }
    Index get_q_heads() const { return q_heads; }
    Index get_kv_heads() const { return kv_heads; }
    Index get_head_dim() const { return head_dim; }
    bool  get_use_qk_norm() const { return use_qk_norm; }

    void set(const Shape&, Index, Index, Index, float, float, bool, const string&);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 2, 3); }

    void apply_input_shape(const Shape&) override;

    void on_compute_dtype_changed() override { attention.compute_dtype = get_compute_dtype(); }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index sequence_length = 0;
    Index hidden          = 0;
    Index q_heads         = 0;
    Index kv_heads        = 0;
    Index head_dim        = 0;
    float rope_theta      = 1000000.0f;
    float rms_epsilon     = 1.0e-6f;
    bool  use_qk_norm     = true;

    GroupedQueryAttentionOperator attention;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
