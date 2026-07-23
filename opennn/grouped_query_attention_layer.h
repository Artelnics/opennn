//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O U P E D   Q U E R Y   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "grouped_query_attention_operator.h"

namespace opennn
{

// Self-attention layer of a LLaMA-family decoder: grouped-query attention
// (q_heads >= kv_heads) with decoupled head_dim, RoPE, causal mask and optional
// per-head QK-Norm (the Qwen3 variant). (sequence, hidden) -> same shape.
// Inference forward only.
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
    void set_input_shape(const Shape&) override;

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
