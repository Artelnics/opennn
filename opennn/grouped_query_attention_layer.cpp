//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O U P E D   Q U E R Y   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_types.h"
#include "grouped_query_attention_layer.h"

namespace opennn
{

GroupedQueryAttention::GroupedQueryAttention(const Shape& new_input_shape,
                                             Index new_q_heads, Index new_kv_heads, Index new_head_dim,
                                             float new_rope_theta, float new_rms_epsilon,
                                             bool new_use_qk_norm,
                                             const string& new_name)
    : Layer(LayerType::GroupedQueryAttention)
{
    operators = {&attention};

    set(new_input_shape, new_q_heads, new_kv_heads, new_head_dim,
        new_rope_theta, new_rms_epsilon, new_use_qk_norm, new_name);
}

void GroupedQueryAttention::set(const Shape& new_input_shape,
                                Index new_q_heads, Index new_kv_heads, Index new_head_dim,
                                float new_rope_theta, float new_rms_epsilon,
                                bool new_use_qk_norm,
                                const string& new_label)
{
    sequence_length = new_input_shape.dim_or_zero(0);
    hidden          = new_input_shape.dim_or_zero(1);
    q_heads         = new_q_heads;
    kv_heads        = new_kv_heads;
    head_dim        = new_head_dim;
    rope_theta      = new_rope_theta;
    rms_epsilon     = new_rms_epsilon;
    use_qk_norm     = new_use_qk_norm;

    set_label(new_label);

    attention.set(sequence_length, hidden, q_heads, kv_heads, head_dim,
                  rope_theta, rms_epsilon, use_qk_norm);
}

void GroupedQueryAttention::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank < 2) return;
    set({new_input_shape[0], new_input_shape[1]},
        q_heads, kv_heads, head_dim, rope_theta, rms_epsilon, use_qk_norm, label);
}

void GroupedQueryAttention::read_JSON_body(const Json* element)
{
    const Shape new_input_shape = string_to_shape(read_json_string(element, "InputDimensions"));
    const Index new_q_heads  = read_json_index(element, "QueryHeads");
    const Index new_kv_heads = read_json_index(element, "KeyValueHeads");
    const Index new_head_dim = read_json_index(element, "HeadDim");
    const float new_rope_theta  = read_json_float(element, "RopeTheta");
    const float new_rms_epsilon = read_json_float(element, "RmsEpsilon");

    // Optional; defaults to the Qwen3 style.
    const bool new_use_qk_norm = element->has("QKNorm") ? read_json_bool(element, "QKNorm") : true;

    set(new_input_shape, new_q_heads, new_kv_heads, new_head_dim,
        new_rope_theta, new_rms_epsilon, new_use_qk_norm, get_label());
}

void GroupedQueryAttention::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"QueryHeads",    to_string(q_heads)},
        {"KeyValueHeads", to_string(kv_heads)},
        {"HeadDim",       to_string(head_dim)},
        {"RopeTheta",     to_string(rope_theta)},
        {"RmsEpsilon",    to_string(rms_epsilon)},
        {"QKNorm",        to_string(use_qk_norm)}
    });
}

REGISTER(Layer, GroupedQueryAttention, "GroupedQueryAttention")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
