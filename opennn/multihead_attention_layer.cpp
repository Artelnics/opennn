//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "multihead_attention_layer.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "random_utilities.h"

namespace opennn
{

MultiHeadAttention::MultiHeadAttention(const Shape& new_input_shape,
                                       Index new_heads_number,
                                       const string& new_name)
    : MultiHeadAttention(new_input_shape, new_input_shape, new_heads_number, new_name)
{
}

MultiHeadAttention::MultiHeadAttention(const Shape& new_query_dimensions,
                                       const Shape& new_source_dimensions,
                                       Index new_heads_number,
                                       const string& new_name) : Layer()
{
    name = "MultiHeadAttention";
    layer_type = LayerType::MultiHeadAttention;

    if (new_query_dimensions[1] != new_source_dimensions[1])
        throw runtime_error("embedding dimension must be the same for query and source.");

    set(new_query_dimensions[0],
        new_source_dimensions[0],
        new_query_dimensions[1],
        new_heads_number,
        false,
        new_name);
}

Shape MultiHeadAttention::get_input_shape() const
{
    return { query_sequence_length, embedding_dimension };
}

Shape MultiHeadAttention::get_output_shape() const
{
    return get_input_shape();
}

vector<Operator*> MultiHeadAttention::get_operators()
{
    return {&query_projection, &key_projection, &value_projection, &output_projection};
}

vector<pair<Shape, Type>> MultiHeadAttention::get_forward_specs(Index batch_size) const
{
    const Index head_dimension = get_head_dimension();
    const Index max_seq = max(query_sequence_length, source_sequence_length);

    const auto attention_scratch = attention.forward_scratch_specs(batch_size);

    return {
        {{batch_size, heads_number, query_sequence_length, head_dimension},  compute_dtype}, // Query
        {{batch_size, heads_number, source_sequence_length, head_dimension}, compute_dtype}, // Key
        attention_scratch[0],                                                                   // AttentionWeights
        attention_scratch[1],                                                                   // AttentionWeightsDropped
        {{batch_size, query_sequence_length, embedding_dimension},           compute_dtype}, // ConcatenatedAttentionOutputs
        {{batch_size, heads_number, source_sequence_length, head_dimension}, compute_dtype}, // Value
        {{batch_size, max_seq, embedding_dimension},                         compute_dtype}, // TransposeScratch
        {{batch_size, query_sequence_length, embedding_dimension},           compute_dtype}, // Output
    };
}

vector<pair<Shape, Type>> MultiHeadAttention::get_backward_specs(Index batch_size) const
{
    const Index head_dimension = get_head_dimension();

    return {
        {{batch_size, query_sequence_length, embedding_dimension},                  compute_dtype}, // InputQueryDelta
        {{batch_size, source_sequence_length, embedding_dimension},                 compute_dtype}, // InputSourceDelta
        {{batch_size, heads_number, query_sequence_length, source_sequence_length}, compute_dtype}, // AttentionWeightDelta
        {{batch_size, heads_number, source_sequence_length, head_dimension},        compute_dtype}, // ValueDelta (transposed)
    };
}

void MultiHeadAttention::set(Index new_query_sequence_length,
                             Index new_source_sequence_length,
                             Index new_embedding_dimension,
                             Index new_heads_number,
                             bool new_use_causal_mask,
                             const string& new_label)
{
    query_sequence_length  = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    embedding_dimension    = new_embedding_dimension;
    heads_number           = new_heads_number;

    set_label(new_label);

    if (new_heads_number == 0 && new_embedding_dimension == 0)
        return;

    if (new_heads_number <= 0)
        throw runtime_error("Heads number must be greater than 0.");

    if (new_embedding_dimension % new_heads_number != 0)
        throw runtime_error("The embedding dimension must be divisible by the number of heads.");

    const Index head_dimension = get_head_dimension();

    query_projection .set(embedding_dimension, heads_number, head_dimension, compute_dtype);
    key_projection   .set(embedding_dimension, heads_number, head_dimension, compute_dtype);
    value_projection .set(embedding_dimension, heads_number, head_dimension, compute_dtype);
    output_projection.set(embedding_dimension, embedding_dimension, compute_dtype);

    attention.set(heads_number, head_dimension,
                  query_sequence_length, source_sequence_length,
                  new_use_causal_mask, compute_dtype);
}

void MultiHeadAttention::forward_propagate(ForwardPropagation& forward_propagation,
                                           size_t layer,
                                           bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& query_input = get_query_input(forward_views);
    const TensorView& source_input = get_source_input(forward_views);

    TensorView& query = forward_views[Query][0];
    TensorView& key = forward_views[Key][0];
    TensorView& value = forward_views[Value][0];
    TensorView& attention_weights = forward_views[AttentionWeights][0];
    TensorView& concatenated = forward_views[ConcatenatedAttentionOutputs][0];
    TensorView& output = forward_views.back()[0];

    const Index batch_size = forward_propagation.batch_size;

    float* transpose_scratch = forward_views[TransposeScratch][0].as<float>();

    query_projection.apply(query_input,  query, transpose_scratch);
    key_projection  .apply(source_input, key,   transpose_scratch);
    value_projection.apply(source_input, value, transpose_scratch);

    TensorView attention_out_scratch(transpose_scratch, get_heads_shape(batch_size), compute_dtype);

    attention.apply(query, key, value, source_input,
                    attention_weights,
                    forward_views[AttentionWeightsDropped][0],
                    attention_out_scratch,
                    transpose_scratch,
                    is_training);

    TensorView concatenated_4d = concatenated.reshape(get_concat_shape(batch_size));
    merge_heads(attention_out_scratch, concatenated_4d);

    const Shape flat_shape = {batch_size * query_sequence_length, embedding_dimension};
    TensorView concatenated_2d = concatenated.reshape(flat_shape);
    TensorView output_2d       = output.reshape(flat_shape);

    output_projection.apply(concatenated_2d, output_2d);
}

void MultiHeadAttention::back_propagate(ForwardPropagation& forward_propagation,
                                        BackPropagation& back_propagation,
                                        size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& delta_views = back_propagation.delta_views[layer];

    const TensorView& query_input = get_query_input(forward_views);
    const TensorView& source_input = get_source_input(forward_views);
    const bool self_attention = is_self_attention(forward_views);

    TensorView& output_delta = delta_views[OutputDelta][0];

    const Index batch_size = forward_propagation.batch_size;
    const Shape flat_shape = {batch_size * query_sequence_length, embedding_dimension};

    float* transpose_scratch = forward_views[TransposeScratch][0].as<float>();

    TensorView concat_gradient_flat = delta_views[InputQueryDelta][0].reshape(flat_shape);
    TensorView output_delta_flat = output_delta.reshape(flat_shape);
    TensorView concat_in_flat = forward_views[ConcatenatedAttentionOutputs][0].reshape(flat_shape);

    output_projection.apply_delta(output_delta_flat,
                                  concat_in_flat,
                                  concat_gradient_flat,
                                  false);

    TensorView& att_weight_gradient = delta_views[AttentionWeightDelta][0];
    TensorView& value_gradient      = delta_views[ValueDelta][0];

    const Index head_dimension = get_head_dimension();

    TensorView query_gradient(delta_views[InputQueryDelta][0].as<float>(),
                          get_heads_shape(batch_size),
                          compute_dtype);
    TensorView key_gradient(delta_views[InputSourceDelta][0].as<float>(),
                        {batch_size, heads_number, source_sequence_length, head_dimension},
                        compute_dtype);

    TensorView concat_gradient_4d = delta_views[InputQueryDelta][0].reshape(get_concat_shape(batch_size));
    TensorView scratch_4d     = forward_views[TransposeScratch][0].reshape(get_heads_shape(batch_size));

    split_heads(concat_gradient_4d, scratch_4d);

    attention.apply_delta(forward_views[Query][0],
                          forward_views[Key][0],
                          forward_views[Value][0],
                          forward_views[ConcatenatedAttentionOutputs][0],
                          forward_views[AttentionWeights][0],
                          forward_views[AttentionWeightsDropped][0],
                          scratch_4d,
                          att_weight_gradient,
                          query_gradient,
                          key_gradient,
                          value_gradient);

    query_projection.apply_delta(query_gradient, query_input,
                                 delta_views[InputQueryDelta][0],
                                 false, transpose_scratch);

    TensorView& kv_input_gradient = self_attention
        ? delta_views[InputQueryDelta][0]
        : delta_views[InputSourceDelta][0];

    key_projection.apply_delta(key_gradient, source_input,
                               kv_input_gradient,
                               self_attention, transpose_scratch);

    value_projection.apply_delta(value_gradient, source_input,
                                 kv_input_gradient,
                                 true, transpose_scratch);
}
void MultiHeadAttention::read_JSON_body(const Json* root_element)
{
    const string new_label = read_json_string(root_element, "Label");
    const Shape new_input_shape = string_to_shape(read_json_string(root_element, "InputDimensions"));
    const Index new_source_sequence_length = read_json_index(root_element, "SourceSequenceLength");
    const Index new_heads_number = read_json_index(root_element, "HeadsNumber");
    const bool  new_use_causal_mask = read_json_bool(root_element, "CausalMask");

    set(new_input_shape.empty() ? Index(0) : new_input_shape[0],
        new_source_sequence_length,
        new_input_shape.rank >= 2 ? new_input_shape[1] : Index(0),
        new_heads_number, new_use_causal_mask, new_label);
}

void MultiHeadAttention::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"SourceSequenceLength", to_string(source_sequence_length)},
        {"HeadsNumber", to_string(heads_number)},
        {"CausalMask", to_string(attention.use_causal_mask)}
    });
}

REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
