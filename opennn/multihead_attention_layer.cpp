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
                                       const string& new_name)
    : Layer(LayerType::MultiHeadAttention)
{
    // Forward order V, K, Q ensures Layer::back_propagate (reverse iteration) runs
    // as output_projection -> merge -> attention -> query_projection ->
    // key_projection -> value_projection. That order is what produces correctly
    // accumulated dQ + dK + dV input deltas in self-attention (Q overwrites,
    // K/V accumulate) and matches the gradient computation order of the original
    // monolithic backward.
    operators = {&value_projection, &key_projection, &query_projection,
                 &attention, &merge, &output_projection};

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
        {{batch_size, query_sequence_length, embedding_dimension},                  compute_dtype}, // InputQueryDelta - final dInput query
        {{batch_size, source_sequence_length, embedding_dimension},                 compute_dtype}, // InputSourceDelta - final dInput source
        {{batch_size, heads_number, query_sequence_length, source_sequence_length}, compute_dtype}, // AttentionWeightDelta - unfused scratch
        {{batch_size, heads_number, source_sequence_length, head_dimension},        compute_dtype}, // ValueHeadDelta - dV, head shape
        {{batch_size, query_sequence_length, embedding_dimension},                  compute_dtype}, // ConcatenatedOutputDelta - dConcat, embed shape
        {{batch_size, heads_number, query_sequence_length, head_dimension},         compute_dtype}, // QueryHeadDelta - dQ, head shape
        {{batch_size, heads_number, source_sequence_length, head_dimension},        compute_dtype}, // KeyHeadDelta - dK, head shape
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

    // Shared across all 3 projections.
    for (auto* proj : {&query_projection, &key_projection, &value_projection})
    {
        proj->input_slots  = {Input};
        proj->scratch_slots = {TransposeScratch};
        proj->input_delta_slots_self = {InputQueryDelta};
    }

    // Each backward intermediate has its own dedicated slot so Layer::back_propagate
    // (reverse iteration) works without temporal-alias contracts:
    //   - output_projection writes ConcatenatedOutputDelta (concat-shape).
    //   - merge reads ConcatenatedOutputDelta, writes head-shape into forward
    //     TransposeScratch.
    //   - attention reads forward TransposeScratch as dO, writes QueryHeadDelta,
    //     KeyHeadDelta, ValueHeadDelta (and AttentionWeightDelta on unfused path).
    //   - Q reads QueryHeadDelta, writes InputQueryDelta (accumulate=false).
    //   - K reads KeyHeadDelta, writes InputQueryDelta (self, accumulate) or
    //     InputSourceDelta (cross, overwrite).
    //   - V reads ValueHeadDelta, accumulates onto K's destination.
    query_projection.output_slots = {Query};
    query_projection.input_view_index = 0;
    query_projection.output_delta_slots = {QueryHeadDelta};
    query_projection.input_delta_slots_cross = {InputQueryDelta};
    query_projection.accumulate_input_delta_self  = false;
    query_projection.accumulate_input_delta_cross = false;

    key_projection.output_slots = {Key};
    key_projection.input_view_index = 1;
    key_projection.output_delta_slots = {KeyHeadDelta};
    key_projection.input_delta_slots_cross = {InputSourceDelta};
    key_projection.accumulate_input_delta_self  = true;
    key_projection.accumulate_input_delta_cross = false;

    value_projection.output_slots = {Value};
    value_projection.input_view_index = 1;
    value_projection.output_delta_slots = {ValueHeadDelta};
    value_projection.input_delta_slots_cross = {InputSourceDelta};
    value_projection.accumulate_input_delta_self  = true;
    value_projection.accumulate_input_delta_cross = true;

    attention.input_slots  = {Query, Key, Value, Input};
    attention.output_slots = {AttentionWeights, AttentionWeightsDropped};
    attention.scratch_slots = {TransposeScratch};
    attention.source_view_index = 1;
    attention.attention_output_slots = {ConcatenatedAttentionOutputs};
    attention.output_delta_slots = {AttentionWeightDelta, QueryHeadDelta, KeyHeadDelta, ValueHeadDelta};

    output_projection.input_slots  = {ConcatenatedAttentionOutputs};
    output_projection.output_slots = {Output};
    output_projection.output_delta_slots = {OutputDelta};
    output_projection.input_delta_slots  = {ConcatenatedOutputDelta};

    merge.set(heads_number, query_sequence_length, head_dimension, compute_dtype);
    merge.input_slots  = {TransposeScratch};
    merge.output_slots = {ConcatenatedAttentionOutputs};
    merge.output_delta_slots = {ConcatenatedOutputDelta};
}

void MultiHeadAttention::read_JSON_body(const Json* root_element)
{
    const string new_label = read_json_string(root_element, "Label");
    const Shape new_input_shape = string_to_shape(read_json_string(root_element, "InputDimensions"));
    const Index new_source_sequence_length = read_json_index(root_element, "SourceSequenceLength");
    const Index new_heads_number = read_json_index(root_element, "HeadsNumber");
    const bool  new_use_causal_mask = read_json_bool(root_element, "CausalMask");

    set(new_input_shape.dim_or_zero(0),
        new_source_sequence_length,
        new_input_shape.dim_or_zero(1),
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
