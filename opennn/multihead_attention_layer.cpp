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
    if (new_query_dimensions[1] != new_source_dimensions[1])
        throw runtime_error("MultiHeadAttention Error: embedding dimension must be the same for query and source.");

    set(new_query_dimensions[0],
        new_source_dimensions[0],
        new_query_dimensions[1],
        new_heads_number,
        false,
        new_name);
}

float MultiHeadAttention::get_scaling_factor() const
{
    const Index head_dimension = get_head_dimension();
    return (head_dimension == 0) ? 0.25f : float(1) / float(sqrt(head_dimension));
}

Index MultiHeadAttention::get_head_dimension() const
{
    return (heads_number == 0)
        ? 0
        : Index(embedding_dimension / heads_number);
}

Shape MultiHeadAttention::get_input_shape() const
{
    return { query_sequence_length, embedding_dimension };
}

Shape MultiHeadAttention::get_output_shape() const
{
    return get_input_shape();
}

vector<pair<Shape, Type>> MultiHeadAttention::get_parameter_specs() const
{
    const Type act = activation_dtype;
    return {
        /*QueryWeight*/      {{embedding_dimension, embedding_dimension}, act},
        /*QueryBias*/        {{embedding_dimension},                      act},
        /*KeyWeight*/        {{embedding_dimension, embedding_dimension}, act},
        /*KeyBias*/          {{embedding_dimension},                      act},
        /*ValueWeight*/      {{embedding_dimension, embedding_dimension}, act},
        /*ValueBias*/        {{embedding_dimension},                      act},
        /*ProjectionWeight*/ {{embedding_dimension, embedding_dimension}, act},
        /*ProjectionBias*/   {{embedding_dimension},                      act},
    };
}

void MultiHeadAttention::set_parameters_random()
{
    if (embedding_dimension == 0) return;

    const float weight_limit = sqrt(float(6) / float(2 * embedding_dimension));

    for (const int slot : {QueryWeight, KeyWeight, ValueWeight, ProjectionWeight})
        if (!parameters[slot].empty())
            set_random_uniform(VectorMap(parameters[slot].as<float>(), parameters[slot].size()),
                               -weight_limit, weight_limit);

    for (const int slot : {QueryBias, KeyBias, ValueBias, ProjectionBias})
        if (!parameters[slot].empty())
            parameters[slot].fill(0.0f);
}

void MultiHeadAttention::set(Index new_query_sequence_length,
                             Index new_source_sequence_length,
                             Index new_embedding_dimension,
                             Index new_heads_number,
                             bool new_use_causal_mask,
                             const string& new_label)
{
    name = "MultiHeadAttention";
    layer_type = LayerType::MultiHeadAttention;
    query_sequence_length = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    embedding_dimension = new_embedding_dimension;
    heads_number = new_heads_number;
    label = new_label;

    if (new_heads_number == 0 && new_embedding_dimension == 0)
    {
        heads_number = 0;
        return;
    }

    if (new_heads_number <= 0)
        throw runtime_error("MultiHeadAttention Error: Heads number must be greater than 0.");

    if (new_embedding_dimension % new_heads_number != 0)
        throw runtime_error("MultiHeadAttention Error: The embedding dimension must be divisible by the number of heads.");

    use_causal_mask = new_use_causal_mask;

    if (use_causal_mask)
    {
        causal_mask.resize(query_sequence_length, source_sequence_length);

        for (Index row = 0; row < query_sequence_length; ++row)
            for (Index column = 0; column < source_sequence_length; ++column)
                causal_mask(row, column) = (column > row) ? NEG_INFINITY : float(0);
    }

    query_projection .set(embedding_dimension, embedding_dimension, activation_dtype);
    key_projection   .set(embedding_dimension, embedding_dimension, activation_dtype);
    value_projection .set(embedding_dimension, embedding_dimension, activation_dtype);
    output_projection.set(embedding_dimension, embedding_dimension, activation_dtype);
}

float* MultiHeadAttention::link_parameters(float* pointer)
{
    pointer = Layer::link_parameters(pointer);

    if (parameters.size() > ProjectionBias)
    {
        query_projection .link_parameters({parameters[QueryBias],      parameters[QueryWeight]});
        key_projection   .link_parameters({parameters[KeyBias],        parameters[KeyWeight]});
        value_projection .link_parameters({parameters[ValueBias],      parameters[ValueWeight]});
        output_projection.link_parameters({parameters[ProjectionBias], parameters[ProjectionWeight]});
    }

    return pointer;
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

    const Index batch_size     = forward_propagation.batch_size;
    const Index head_dimension = get_head_dimension();

    float* transpose_scratch = forward_views[TransposeScratch][0].as<float>();

    auto apply_qkv_projection = [&](Combination& combo, const TensorView& input,
                                    Index seq_len, TensorView& head_output)
    {
        const Index rows = batch_size * seq_len;
        TensorView input_2d   = input.reshape({rows, embedding_dimension});
        TensorView scratch_2d(transpose_scratch, {rows, embedding_dimension}, head_output.type);
        TensorView scratch_4d(transpose_scratch,
                              {batch_size, seq_len, heads_number, head_dimension},
                              head_output.type);
        combo.apply(input_2d, scratch_2d);
        split_heads(scratch_4d, head_output);
    };

    apply_qkv_projection(query_projection, query_input,  query_sequence_length,  query);
    apply_qkv_projection(key_projection,   source_input, source_sequence_length, key);
    apply_qkv_projection(value_projection, source_input, source_sequence_length, value);

    multiply(query, false, key, true, attention_weights, get_scaling_factor(), float(0));

    attention_masks(source_input, attention_weights, causal_mask, use_causal_mask, forward_views[PaddingMask][0].as<float>());

    softmax(attention_weights);

    const bool apply_dropout = is_training && dropout.active();
    TensorView& attention_used = apply_dropout
        ? forward_views[AttentionWeightsDropped][0]
        : attention_weights;

    if (apply_dropout)
    {
        copy(attention_weights, attention_used);
        dropout.apply(attention_used);
    }

    TensorView attention_out_scratch = forward_views[AttentionOutputTransposed][0].reshape(heads_shape(batch_size));
    TensorView concatenated_4d       = concatenated.reshape(concat_shape(batch_size));

    multiply(attention_used, false, value, false, attention_out_scratch);
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
    auto& gradient_views = back_propagation.gradient_views[layer];

    const TensorView& query_input = get_query_input(forward_views);
    const TensorView& source_input = get_source_input(forward_views);
    const bool self_attention = is_self_attention(forward_views);

    TensorView& output_delta = delta_views[OutputDelta][0];

    const Index batch_size = forward_propagation.batch_size;
    const Shape flat_shape = {batch_size * query_sequence_length, embedding_dimension};

    float* transpose_scratch = forward_views[TransposeScratch][0].as<float>();
    const float scaling_factor = get_scaling_factor();

    TensorView concat_grad_flat = delta_views[ConcatenatedOutputDelta][0].reshape(flat_shape);
    TensorView output_delta_flat = output_delta.reshape(flat_shape);
    TensorView concat_in_flat    = forward_views[ConcatenatedAttentionOutputs][0].reshape(flat_shape);

    output_projection.apply_delta(output_delta_flat,
                                  concat_in_flat,
                                  concat_grad_flat,
                                  gradient_views[ProjectionWeight],
                                  gradient_views[ProjectionBias],
                                  false);

    const TensorView& attention_weights = forward_views[AttentionWeights][0];
    TensorView& concat_grad     = delta_views[ConcatenatedOutputDelta][0];
    TensorView& att_weight_grad = delta_views[AttentionWeightDelta][0];
    TensorView& query_grad      = delta_views[QueryDelta][0];
    TensorView& key_grad        = delta_views[KeyDelta][0];
    TensorView& value_grad      = delta_views[ValueDelta][0];

    TensorView concat_grad_4d = concat_grad.reshape(concat_shape(batch_size));
    TensorView scratch_4d     = forward_views[TransposeScratch][0].reshape(heads_shape(batch_size));

    split_heads(concat_grad_4d, scratch_4d);

    const TensorView& attention_forward_output = dropout.active()
        ? forward_views[AttentionWeightsDropped][0]
        : attention_weights;

    multiply(attention_forward_output, true, scratch_4d, false, value_grad);
    multiply(scratch_4d, false, forward_views[Value][0], true, att_weight_grad);

    if (dropout.active())
        dropout.apply_delta(att_weight_grad);

    softmax_backward(attention_weights, att_weight_grad);

    multiply(att_weight_grad, false, forward_views[Key][0],   false, query_grad, scaling_factor, float(0));
    multiply(att_weight_grad, true,  forward_views[Query][0], false, key_grad,   scaling_factor, float(0));

    const Index head_dimension = get_head_dimension();

    auto apply_qkv_projection_grad = [&](const Combination& combo,
                                         const TensorView& head_grad,
                                         const TensorView& input,
                                         Index seq_len,
                                         TensorView& input_grad,
                                         TensorView& weight_grad,
                                         TensorView& bias_grad,
                                         bool accumulate)
    {
        const Index rows = batch_size * seq_len;
        TensorView scratch_4d(transpose_scratch,
                              {batch_size, seq_len, heads_number, head_dimension},
                              head_grad.type);
        merge_heads(head_grad, scratch_4d);

        TensorView scratch_2d(transpose_scratch, {rows, embedding_dimension}, head_grad.type);
        TensorView input_2d      = input.reshape({rows, embedding_dimension});
        TensorView input_grad_2d = input_grad.reshape({rows, embedding_dimension});

        combo.apply_delta(scratch_2d, input_2d, input_grad_2d, weight_grad, bias_grad, accumulate);
    };

    apply_qkv_projection_grad(query_projection, query_grad, query_input,
                              query_sequence_length,
                              delta_views[InputQueryDelta][0],
                              gradient_views[QueryWeight], gradient_views[QueryBias],
                              false);

    TensorView& kv_input_grad = self_attention
        ? delta_views[InputQueryDelta][0]
        : delta_views[InputSourceDelta][0];

    apply_qkv_projection_grad(key_projection, key_grad, source_input,
                              source_sequence_length,
                              kv_input_grad,
                              gradient_views[KeyWeight], gradient_views[KeyBias],
                              self_attention);

    apply_qkv_projection_grad(value_projection, value_grad, source_input,
                              source_sequence_length,
                              kv_input_grad,
                              gradient_views[ValueWeight], gradient_views[ValueBias],
                              true);
}

void MultiHeadAttention::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "MultiHeadAttention");

    const string new_label = read_json_string(root_element, "Label");
    const Index new_query_sequence_length = read_json_index(root_element, "QuerySequenceLength");
    const Index new_source_sequence_length = read_json_index(root_element, "SourceSequenceLength");
    const Index new_embedding_dimension = read_json_index(root_element, "EmbeddingDimension");
    const Index new_heads_number = read_json_index(root_element, "HeadsNumber");
    const bool  new_use_causal_mask = read_json_bool(root_element, "CausalMask");

    set(new_query_sequence_length, new_source_sequence_length, new_embedding_dimension,
        new_heads_number, new_use_causal_mask, new_label);
}

void MultiHeadAttention::to_JSON(JsonWriter& printer) const
{
    printer.open_element("MultiHeadAttention");

    write_json(printer, {
        {"Label", label},
        {"QuerySequenceLength", to_string(query_sequence_length)},
        {"SourceSequenceLength", to_string(source_sequence_length)},
        {"EmbeddingDimension", to_string(embedding_dimension)},
        {"HeadsNumber", to_string(heads_number)},
        {"CausalMask", to_string(use_causal_mask)}
    });

    printer.close_element();
}

REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
