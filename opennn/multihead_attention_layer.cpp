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
#include "neural_network.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

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

type MultiHeadAttention::get_scaling_factor() const
{
    const Index head_dimension = get_head_dimension();

    return (head_dimension == 0)
        ? 0.25
        : type(1) / type(sqrt(head_dimension));
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

vector<Shape> MultiHeadAttention::get_parameter_shapes() const
{
    return {{embedding_dimension, embedding_dimension},
            {embedding_dimension},
            {embedding_dimension, embedding_dimension},
            {embedding_dimension},
            {embedding_dimension, embedding_dimension},
            {embedding_dimension},
            {embedding_dimension, embedding_dimension},
            {embedding_dimension}};
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

    if(new_heads_number == 0 && new_embedding_dimension == 0)
    {
        heads_number = 0;
        return;
    }

    if(new_heads_number <= 0)
        throw runtime_error("MultiHeadAttention Error: Heads number must be greater than 0.");

    if(new_embedding_dimension % new_heads_number != 0)
        throw runtime_error("MultiHeadAttention Error: The embedding dimension must be divisible by the number of heads.");

    use_causal_mask = new_use_causal_mask;

    if (use_causal_mask)
    {
        causal_mask.resize(query_sequence_length, source_sequence_length);

        for(Index row = 0; row < query_sequence_length; ++row)
            for(Index column = 0; column < source_sequence_length; ++column)
                causal_mask(row, column) = (column > row) ? NEG_INFINITY : type(0);
    }
}

void MultiHeadAttention::forward_propagate(ForwardPropagation& forward_propagation,
                                           size_t layer,
                                           bool) noexcept
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
    const Index total_rows = batch_size * query_sequence_length;

    float* transpose_scratch = forward_views[TransposeScratch][0].data;

    projection(query_input,  parameters[QueryWeight], parameters[QueryBias], query, transpose_scratch);
    projection(source_input, parameters[KeyWeight],   parameters[KeyBias],   key,   transpose_scratch);
    projection(source_input, parameters[ValueWeight], parameters[ValueBias], value, transpose_scratch);

    multiply(query, false, key, true, attention_weights, get_scaling_factor(), type(0));

    attention_masks(source_input, attention_weights, causal_mask, use_causal_mask, forward_views[PaddingMask][0].data);

    softmax(attention_weights);

    // Attention · V into [B, H, Sq, D] scratch, then transpose to [B, Sq, H, D]
    TensorView attention_out_scratch = forward_views[AttentionOutputTransposed][0].reshape(heads_shape(batch_size));
    TensorView concatenated_4d       = concatenated.reshape(concat_shape(batch_size));

    multiply(attention_weights, false, value, false, attention_out_scratch);
    merge_heads(attention_out_scratch, concatenated_4d);

    const Shape flat_shape = {total_rows, embedding_dimension};
    TensorView concatenated_2d = concatenated.reshape(flat_shape);
    TensorView output_2d       = output.reshape(flat_shape);

    combination(concatenated_2d,
                parameters[ProjectionWeight],
                parameters[ProjectionBias],
                output_2d);
}

void MultiHeadAttention::back_propagate(ForwardPropagation& forward_propagation,
                                        BackPropagation& back_propagation,
                                        size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    const TensorView& query_input = get_query_input(forward_views);
    const TensorView& source_input = get_source_input(forward_views);
    const bool self_attention = is_self_attention(forward_views);

    TensorView& output_gradient = backward_views[OutputGradient][0];

    const Index batch_size = forward_propagation.batch_size;
    const Index total_rows = batch_size * query_sequence_length;
    const Shape flat_shape = {total_rows, embedding_dimension};

    float* transpose_scratch = forward_views[TransposeScratch][0].data;
    const type scaling_factor = get_scaling_factor();

    combination_gradient(output_gradient.reshape(flat_shape),
                         forward_views[ConcatenatedAttentionOutputs][0].reshape(flat_shape),
                         parameters[ProjectionWeight],
                         backward_views[ConcatenatedOutputGradient][0].reshape(flat_shape),
                         gradient_views[ProjectionWeight],
                         gradient_views[ProjectionBias],
                         false);

    const TensorView& attention_weights = forward_views[AttentionWeights][0];
    TensorView& concat_grad     = backward_views[ConcatenatedOutputGradient][0];
    TensorView& att_weight_grad = backward_views[AttentionWeightGradient][0];
    TensorView& query_grad      = backward_views[QueryGradient][0];
    TensorView& key_grad        = backward_views[KeyGradient][0];
    TensorView& value_grad      = backward_views[ValueGradient][0];

    // Transpose concat_grad [B, Sq, H, D] -> scratch [B, H, Sq, D]
    TensorView concat_grad_4d = concat_grad.reshape(concat_shape(batch_size));
    TensorView scratch_4d     = forward_views[TransposeScratch][0].reshape(heads_shape(batch_size));

    split_heads(concat_grad_4d, scratch_4d);

    // value_grad = attention_weights^T · scratch
    multiply(attention_weights, true,  scratch_4d, false, value_grad);

    // att_weight_grad = scratch · value^T
    multiply(scratch_4d, false, forward_views[Value][0], true, att_weight_grad);

    softmax_backward(attention_weights, att_weight_grad);

    // query_grad = att_weight_grad · key  (scaling folded into alpha)
    multiply(att_weight_grad, false, forward_views[Key][0],   false, query_grad, scaling_factor, type(0));

    // key_grad = att_weight_grad^T · query
    multiply(att_weight_grad, true,  forward_views[Query][0], false, key_grad,   scaling_factor, type(0));

    projection_gradient(query_grad, query_input, parameters[QueryWeight],
                        gradient_views[QueryBias], gradient_views[QueryWeight],
                        backward_views[InputQueryGradient][0],
                        transpose_scratch, /*accumulate*/ false);

    TensorView& kv_input_grad = self_attention
        ? backward_views[InputQueryGradient][0]
        : backward_views[InputSourceGradient][0];

    projection_gradient(key_grad, source_input, parameters[KeyWeight],
                        gradient_views[KeyBias], gradient_views[KeyWeight],
                        kv_input_grad,
                        transpose_scratch, self_attention);

    projection_gradient(value_grad, source_input, parameters[ValueWeight],
                        gradient_views[ValueBias], gradient_views[ValueWeight],
                        kv_input_grad,
                        transpose_scratch, true);
}

void MultiHeadAttention::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "MultiHeadAttention");

    const string new_label = read_xml_string(root_element, "Label");
    const Index new_query_sequence_length = read_xml_index(root_element, "QuerySequenceLength");
    const Index new_source_sequence_length = read_xml_index(root_element, "SourceSequenceLength");
    const Index new_embedding_dimension = read_xml_index(root_element, "EmbeddingDimension");
    const Index new_heads_number = read_xml_index(root_element, "HeadsNumber");
    const bool  new_use_causal_mask = read_xml_bool(root_element, "CausalMask");

    set(new_query_sequence_length, new_source_sequence_length, new_embedding_dimension,
        new_heads_number, new_use_causal_mask, new_label);
}

void MultiHeadAttention::to_XML(XmlPrinter& printer) const
{
    printer.open_element("MultiHeadAttention");
    write_xml_properties(printer, {
        {"Label", label},
        {"QuerySequenceLength", to_string(query_sequence_length)},
        {"SourceSequenceLength", to_string(source_sequence_length)},
        {"EmbeddingDimension", to_string(embedding_dimension)},
        {"HeadsNumber", to_string(heads_number)},
        {"CausalMask", to_string(use_causal_mask ? 1 : 0)}
    });
    printer.close_element();
}

REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
