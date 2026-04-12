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

namespace opennn
{

MultiHeadAttention::MultiHeadAttention(const Shape& new_input_shape,
                                       Index new_heads_number,
                                       const string& new_name) : Layer()
{
    // Self-attention

    set(new_input_shape[0],    // query_sequence_length
        new_input_shape[0],    // source_sequence_length
        new_input_shape[1],    // embedding_dimension
        new_heads_number,
        false,
        new_name);
}

MultiHeadAttention::MultiHeadAttention(const Shape& new_query_dimensions,
                                       const Shape& new_source_dimensions,
                                       Index new_heads_number,
                                       const string& new_name) : Layer()
{
    if (new_query_dimensions[1] != new_source_dimensions[1])
        throw runtime_error("MultiHeadAttention Error: embedding dimension must be the same for query and source.");

    // Cross-attention

    set(new_query_dimensions[0],    // query_sequence_length
        new_source_dimensions[0],   // source_sequence_length
        new_query_dimensions[1],    // embedding_dimension
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
        : Index(get_embedding_dimension() / heads_number);
}

Shape MultiHeadAttention::get_input_shape() const
{
    return { query_sequence_length, get_embedding_dimension() };
}

Shape MultiHeadAttention::get_output_shape() const
{
    return { query_sequence_length, get_embedding_dimension() };
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

void MultiHeadAttention::set(const Index new_query_sequence_length,
                             Index new_source_sequence_length,
                             Index new_embedding_dimension,
                             Index new_heads_number,
                             bool new_use_causal_mask,
                             const string& new_label)
{
    name = "MultiHeadAttention";
    query_sequence_length = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    embedding_dimension = new_embedding_dimension;
    input_shape = {new_query_sequence_length, new_embedding_dimension};
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
                causal_mask(row, column) = (column > row) ? minus_inf : type(0);
    }
}

void MultiHeadAttention::forward_propagate(ForwardPropagation& forward_propagation,
                                           size_t layer,
                                           bool)
{
    const TensorView& query_input = forward_propagation.views[layer][Inputs][0];

    const TensorView& source_input = (forward_propagation.views[layer][Inputs].size() == 1)
                                        ? query_input
                                        : forward_propagation.views[layer][Inputs][1];

    TensorView& query = forward_propagation.views[layer][Query][0];
    TensorView& key = forward_propagation.views[layer][Key][0];
    TensorView& value = forward_propagation.views[layer][Value][0];
    TensorView& attention_weights_view = forward_propagation.views[layer][AttentionWeights][0];
    TensorView& concatenated = forward_propagation.views[layer][ConcatenatedAttentionOutputs][0];
    TensorView& output = forward_propagation.views[layer].back()[0];

    projection(query_input, parameters[QueryWeights], parameters[QueryBiases], query);
    projection(source_input, parameters[KeyWeights], parameters[KeyBiases], key);
    projection(source_input, parameters[ValueWeights], parameters[ValueBiases], value);

    multihead_attention_forward(
        query, key, value,
        attention_weights_view, concatenated, output,
        parameters[ProjectionWeights], parameters[ProjectionBiases],
        source_input,
        forward_propagation.batch_size, heads_number,
        query_sequence_length, source_sequence_length,
        get_embedding_dimension(), get_head_dimension(),
        get_scaling_factor(), use_causal_mask, causal_mask);


#ifdef CUDA
    // @todo CUDA path
#endif
}

void MultiHeadAttention::back_propagate(ForwardPropagation& forward_propagation,
                                        BackPropagation& back_propagation,
                                        size_t layer) const
{
    const TensorView& query_input = forward_propagation.views[layer][Inputs][0];
    const TensorView& source_input = (forward_propagation.views[layer][Inputs].size() == 1)
                                         ? query_input
                                         : forward_propagation.views[layer][Inputs][1];
    const bool self_attention = (forward_propagation.views[layer][Inputs].size() == 1);

    TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradient][0];

    multihead_attention_backward(
        query_input, source_input, output_gradient,
        forward_propagation.views[layer][Query][0],
        forward_propagation.views[layer][Key][0],
        forward_propagation.views[layer][Value][0],
        forward_propagation.views[layer][AttentionWeights][0],
        forward_propagation.views[layer][ConcatenatedAttentionOutputs][0],
        parameters[ProjectionWeights],
        back_propagation.gradient_views[layer][ProjectionWeights],
        back_propagation.gradient_views[layer][ProjectionBiases],
        back_propagation.backward_views[layer][ConcatenatedOutputGradient][0],
        back_propagation.backward_views[layer][AttentionWeightGradient][0],
        back_propagation.backward_views[layer][QueryGradient][0],
        back_propagation.backward_views[layer][KeyGradient][0],
        back_propagation.backward_views[layer][ValueGradient][0],
        back_propagation.gradient_views[layer][QueryWeights],
        back_propagation.gradient_views[layer][QueryBiases],
        back_propagation.gradient_views[layer][KeyWeights],
        back_propagation.gradient_views[layer][KeyBiases],
        back_propagation.gradient_views[layer][ValueWeights],
        back_propagation.gradient_views[layer][ValueBiases],
        back_propagation.backward_views[layer][InputQueryGradient][0],
        parameters[QueryWeights], parameters[KeyWeights], parameters[ValueWeights],
        forward_propagation.batch_size, heads_number,
        query_sequence_length, source_sequence_length,
        get_embedding_dimension(), get_head_dimension(),
        get_scaling_factor(), self_attention);

#ifdef CUDA
    // @todo CUDA path
#endif
}

void MultiHeadAttention::apply_causal_mask(Tensor4& attention_scores) const
{
    const Index batch_size = attention_scores.dimension(0);
    const Index query_sequence_length = attention_scores.dimension(2);
    const Index source_sequence_length = attention_scores.dimension(3);

    const Index matrix_size = query_sequence_length * source_sequence_length;

    const Index total_matrices = batch_size * heads_number;

    MatrixMap scores(attention_scores.data(), total_matrices, matrix_size);

    const VectorMap causal_mask_map(const_cast<type*>(causal_mask.data()), matrix_size);

    scores.rowwise() += causal_mask_map.transpose();
}

void MultiHeadAttention::apply_key_padding_mask(const TensorMap3& source_input,
                                                Tensor4& attention_weights) const
{
    const Index batch_size = attention_weights.dimension(0);
    const Index query_sequence_length = attention_weights.dimension(2);
    const Index source_sequence_length = attention_weights.dimension(3);
    const Index embedding_dimension = source_input.dimension(2);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index s = 0; s < source_sequence_length; ++s)
        {
            const type* row_ptr = &source_input(b, s, 0);
            const bool is_pad = Eigen::Map<const VectorR>(row_ptr, embedding_dimension)
                                    .cwiseAbs().maxCoeff() <= padding_threshold;

            if(is_pad)
            {
                const Index slice_size = heads_number * query_sequence_length;
                MatrixMap att_map(attention_weights.data() + b * slice_size * source_sequence_length,
                                  slice_size, source_sequence_length);
                att_map.col(s).setConstant(mask_value);
            }
        }
    }
}

void MultiHeadAttention::to_XML(XmlPrinter& printer) const
{
    printer.open_element("MultiHeadAttention");
    write_xml_properties(printer, {
        {"Label", label},
        {"InputSize", to_string(get_query_sequence_length())},
        {"ContextSize", to_string(get_source_sequence_length())},
        {"Depth", to_string(get_embedding_dimension())},
        {"HeadDimension", to_string(get_head_dimension())},
        {"HeadsNumber", to_string(get_heads_number())},
        {"CausalMask", to_string(use_causal_mask ? 1 : 0)}
    });
    printer.close_element();
}

void MultiHeadAttention::from_XML(const XmlDocument& document)
{
    // @todo update notation

    const XmlElement* multihead_attention_layer_element = get_xml_root(document, "MultiHeadAttention");

    const string new_label = read_xml_string(multihead_attention_layer_element, "Label");
    const Index new_input_size = read_xml_index(multihead_attention_layer_element, "InputSize");
    const Index new_context_size = read_xml_index(multihead_attention_layer_element, "ContextSize");
    const Index new_depth = read_xml_index(multihead_attention_layer_element, "Depth");
    const Index new_heads_number = read_xml_index(multihead_attention_layer_element, "HeadsNumber");
    const Index new_use_causal_mask = read_xml_bool(multihead_attention_layer_element, "CausalMask");

    set(new_input_size, new_context_size, new_depth, new_heads_number, new_use_causal_mask, new_label);
}

REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
