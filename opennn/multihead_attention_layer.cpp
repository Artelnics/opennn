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
                causal_mask(row, column) = (column > row) ? minus_inf : type(0);
    }
}

void MultiHeadAttention::forward_propagate(ForwardPropagation& forward_propagation,
                                           size_t layer,
                                           bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& query_input = forward_views[Input][0];

    const TensorView& source_input = (forward_views[Input].size() == 1)
                                        ? query_input
                                        : forward_views[Input][1];

    TensorView& query = forward_views[Query][0];
    TensorView& key = forward_views[Key][0];
    TensorView& value = forward_views[Value][0];
    TensorView& attention_weights_view = forward_views[AttentionWeights][0];
    TensorView& concatenated = forward_views[ConcatenatedAttentionOutputs][0];
    TensorView& output = forward_views.back()[0];

    MultiheadAttentionArguments args;
    args.batch_size = forward_propagation.batch_size;
    args.heads_number = heads_number;
    args.query_sequence_length = query_sequence_length;
    args.source_sequence_length = source_sequence_length;
    args.embedding_dimension = get_embedding_dimension();
    args.head_dimension = get_head_dimension();
    args.scaling_factor = get_scaling_factor();
    args.use_causal_mask = use_causal_mask;
    args.causal_mask = &causal_mask;
    args.padding_mask = forward_views[PaddingMask][0].data;
    args.transpose_scratch = forward_views[TransposeScratch][0].data;
    args.attention_output_transposed = forward_views[AttentionOutputTransposed][0].data;

    projection(query_input, 
               parameters[QueryWeight], 
               parameters[QueryBias], 
               query, 
               args);

    projection(source_input, parameters[KeyWeight], parameters[KeyBias], key, args);

    projection(source_input, parameters[ValueWeight], parameters[ValueBias], value, args);

    multihead_attention_forward(
        query, key, value,
        attention_weights_view, concatenated, output,
        parameters[ProjectionWeight], parameters[ProjectionBias],
        source_input, args);
}

void MultiHeadAttention::back_propagate(ForwardPropagation& forward_propagation,
                                        BackPropagation& back_propagation,
                                        size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    const TensorView& query_input = forward_views[Input][0];
    const TensorView& source_input = (forward_views[Input].size() == 1)
                                         ? query_input
                                         : forward_views[Input][1];
    const bool self_attention = (forward_views[Input].size() == 1);

    TensorView& output_gradient = backward_views[OutputGradient][0];

    MultiheadAttentionArguments args;
    args.batch_size = forward_propagation.batch_size;
    args.heads_number = heads_number;
    args.query_sequence_length = query_sequence_length;
    args.source_sequence_length = source_sequence_length;
    args.embedding_dimension = get_embedding_dimension();
    args.head_dimension = get_head_dimension();
    args.scaling_factor = get_scaling_factor();
    args.use_causal_mask = false;
    args.causal_mask = nullptr;
    args.transpose_scratch = forward_views[TransposeScratch][0].data;
    args.softmax_gradient = backward_views[SoftmaxGradient][0].data;
    args.query_input_gradient_scratch = backward_views[QueryInputGradientScratch][0].data;
    args.source_input_gradient_scratch = backward_views[SourceInputGradientScratch][0].data;

    multihead_attention_backward(
        query_input, source_input, output_gradient,
        forward_views[Query][0],
        forward_views[Key][0],
        forward_views[Value][0],
        forward_views[AttentionWeights][0],
        forward_views[ConcatenatedAttentionOutputs][0],
        parameters[ProjectionWeight],
        gradient_views[ProjectionWeight],
        gradient_views[ProjectionBias],
        backward_views[ConcatenatedOutputGradient][0],
        backward_views[AttentionWeightGradient][0],
        backward_views[QueryGradient][0],
        backward_views[KeyGradient][0],
        backward_views[ValueGradient][0],
        gradient_views[QueryWeight],
        gradient_views[QueryBias],
        gradient_views[KeyWeight],
        gradient_views[KeyBias],
        gradient_views[ValueWeight],
        gradient_views[ValueBias],
        backward_views[InputQueryGradient][0],
        backward_views[InputSourceGradient][0],
        parameters[QueryWeight], parameters[KeyWeight], parameters[ValueWeight],
        args, self_attention);
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
