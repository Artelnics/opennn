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

void MultiHeadAttention::set_parameters_random()
{
    if(embedding_dimension == 0) return;

    const type weight_limit = sqrt(type(6) / type(2 * embedding_dimension));

    const int weight_slots[] = {QueryWeight, KeyWeight, ValueWeight, ProjectionWeight};
    for(const int slot : weight_slots)
    {
        if(parameters[slot].empty()) continue;
        set_random_uniform(VectorMap(parameters[slot].as<float>(), parameters[slot].size()),
                           -weight_limit, weight_limit);
    }

    for(const int slot : {QueryBias, KeyBias, ValueBias, ProjectionBias})
        if(!parameters[slot].empty()) parameters[slot].fill(0.0f);
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

#ifdef OPENNN_WITH_CUDA

void MultiHeadAttention::init_cuda(Index batch_size)
{
    if(dropout_rate <= type(0)) return;
    if(heads_number == 0 || embedding_dimension == 0) return;

    if(dropout_arguments.descriptor)    { cudnnDestroyDropoutDescriptor(dropout_arguments.descriptor); dropout_arguments.descriptor = nullptr; }
    if(dropout_arguments.states)        { cudaFree(dropout_arguments.states);        dropout_arguments.states = nullptr; }
    if(dropout_arguments.reserve_space) { cudaFree(dropout_arguments.reserve_space); dropout_arguments.reserve_space = nullptr; }

    cudnnTensorDescriptor_t temp_desc = nullptr;
    cudnnCreateTensorDescriptor(&temp_desc);
    cudnnSetTensor4dDescriptor(temp_desc, CUDNN_TENSOR_NHWC, activation_dtype,
                               static_cast<int>(batch_size),
                               static_cast<int>(source_sequence_length),
                               static_cast<int>(heads_number),
                               static_cast<int>(query_sequence_length));

    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_arguments.descriptor));
    CHECK_CUDNN(cudnnDropoutGetStatesSize(Device::get_cudnn_handle(), &dropout_arguments.states_size));
    CHECK_CUDA(cudaMalloc(&dropout_arguments.states, dropout_arguments.states_size));
    CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_arguments.descriptor, Device::get_cudnn_handle(),
                                          static_cast<float>(dropout_rate),
                                          dropout_arguments.states, dropout_arguments.states_size,
                                          static_cast<unsigned long long>(random_integer(0, 1 << 30))));
    CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(temp_desc, &dropout_arguments.reserve_size));
    CHECK_CUDA(cudaMalloc(&dropout_arguments.reserve_space, dropout_arguments.reserve_size));

    dropout_arguments.rate = dropout_rate;

    cudnnDestroyTensorDescriptor(temp_desc);
}

#endif

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
    const Index total_rows = batch_size * query_sequence_length;

    float* transpose_scratch = forward_views[TransposeScratch][0].as<float>();

    projection(query_input,  parameters[QueryWeight], parameters[QueryBias], query, transpose_scratch);
    projection(source_input, parameters[KeyWeight],   parameters[KeyBias],   key,   transpose_scratch);
    projection(source_input, parameters[ValueWeight], parameters[ValueBias], value, transpose_scratch);

    multiply(query, false, key, true, attention_weights, get_scaling_factor(), type(0));

    attention_masks(source_input, attention_weights, causal_mask, use_causal_mask, forward_views[PaddingMask][0].as<float>());

    softmax(attention_weights);

    const bool apply_dropout = is_training && dropout_rate > type(0);
    TensorView& attention_used = apply_dropout
        ? forward_views[AttentionWeightsDropped][0]
        : attention_weights;

    if(apply_dropout)
    {
        copy(attention_weights, attention_used);
        dropout(attention_used, dropout_arguments);
    }

    TensorView attention_out_scratch = forward_views[AttentionOutputTransposed][0].reshape(heads_shape(batch_size));
    TensorView concatenated_4d       = concatenated.reshape(concat_shape(batch_size));

    multiply(attention_used, false, value, false, attention_out_scratch);
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
    auto& delta_views = back_propagation.delta_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    const TensorView& query_input = get_query_input(forward_views);
    const TensorView& source_input = get_source_input(forward_views);
    const bool self_attention = is_self_attention(forward_views);

    TensorView& output_delta = delta_views[OutputDelta][0];

    const Index batch_size = forward_propagation.batch_size;
    const Index total_rows = batch_size * query_sequence_length;
    const Shape flat_shape = {total_rows, embedding_dimension};

    float* transpose_scratch = forward_views[TransposeScratch][0].as<float>();
    const type scaling_factor = get_scaling_factor();

    TensorView concat_grad_flat = delta_views[ConcatenatedOutputDelta][0].reshape(flat_shape);

    combination_gradient(output_delta.reshape(flat_shape),
                         forward_views[ConcatenatedAttentionOutputs][0].reshape(flat_shape),
                         parameters[ProjectionWeight],
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

    const bool dropout_active = dropout_rate > type(0);
    const TensorView& attention_forward_output = dropout_active
        ? forward_views[AttentionWeightsDropped][0]
        : attention_weights;

    multiply(attention_forward_output, true, scratch_4d, false, value_grad);

    multiply(scratch_4d, false, forward_views[Value][0], true, att_weight_grad);

    if(dropout_active)
        dropout_delta(att_weight_grad, att_weight_grad, dropout_arguments);

    softmax_backward(attention_weights, att_weight_grad);

    multiply(att_weight_grad, false, forward_views[Key][0],   false, query_grad, scaling_factor, type(0));

    multiply(att_weight_grad, true,  forward_views[Query][0], false, key_grad,   scaling_factor, type(0));

    projection_gradient(query_grad, query_input, parameters[QueryWeight],
                        gradient_views[QueryBias], gradient_views[QueryWeight],
                        delta_views[InputQueryDelta][0],
                        transpose_scratch, /*accumulate*/ false);

    TensorView& kv_input_grad = self_attention
        ? delta_views[InputQueryDelta][0]
        : delta_views[InputSourceDelta][0];

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
    write_xml(printer, {
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
