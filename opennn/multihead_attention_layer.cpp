//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "multihead_attention_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.


MultiheadAttentionLayer::MultiheadAttentionLayer() : Layer()
{
    set();

    layer_type = Type::MultiheadAttention;
}


/// Layer architecture constructor.
/// It creates a layer object with given input size, embedding depth and number of attention heads.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.

MultiheadAttentionLayer::MultiheadAttentionLayer(const Index& new_input_size,
                                                 const Index& new_context_size,
                                                 const Index& new_depth,
                                                 const Index& new_heads_number,
                                                 const bool& apply_causal_mask) : Layer()
{
    set(new_input_size, new_context_size, new_depth, new_heads_number);

    set_causal_mask(apply_causal_mask);

    layer_type = Type::MultiheadAttention;

    layer_name = "multihead_attention_layer";
}


/// Returns the size of the input to the layer.

Index MultiheadAttentionLayer::get_input_size() const
{
    return input_size;
}


/// Returns the size of the context to the layer.

Index MultiheadAttentionLayer::get_context_size() const
{
    return context_size;
}


/// Returns the embedding depth used in the layer.

Index MultiheadAttentionLayer::get_depth() const
{
    return depth;
}


/// Returns the number of attention heads of the layer.

Index MultiheadAttentionLayer::get_heads_number() const
{
    return heads_number;
}


/// Returns linear transformation kernels

Tensor<type, 3> MultiheadAttentionLayer::get_query_kernel() const
{
    return query_kernel;
}

Tensor<type, 3> MultiheadAttentionLayer::get_key_kernel() const
{
    return key_kernel;
}

Tensor<type, 3> MultiheadAttentionLayer::get_value_kernel() const
{
    return value_kernel;
}


/// Returns the linear projection kernel

Tensor<type, 3> MultiheadAttentionLayer::get_projection_kernel() const
{
    return projection_kernel;
}


/// Returns the number of parameters of the layer.

Index MultiheadAttentionLayer::get_parameters_number() const
{
    return query_kernel.size() + key_kernel.size() + value_kernel.size() + projection_kernel.size();
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& MultiheadAttentionLayer::get_display() const
{
    return display;
}


/// Sets an empty layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set()
{
    input_size = 0;

    depth = 0;

    heads_number = 0;

    query_kernel.resize(0, 0, 0);
    key_kernel.resize(0, 0, 0);
    value_kernel.resize(0, 0, 0);

    projection_kernel.resize(0, 0, 0);

    set_default();
}


/// Sets new input size, embedding depth, number of attention heads and activation function of the layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set(const Index& new_input_size,
                                  const Index& new_context_size,
                                  const Index& new_depth,
                                  const Index& new_heads_number)
{
    input_size = new_input_size;

    context_size = new_context_size;

    depth = new_depth;

    heads_number = new_heads_number;

    set_kernels();

    set_default();
}


/// Sets those members not related to the perceptrons to their default value.

void MultiheadAttentionLayer::set_default()
{
    layer_name = "multihead_attention_layer";

    display = true;

    layer_type = Type::MultiheadAttention;
}


void MultiheadAttentionLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new input size in the layer.

void MultiheadAttentionLayer::set_input_size(const Index& new_input_size)
{
    input_size = new_input_size;
}


/// Sets a new input size in the layer.

void MultiheadAttentionLayer::set_context_size(const Index& new_context_size)
{
    context_size = new_context_size;
}


/// Sets a new embedding depth in the layer.

void MultiheadAttentionLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_kernels();
}


/// Sets a new number of attention heads in the layer.

void MultiheadAttentionLayer::set_heads_number(const Index& new_heads_number)
{
    heads_number = new_heads_number;

    set_kernels();
}


/// Sets the layer's kernels according to the parameters.

void MultiheadAttentionLayer::set_kernels()
{
    query_kernel.resize(depth, depth, heads_number);
    key_kernel.resize(depth, depth, heads_number);
    value_kernel.resize(depth, depth, heads_number);

    projection_kernel.resize(depth, depth, heads_number);

    set_parameters_random();
}


void MultiheadAttentionLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);


    /// @todo in Tensor form

#pragma omp parallel for
    for(Index i = 0; i < query_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        query_kernel(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < key_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        key_kernel(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < value_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        value_kernel(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < projection_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        projection_kernel(i) = minimum + (maximum - minimum)*random;
    }
}


void MultiheadAttentionLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void MultiheadAttentionLayer::set_causal_mask(const bool& apply_causal_mask)
{
    if(apply_causal_mask && input_size != context_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void set_causal_mask(const bool&) method.\n"
               << "Causal mask can only be applied to self-attention. In this case, input size (" << input_size << ") should be equal to context size (" << context_size << ").";

        throw invalid_argument(buffer.str());
    }

    causal_mask = apply_causal_mask;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void MultiheadAttentionLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void MultiheadAttentionLayer::softmax(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Index batch_size = x.dimension(0);

    const Eigen::array<int, 1> sum_dimensions({{2}});

    const Eigen::array<Index, 4> reshape_dimensions({{batch_size, input_size, 1, heads_number}});

    y.device(*thread_pool_device) = x.exp();

    y.device(*thread_pool_device) = y/y.sum(sum_dimensions).reshape(reshape_dimensions);

}


void MultiheadAttentionLayer::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    const Index batch_size = attention_scores.dimension(0);

    const type m_inf = type(-1)*numeric_limits<type>::infinity();

    for(Index head_index = 0; head_index < heads_number ; head_index++)
    {
        for(Index context_index = 0; context_index < context_size; context_index++)
        {
            TensorMap<Tensor<type, 2>> masked_values(attention_scores.data() + context_index * batch_size*input_size + head_index * batch_size*input_size*context_size,
                                                     batch_size,
                                                     context_index);

            masked_values.setConstant(m_inf);
        }
    }
}


/// @todo explain

void MultiheadAttentionLayer::calculate_transformation(const Tensor<type, 3>& data,
                                                       Tensor<type, 4>& transformed_data,
                                                       const Tensor<type, 3>& kernel)
{
    const Index batch_size = data.dimension(0);

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        transformed_data.chip(sample_index, 0).device(*thread_pool_device)
            = data.chip(sample_index, 0).contract(kernel, A_B);
    }
}


void MultiheadAttentionLayer::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
                                                          Tensor<type, 3>& outputs)
{
    const Index batch_size = outputs.dimension(0);

    const Eigen::array<IndexPair<Index>, 2> contraction_indices = {IndexPair<Index>(1, 0), IndexPair<Index>(2, 2)};

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        outputs.chip(sample_index, 0).device(*thread_pool_device)
            = attention_outputs.chip(sample_index, 0).contract(projection_kernel, contraction_indices);
    }
}


/// Computes the attention scores by comparing (via dot product) query and key.
/// Attention scores must be computed separately for each batch element and each attention head (batch matrix multiplication).

void MultiheadAttentionLayer::compute_attention_scores(const Tensor<type, 4>& transformed_query,
                                                       const Tensor<type, 4>& transformed_key,
                                                       Tensor<type, 4>& attention_scores)
{
    const Index batch_size = transformed_query.dimension(0);

    /// @todo do not assign memory

    const Tensor<type, 4> scaled_query = transformed_query /type(sqrt(depth));

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        for(Index head_index = 0; head_index < heads_number ; head_index++)
        {
            attention_scores.chip(sample_index, 0).chip(head_index, 2).device(*thread_pool_device) =
                scaled_query.chip(sample_index, 0).chip(head_index, 2)
                .contract(transformed_key.chip(sample_index, 0).chip(head_index, 2), A_BT);
        }
    }

    if(causal_mask)
    {
        apply_causal_mask(attention_scores);
    }

    softmax(attention_scores, attention_scores);

    /// @todo add dropout
}


void MultiheadAttentionLayer::compute_attention_outputs(const Tensor<type, 4>& transformed_value,
                                                       const Tensor<type, 4>& attention_scores,
                                                       Tensor<type, 4>& attention_outputs)
{    
    const Index batch_size = transformed_value.dimension(0);

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        for(Index head_index = 0; head_index < heads_number ; head_index++)
        {
            attention_outputs.chip(sample_index, 0).chip(head_index, 2).device(*thread_pool_device) =
                attention_scores.chip(sample_index, 0).chip(head_index, 2).contract(
                transformed_value.chip(sample_index, 0).chip(head_index, 2), A_B);
        }
    }
}


void MultiheadAttentionLayer::forward_propagate(const pair<type*, dimensions>& inputs,
                                                LayerForwardPropagation* layer_forward_propagation,
                                                const bool& is_training)
{
    if(inputs.second.size() != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "Number of input tensors (" << inputs.second.size() << ") must be 2 (input and context).\n";

        throw invalid_argument(buffer.str());
    }

    if(inputs.second[0][0] != inputs.second[1][0])
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "Batch sizes of input and context must be equal.\n";

        throw invalid_argument(buffer.str());
    }

    if(inputs.second[0][1] != input_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "2nd dimension of input must be equal to layer input_size.\n";

        throw invalid_argument(buffer.str());
    }

    if(inputs.second[1][1] != context_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "2nd dimension of context must be equal to layer context_size.\n";
        throw invalid_argument(buffer.str());
    }

    if(inputs.second[0][2] != depth || inputs.second[1][2] != depth)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "3rd dimension of input and context must be equal to layer depth.\n";
        throw invalid_argument(buffer.str());
    }

    MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation
        = static_cast<MultiheadAttentionLayerForwardPropagation*>(layer_forward_propagation);

    const TensorMap<Tensor<type, 3>> query_map(inputs.first,
                                               inputs.second[0][0],
                                               inputs.second[0][1],
                                               inputs.second[0][2]);

    const TensorMap<Tensor<type, 3>> key_map(inputs.first + inputs.second[0][0] + inputs.second[0][1] + inputs.second[0][2],
                                             inputs.second[1][0],
                                             inputs.second[1][1],
                                             inputs.second[1][2]);

    const TensorMap<Tensor<type, 3>> value_map = key_map;

    Tensor<type, 4>& transformed_query = multihead_attention_layer_forward_propagation->transformed_query;
    Tensor<type, 4>& transformed_key = multihead_attention_layer_forward_propagation->transformed_key;
    Tensor<type, 4>& transformed_value = multihead_attention_layer_forward_propagation->transformed_value;

    calculate_transformation(query_map, transformed_query, query_kernel);

    calculate_transformation(key_map, transformed_key, key_kernel);

    calculate_transformation(value_map, transformed_value, value_kernel);

    Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;

    compute_attention_scores(transformed_query,
                             transformed_key,
                             attention_scores);

    Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    compute_attention_outputs(transformed_value,
                             attention_scores,
                             attention_outputs);


    Tensor<type, 3>& outputs = multihead_attention_layer_forward_propagation->outputs;

    calculate_output_projection(attention_outputs,
                                outputs);

}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
