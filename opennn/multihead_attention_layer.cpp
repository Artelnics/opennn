//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "multihead_attention_layer.h"
#include "perceptron_layer_3d.h"

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

Index MultiheadAttentionLayer::get_weights_depth() const
{
    return weights_depth;
}


/// Returns linear transformation weights

Tensor<type, 3> MultiheadAttentionLayer::get_query_weights() const
{
    return query_weights;
}

Tensor<type, 3> MultiheadAttentionLayer::get_key_weights() const
{
    return key_weights;
}

Tensor<type, 3> MultiheadAttentionLayer::get_value_weights() const
{
    return value_weights;
}


/// Returns the linear projection weights

Tensor<type, 3> MultiheadAttentionLayer::get_projection_weights() const
{
    return projection_weights;
}

Tensor<type, 1> MultiheadAttentionLayer::get_projection_biases() const
{
    return projection_biases;
}


/// Returns the number of parameters of the layer.

Index MultiheadAttentionLayer::get_parameters_number() const
{
    return query_weights.size() + key_weights.size() + value_weights.size() + projection_weights.size() + projection_biases.size();
}

Tensor<type, 1> MultiheadAttentionLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());
    /*
        memcpy(parameters.data(),
               synaptic_weights.data(), size_t(synaptic_weights.size())*sizeof(type));

        memcpy(parameters.data() + synaptic_weights.size(),
               biases.data(), size_t(biases.size())*sizeof(type));
    */

    Index parameters_index = 0;

    copy(execution::par,
        query_weights.data(),
        query_weights.data() + query_weights.size(),
        parameters.data());

    parameters_index += query_weights.size();

    copy(execution::par,
        key_weights.data(),
        key_weights.data() + key_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += key_weights.size();

    copy(execution::par,
        value_weights.data(),
        value_weights.data() + value_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += value_weights.size();

    copy(execution::par,
        projection_weights.data(),
        projection_weights.data() + projection_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += projection_weights.size();

    copy(execution::par,
        projection_biases.data(),
        projection_biases.data() + projection_biases.size(),
        parameters.data() + parameters_index);

    return parameters;
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

    query_weights.resize(0, 0, 0);
    key_weights.resize(0, 0, 0);
    value_weights.resize(0, 0, 0);

    projection_weights.resize(0, 0, 0);
    projection_biases.resize(0);

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

    weights_depth = Index(depth / heads_number);

    set_weights();

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

    set_weights();
}


/// Sets a new number of attention heads in the layer.

void MultiheadAttentionLayer::set_heads_number(const Index& new_heads_number)
{
    heads_number = new_heads_number;

    set_weights();
}


/// Sets the layer's weights according to the parameters.

void MultiheadAttentionLayer::set_weights()
{
    query_weights.resize(depth, weights_depth, heads_number);
    key_weights.resize(depth, weights_depth, heads_number);
    value_weights.resize(depth, weights_depth, heads_number);

    projection_weights.resize(weights_depth, depth, heads_number);
    projection_biases.resize(depth);

    set_parameters_random();
}


void MultiheadAttentionLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);


    /// @todo in Tensor form

#pragma omp parallel for
    for(Index i = 0; i < query_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        query_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < key_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        key_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < value_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        value_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < projection_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        projection_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for (Index i = 0; i < projection_biases.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        projection_biases(i) = minimum + (maximum - minimum) * random;
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

        throw runtime_error(buffer.str());
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


void MultiheadAttentionLayer::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    const Index batch_size = attention_scores.dimension(0);

    constexpr type m_inf = -numeric_limits<type>::infinity();

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
                                                       const Tensor<type, 3>& weights) const
{
    const Index batch_size = data.dimension(0);

    // This is NOT batch matrix multiplication

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        transformed_data.chip(sample_index, 0).device(*thread_pool_device)
            = data.chip(sample_index, 0).contract(weights, A_B);
    }
}


void MultiheadAttentionLayer::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
                                                          Tensor<type, 3>& outputs) const 
{
    const Index batch_size = outputs.dimension(0);

    Tensor<type, 4> projection_outputs(batch_size, input_size, depth, heads_number);

    /// batch_matrix_multiplication(*thread_pool_device, attention_outputs, projection_weights, projection_outputs);

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        for(Index head_index = 0; head_index < heads_number; head_index++)
        {
            projection_outputs.chip(sample_index, 0).chip(head_index, 2).device(*thread_pool_device) =
                attention_outputs.chip(sample_index, 0).chip(head_index, 2)
                .contract(projection_weights.chip(head_index, 2), A_B);
        }
    }

    outputs = projection_outputs.sum(Eigen::array<Index, 1>({ 3 }));
}


/// Computes the attention scores by comparing (via dot product) query and key.
/// Attention scores must be computed separately for each batch element and each attention head 
/// (batch matrix multiplication).

void MultiheadAttentionLayer::compute_attention_scores(const Tensor<type, 4>& transformed_query,
                                                       const Tensor<type, 4>& transformed_key,
                                                       Tensor<type, 4>& attention_scores,
                                                       Tensor<type, 4>& softmax_attention_scores) const
{
    const Index batch_size = transformed_query.dimension(0);

    /// @todo do not assign memory

    const Tensor<type, 4> scaled_query = transformed_query /type(sqrt(weights_depth));

    /// batch_matrix_multiplication(*thread_pool_device, scaled_query, transformed_key, attention_scores);

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

    softmax(attention_scores, softmax_attention_scores);
}


void MultiheadAttentionLayer::compute_attention_outputs(const Tensor<type, 4>& transformed_value,
                                                       const Tensor<type, 4>& softmax_attention_scores,
                                                       Tensor<type, 4>& attention_outputs) const 
{    
    const Index batch_size = transformed_value.dimension(0);

    /// batch_matrix_multiplication(*thread_pool_device, softmax_attention_scores, transformed_value, attention_outputs)

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        for(Index head_index = 0; head_index < heads_number ; head_index++)
        {
            attention_outputs.chip(sample_index, 0).chip(head_index, 2).device(*thread_pool_device) =
                softmax_attention_scores.chip(sample_index, 0).chip(head_index, 2).contract(
                transformed_value.chip(sample_index, 0).chip(head_index, 2), A_B);
        }
    }
}


void MultiheadAttentionLayer::dropout(Tensor<type, 4>& attention_scores) const
{/*
    const Index batch_samples_number = attention_scores.dimension(0);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        for(Index )
        {
            TensorMap<Tensor<type, 2>> matrix(attention_scores.data() + neuron_index * batch_samples_number * inputs_number,
                batch_samples_number, inputs_number);

            random = calculate_random_uniform((type)0, (type)1);

            random < dropout_rate ? matrix.setZero()
                : matrix = matrix * scaling_factor;
        }
    }
*/}


void MultiheadAttentionLayer::forward_propagate(const pair<type*, dimensions>& inputs_pair,
                                                LayerForwardPropagation* layer_forward_propagation,
                                                const bool& is_training)
{
    MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation
        = static_cast<MultiheadAttentionLayerForwardPropagation*>(layer_forward_propagation);

    const TensorMap<Tensor<type, 3>> input(inputs_pair.first,
                                           inputs_pair.second[0][0],
                                           inputs_pair.second[0][1],
                                           inputs_pair.second[0][2]);

    const TensorMap<Tensor<type, 3>> context(inputs_pair.first + inputs_pair.second[0][0] + inputs_pair.second[0][1] + inputs_pair.second[0][2],
                                             inputs_pair.second[1][0],
                                             inputs_pair.second[1][1],
                                             inputs_pair.second[1][2]);

    Tensor<type, 4>& query = multihead_attention_layer_forward_propagation->query;
    Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;

    calculate_transformation(input, query, query_weights);

    calculate_transformation(context, key, key_weights);

    calculate_transformation(context, value, value_weights);

    Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;
    Tensor<type, 4>& softmax_attention_scores = multihead_attention_layer_forward_propagation->softmax_attention_scores;

    compute_attention_scores(query,
                             key,
                             attention_scores,
                             softmax_attention_scores);

    if (dropout_rate > type(0))
    {
        dropout(softmax_attention_scores);
    }

    Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    compute_attention_outputs(value,
                             softmax_attention_scores,
                             attention_outputs);


    Tensor<type, 3>& outputs = multihead_attention_layer_forward_propagation->outputs;

    calculate_output_projection(attention_outputs,
                                outputs);

}


void MultiheadAttentionLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                                     LayerBackPropagation* next_back_propagation,
                                                     LayerBackPropagation* back_propagation) const
{
    MultiheadAttentionLayerBackPropagation* multihead_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    switch (next_back_propagation->layer->get_type())
    {

    case Type::Perceptron3D:
    {
        PerceptronLayer3DForwardPropagation* next_perceptron_layer_forward_propagation =
            reinterpret_cast<PerceptronLayer3DForwardPropagation*>(next_forward_propagation);

        PerceptronLayer3DBackPropagation* next_multihead_attention_layer_back_propagation =
            reinterpret_cast<PerceptronLayer3DBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_perceptron_layer_forward_propagation,
                               next_multihead_attention_layer_back_propagation,
                               multihead_layer_back_propagation);
    }
    return;

    case Type::MultiheadAttention:
    {
        MultiheadAttentionLayerForwardPropagation* next_multihead_attention_layer_forward_propagation =
            reinterpret_cast<MultiheadAttentionLayerForwardPropagation*>(next_forward_propagation);

        MultiheadAttentionLayerBackPropagation* next_multihead_attention_layer_back_propagation =
            reinterpret_cast<MultiheadAttentionLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_multihead_attention_layer_forward_propagation,
                               next_multihead_attention_layer_back_propagation,
                               multihead_layer_back_propagation);
    }
    return;

    default:

        return;
    }
}


void MultiheadAttentionLayer::calculate_hidden_delta(PerceptronLayer3DForwardPropagation* next_forward_propagation,
                                                     PerceptronLayer3DBackPropagation* next_back_propagation,
                                                     MultiheadAttentionLayerBackPropagation* back_propagation) const
{
    // Next layer

    const PerceptronLayer3D* next_perceptron_layer = static_cast<PerceptronLayer3D*>(next_back_propagation->layer);

    const Tensor<type, 2>& next_synaptic_weights = next_perceptron_layer->get_synaptic_weights();

    // Next back propagation

    const Tensor<type, 3>& next_error_combinations_derivatives = next_back_propagation->error_combinations_derivatives;

    // This back propagation

    Tensor<type, 3>& deltas = back_propagation->deltas;

    const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 1) };

    deltas.device(*thread_pool_device) = next_error_combinations_derivatives.contract(next_synaptic_weights, contraction_indices);

}


void MultiheadAttentionLayer::calculate_hidden_delta(MultiheadAttentionLayerForwardPropagation* next_forward_propagation,
                                                     MultiheadAttentionLayerBackPropagation* next_back_propagation,
                                                     MultiheadAttentionLayerBackPropagation* back_propagation) const
{
    // Next layer

    const MultiheadAttentionLayer* next_multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(next_back_propagation->layer);

}


void MultiheadAttentionLayer::calculate_error_gradient(const pair<type*, dimensions>& inputs,
                                                       LayerForwardPropagation* forward_propagation,
                                                       LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input(inputs.first,
                                               inputs.second[0][0],
                                               inputs.second[0][1],
                                               inputs.second[0][2]);

    const TensorMap<Tensor<type, 3>> context(inputs.first + inputs.second[0][0] + inputs.second[0][1] + inputs.second[0][2],
                                             inputs.second[1][0],
                                             inputs.second[1][1],
                                             inputs.second[1][2]);

    Index batch_samples_number = inputs.second[0][0];

    // Forward propagation

    const MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation =
        static_cast<MultiheadAttentionLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;
    const Tensor<type, 4>& softmax_attention_scores = multihead_attention_layer_forward_propagation->softmax_attention_scores;
    const Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    const Tensor<type, 4>& query = multihead_attention_layer_forward_propagation-> query;
    const Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    const Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;

    // Back propagation

    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    const Tensor<type, 3>& deltas = multihead_attention_layer_back_propagation->deltas;

    Tensor<type, 4>& error_attention_scores_derivatives = multihead_attention_layer_back_propagation->error_attention_scores_derivatives;

    Tensor<type, 4>& error_attention_output_derivatives = multihead_attention_layer_back_propagation->error_attention_output_derivatives;

    Tensor<type, 4>& error_query_derivatives = multihead_attention_layer_back_propagation->error_query_derivatives;
    Tensor<type, 4>& error_key_derivatives = multihead_attention_layer_back_propagation->error_key_derivatives;
    Tensor<type, 4>& error_value_derivatives = multihead_attention_layer_back_propagation->error_value_derivatives;

    Tensor<type, 3>& error_input_derivatives = multihead_attention_layer_back_propagation->error_input_derivatives;
    Tensor<type, 3>& error_context_derivatives = multihead_attention_layer_back_propagation->error_context_derivatives;


    Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;
    Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;
    Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;

    Tensor<type, 3>& projection_weights_derivatives = multihead_attention_layer_back_propagation->projection_weights_derivatives;
    Tensor<type, 1>& projection_biases_derivatives = multihead_attention_layer_back_propagation->projection_biases_derivatives;

    /// PROJECTION DERIVATIVES

    //calculate_error_projection_weights_derivatives() // using attention_outputs and deltas

    const Eigen::array<IndexPair<Index>, 2> projection_weights_derivatives_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
    
    /// This is NOT batch matrix multiplication

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        projection_weights_derivatives.chip(head_index, 2).device(*thread_pool_device) =
            attention_outputs.chip(head_index, 3).contract(deltas, projection_weights_derivatives_contraction_indices);
    }

    //calculate_error_projection_biases_derivatives() // using deltas

    projection_biases_derivatives.device(*thread_pool_device) = deltas.sum(Eigen::array<Index, 2>({ 0, 1 }));


    /// VALUE DERIVATIVES

    //calculate_error_value_derivatives() // using softmax_attention_scores, deltas, projection_weights and value_weights (sum)

    const Eigen::array<IndexPair<Index>, 1> attention_output_derivatives_contraction_indices = { IndexPair<Index>(2, 1) };

    /// This is NOT batch matrix multiplication

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        error_attention_output_derivatives.chip(head_index, 2).device(*thread_pool_device) =
            deltas.contract(projection_weights.chip(head_index, 2), attention_output_derivatives_contraction_indices);
    }

    const Eigen::array<IndexPair<Index>, 1> value_derivatives_contraction_indices = { IndexPair<Index>(2, 1) };

    /// batch_matrix_multiplication(*thread_pool_device, softmax_attention_scores, error_attention_output_derivatives, error_value_derivatives)

    for(Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
    {
        for (Index head_index = 0; head_index < heads_number; head_index++)
        {
            error_value_derivatives.chip(head_index, 3).chip(sample_index, 0).device(*thread_pool_device) =
                softmax_attention_scores.chip(head_index, 3).chip(sample_index, 0).contract(
                    error_attention_output_derivatives.chip(head_index, 3).chip(sample_index, 0), AT_B);
        }
    }



    //calculate_error_value_weights_derivatives() // using context and error_value_derivatives 


    /// QUERY AND KEY DERIVATIVES

    //calculate_error_attention_scores_derivatives(); // using deltas, projection_weights, value and softmax_derivatives(attention_scores)

    //calculate_error_query_derivatives() // using error_attention_scores_derivatives, key, weight_depth and query_weights (sum)
    //calculate_error_key_derivatives() // using error_attention_scores_derivatives, query, weight_depth and key_weights (sum)

    //calculate_error_query_weights_derivatives() // using input and error_query_derivatives
    //calculate_error_key_weights_derivatives() // using context and error_key_derivatives
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
