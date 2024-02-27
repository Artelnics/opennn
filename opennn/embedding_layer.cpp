//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "embedding_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.


EmbeddingLayer::EmbeddingLayer() : Layer()
{
    set();

    layer_type = Type::Embedding;
}


/// Layer architecture constructor.
/// It creates a layer object with given input dimension, input length and embedding depth.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.

EmbeddingLayer::EmbeddingLayer(const Index& new_inputs_dimension,
                               const Index& new_input_length,
                               const Index& new_depth,
                               const bool& new_positional_encoding) : Layer()
{
    set(new_inputs_dimension, new_input_length, new_depth, new_positional_encoding);

    layer_type = Type::Embedding;

    layer_name = "embedding_layer";
}


/// Returns the dimension (maximum value + 1) of the input to the layer.

Index EmbeddingLayer::get_input_dimension() const
{
    return inputs_dimension;
}


/// Returns the length of the input to the layer.

Index EmbeddingLayer::get_input_length() const
{
    return inputs_length;
}


/// Returns the embedding depth to be used in the layer.

Index EmbeddingLayer::get_depth() const
{
    return depth;
}

Tensor<type, 2> EmbeddingLayer::get_embedding_weights() const
{
    return embedding_weights;
}


/// Returns the number of parameters of the layer.

Index EmbeddingLayer::get_parameters_number() const
{
    return embedding_weights.size();
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& EmbeddingLayer::get_display() const
{
    return display;
}


/// Sets an empty layer.
/// It also sets the rest of the members to their default values.

void EmbeddingLayer::set()
{
    inputs_dimension = 0;

    inputs_length = 0;

    depth = 0;

    positional_encoding = false;

    embedding_weights.resize(0, 0);

    set_default();
}


/// Sets new input dimension, input length, embedding depth and activation function of the layer.
/// It also sets the rest of the members to their default values.

void EmbeddingLayer::set(const Index& new_inputs_dimension,
                         const Index& new_input_length,
                         const Index& new_depth,
                         const bool& new_positional_encoding)
{
    inputs_dimension = new_inputs_dimension;

    inputs_length = new_input_length;

    depth = new_depth;

    set_embedding_weights();

    positional_encoding = new_positional_encoding;

    set_default();
}


/// Sets those members not related to the perceptrons to their default value.

void EmbeddingLayer::set_default()
{
    layer_name = "embedding_layer";

    display = true;

    layer_type = Type::Embedding;
}


void EmbeddingLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new input dim in the layer.

void EmbeddingLayer::set_input_dim(const Index& new_inputs_dimension)
{
    inputs_dimension = new_inputs_dimension;

    set_embedding_weights();
}


/// Sets a new input length in the layer.

void EmbeddingLayer::set_input_length(const Index& new_input_length)
{
    inputs_length = new_input_length;
}


/// Sets a new embedding depth in the layer.

void EmbeddingLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_embedding_weights();
}


/// Sets the lookup table and randomizes its parameters.

void EmbeddingLayer::set_embedding_weights()
{
    embedding_weights.resize(inputs_dimension, depth);

    set_parameters_random();
}


void EmbeddingLayer::set_parameters_random()
{
    /// @todo Avoid loops

    const type minimum = type(-0.2);
    const type maximum = type(0.2);

//    embedding_weights = Eigen::internal::random<Eigen::Tensor<type, 2>>(1, 1).array() * 0.4 - 0.2;

    // first row must be 0s because input value 0 is padding

    for(Index j = 0; j < depth; j++)
    {
        embedding_weights(0, j) = type(0);
    }

    for(Index i = 1; i < inputs_dimension; i++)
    {
        for(Index j = 0; j < depth; j++)
        {
            const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

            embedding_weights(i, j) = minimum + (maximum - minimum)*random;
        }
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void EmbeddingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/*
/// Calculates one-hot encoding, of dimension = inputs_dimension, of an input row (assuming all input values are integers)
/// @return Matrix of one-hot encodings of all values in input_row

Tensor<type, 2> EmbeddingLayer::one_hot_encode_row(const Tensor<type, 1>& input_row)
{
    Tensor<type, 2> one_hot_encoded_input_row(inputs_length, inputs_dimension);
    one_hot_encoded_input_row.setZero();

    const Tensor<type, 0> max_input = input_row.maximum();

    if(max_input(0) >= type(inputs_dimension))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
               << "void EmbeddingLayer::one_hot_encode_row(const Tensor<Index, 1>&)\n"
               << "All input values must be less than " << inputs_dimension << " (" << max_input(0) << ").\n";
        throw invalid_argument(buffer.str());
    }

#pragma omp parallel for
    for(Index i = 0; i < inputs_length; i++)
        one_hot_encoded_input_row(i, Index(input_row(i))) = 1;

    return one_hot_encoded_input_row;
}
*/


/// Looks up embedding of an input row, by passing its one-hot encoding through a perceptron layer (that corresponds to the lookup table)
/// Saves the embedding matrix of the row in outputs_data of the given perceptron layer forward propagation structure

void EmbeddingLayer::lookup_embedding(const Tensor<type, 2>& inputs, Tensor<type, 3>& outputs)
{
    const Index batch_size = inputs.dimension(0);

    for(Index row = 0; row < batch_size; row++)
    {
        for(Index input_position = 0; input_position < inputs_length; input_position++)
        {
            outputs.chip(row, 0).chip(input_position, 0)
                = embedding_weights.chip(inputs(row, input_position), 0);
        }
    }
}


void EmbeddingLayer::forward_propagate(const pair<type*, dimensions>& inputs_pair,
                                       LayerForwardPropagation* layer_forward_propagation,
                                       const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs(inputs_pair.first, inputs_pair.second[0][0], inputs_pair.second[0][1]);

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation
        = static_cast<EmbeddingLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 3>& outputs = embedding_layer_forward_propagation->outputs;

    lookup_embedding(inputs, outputs);

    if(positional_encoding)
    {
        if(!embedding_layer_forward_propagation->built_positional_encoding_matrix)
        {
            embedding_layer_forward_propagation->build_positional_encoding_matrix();
        }

        const Tensor<type, 2>& positional_encoding = embedding_layer_forward_propagation->positional_encoding;

        for(Index batch_element = 0; batch_element < outputs.dimension(0); batch_element++)
        {
            outputs.chip(batch_element, 0)/*.device(thread_pool_device)*/ += positional_encoding;
        }
    }
}

void EmbeddingLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                            LayerBackPropagation* next_back_propagation,
                                            LayerBackPropagation* back_propagation) const
{

    EmbeddingLayerBackPropagation* embedding_layer_back_propagation =
        static_cast<EmbeddingLayerBackPropagation*>(back_propagation);

    switch (next_back_propagation->layer->get_type())
    {
    case Type::MultiheadAttention:
    {
        MultiheadAttentionLayerForwardPropagation* next_multihead_attention_layer_forward_propagation =
            reinterpret_cast<MultiheadAttentionLayerForwardPropagation*>(next_forward_propagation);

        MultiheadAttentionLayerBackPropagation* next_multihead_attention_layer_back_propagation =
            reinterpret_cast<MultiheadAttentionLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_multihead_attention_layer_forward_propagation,
                               next_multihead_attention_layer_back_propagation,
                               embedding_layer_back_propagation);
    }
    return;

    default:

        return;
    }
}

void EmbeddingLayer::calculate_hidden_delta(MultiheadAttentionLayerForwardPropagation* next_forward_propagation,
                                            MultiheadAttentionLayerBackPropagation* next_back_propagation,
                                            EmbeddingLayerBackPropagation* back_propagation) const
{
    // Next layer

    const MultiheadAttentionLayer* next_multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(next_back_propagation->layer);

    // Next back propagation

    const Tensor<type, 3>& next_error_input_derivatives = next_back_propagation->error_input_derivatives;
    const Tensor<type, 3>& next_error_context_derivatives = next_back_propagation->error_context_derivatives;

    // This back propagation

    Tensor<type, 3>& deltas = back_propagation->deltas;

    deltas.device(*thread_pool_device) = next_error_input_derivatives + next_error_context_derivatives;
}


void EmbeddingLayer::calculate_error_gradient(const pair<type*, dimensions>& inputs,
                                              LayerForwardPropagation* forward_propagation,
                                              LayerBackPropagation* back_propagation) const
{
    Index batch_samples_number = inputs.second[0][0];
    Index inputs_number = inputs.second[0][1];

    const TensorMap<Tensor<type, 2>> inputs_map(inputs.first, batch_samples_number, inputs_number);

    // Forward propagation

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation = static_cast<EmbeddingLayerForwardPropagation*>(forward_propagation);

    // Back propagation

    EmbeddingLayerBackPropagation* embedding_layer_back_propagation = static_cast<EmbeddingLayerBackPropagation*>(back_propagation);

    const Tensor<type, 3>& deltas = embedding_layer_back_propagation->deltas;

    Tensor<type, 2>& embedding_weights_derivatives = embedding_layer_back_propagation->embedding_weights_derivatives;

    embedding_weights_derivatives.setZero();

    for (Index i = 0; i < batch_samples_number; i++)
    {
        for (Index j = 0; j < inputs_number; j++)
        {
            embedding_weights_derivatives.chip(inputs_map(i, j), 0).device(*thread_pool_device) += deltas.chip(i, 0).chip(j, 0);
        }
    }
}

void EmbeddingLayerForwardPropagation::build_positional_encoding_matrix()
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index inputs_length = embedding_layer->get_input_length();
    const Index depth = embedding_layer->get_depth();

    positional_encoding.resize(inputs_length, depth);

    positional_encoding.setZero();

    const type half_depth = type(depth) / type(2);

    /// @todo (because h file?) Try to use matrix form

    #pragma omp parallel for

    for (Index i = 0; i < inputs_length; i++)
    {
        for (Index j = 0; j < Index(half_depth - 1); j++)
        {
            positional_encoding(i, 2 * j) = type(sin((i + 1) / pow(10000, (j + 1) / half_depth)));
            positional_encoding(i, 2 * j + 1) = type(cos((i + 1) / pow(10000, (j + 1) / half_depth)));
        }
    }

    if (depth % 2 == 0)
    {
        #pragma omp parallel for

        for (Index i = 0; i < inputs_length; i++)
        {
            positional_encoding(i, depth - 2) = type(sin((i + 1) / 10000));
            positional_encoding(i, depth - 1) = type(cos((i + 1) / 10000));
        }
    }
    else
    {
        #pragma omp parallel for

        for (Index i = 0; i < inputs_length; i++)
        {
            positional_encoding(i, depth - 1) = type(sin((i + 1) / 10000));
        }
    }

    built_positional_encoding_matrix = true;
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
