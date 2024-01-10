//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "embedding_layer.h"
#include "tensor_utilities.h"

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

EmbeddingLayer::EmbeddingLayer(const Index& new_input_dim,
                               const Index& new_input_length,
                               const Index& new_depth,
                               const bool& new_positional_encoding) : Layer()
{
    set(new_input_dim, new_input_length, new_depth, new_positional_encoding);

    layer_type = Type::Embedding;

    layer_name = "embedding_layer";
}


/// Returns the dimension (maximum value + 1) of the input to the layer.

Index EmbeddingLayer::get_input_dim() const
{
    return input_dim;
}


/// Returns the length of the input to the layer.

Index EmbeddingLayer::get_input_length() const
{
    return input_length;
}


/// Returns the embedding depth to be used in the layer.

Index EmbeddingLayer::get_depth() const
{
    return depth;
}


/// Returns the number of parameters of the layer.

Index EmbeddingLayer::get_parameters_number() const
{
    return lookup_table.size();
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
    input_dim = 0;

    input_length = 0;

    depth = 0;

    positional_encoding = false;

    lookup_table.resize(0, 0);

    set_default();
}


/// Sets new input dimension, input length, embedding depth and activation function of the layer.
/// It also sets the rest of the members to their default values.

void EmbeddingLayer::set(const Index& new_input_dim,
                         const Index& new_input_length,
                         const Index& new_depth,
                         const bool& new_positional_encoding)
{
    input_dim = new_input_dim;

    input_length = new_input_length;

    depth = new_depth;

    set_lookup_table();

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

void EmbeddingLayer::set_input_dim(const Index& new_input_dim)
{
    input_dim = new_input_dim;

    set_lookup_table();
}


/// Sets a new input length in the layer.

void EmbeddingLayer::set_input_length(const Index& new_input_length)
{
    input_length = new_input_length;
}


/// Sets a new embedding depth in the layer.

void EmbeddingLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_lookup_table();
}


/// Sets the lookup table and randomizes its parameters.

void EmbeddingLayer::set_lookup_table()
{
    lookup_table.resize(input_dim, depth);

    set_parameters_random();
}


void EmbeddingLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    // first row must is 0s because input value 0 is padding
    for(Index j = 0; j < depth; j++)
    {
        lookup_table(0, j) = 0;
    }

    for(Index i = 1; i < input_dim; i++)
    {
        for(Index j = 0; j < depth; j++)
        {
            const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

            lookup_table(i, j) = minimum + (maximum - minimum)*random;
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
/// Calculates one-hot encoding, of dimension = input_dim, of an input row (assuming all input values are integers)
/// @return Matrix of one-hot encodings of all values in input_row

Tensor<type, 2> EmbeddingLayer::one_hot_encode_row(const Tensor<type, 1>& input_row)
{
    Tensor<type, 2> one_hot_encoded_input_row(input_length, input_dim);
    one_hot_encoded_input_row.setZero();

    const Tensor<type, 0> max_input = input_row.maximum();

    if(max_input(0) >= type(input_dim))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
               << "void EmbeddingLayer::one_hot_encode_row(const Tensor<Index, 1>&)\n"
               << "All input values must be less than " << input_dim << " (" << max_input(0) << ").\n";
        throw invalid_argument(buffer.str());
    }

#pragma omp parallel for
    for(Index i = 0; i < input_length; i++)
        one_hot_encoded_input_row(i, Index(input_row(i))) = type(1);

    return one_hot_encoded_input_row;
}
*/


/// Looks up embedding of an input row, by passing its one-hot encoding through a perceptron layer (that corresponds to the lookup table)
/// Saves the embedding matrix of the row in outputs_data of the given perceptron layer forward propagation structure

void EmbeddingLayer::lookup_embedding(const Tensor<type, 2>& inputs, EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation)
{
    TensorMap<Tensor<type, 3>> output = embedding_layer_forward_propagation->outputs(0).to_tensor_map<3>();

    const Index batch_size = inputs.dimension(0);

//#pragma omp parallel for
    for(Index batch_element = 0; batch_element < batch_size; batch_element++)
    {
        for(Index input_position = 0; input_position < input_length; input_position++)
        {
            output.chip(batch_element, 0).chip(input_position, 0) = lookup_table.chip(inputs(batch_element, input_position), 0);
        }
    }

    if(positional_encoding)
    {
        const Tensor<type, 2> positional_encoding_matrix = build_positional_encoding_matrix();
//#pragma omp parallel for
        for(Index batch_element = 0; batch_element < batch_size; batch_element++)
        {
            output.chip(batch_element, 0) += positional_encoding_matrix;
        }
    }
};


/// Builds positional encoding matrix with dimensions (input_length, depth) of the layer.

const Tensor<type, 2> EmbeddingLayer::build_positional_encoding_matrix()
{
    Tensor<type, 2> positional_encoding_matrix(input_length, depth);
    positional_encoding_matrix.setZero();
/*
    type half_depth = type(depth)/type(2);

#pragma omp parallel for collapse(2)
    for(Index i = 0; i < input_length; i++)
    {
        for(Index j = 0; j < static_cast<Index>(half_depth - 1); j++)
        {
            positional_encoding_matrix(i, 2*j) = sin( (i + 1) / pow(100000, (j + 1) / half_depth) );
            positional_encoding_matrix(i, 2*j+1) = cos( (i + 1) / pow(100000, (j + 1) / half_depth) );
        }

        if(depth % 2 == 0)
        {
            positional_encoding_matrix(i, depth - 2) = sin( (i+1) / 10000 );
            positional_encoding_matrix(i, depth - 1) = cos( (i+1) / 10000 );
        }
        else
        {
            positional_encoding_matrix(i, depth - 1) = sin( (i+1) / 10000 );
        }
    }
*/
    return positional_encoding_matrix;
};


void EmbeddingLayer::forward_propagate(const pair<type*, dimensions>& inputs,
                                        LayerForwardPropagation* forward_propagation,
                                        const bool& is_training)
{
    const Index batch_size = forward_propagation->batch_samples_number;

    const TensorMap<Tensor<type, 2>> inputs_map(inputs.first, inputs.second[0][0], inputs.second[0][1]);

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation
        = static_cast<EmbeddingLayerForwardPropagation*>(forward_propagation);

    PerceptronLayerForwardPropagation perceptron_layer_forward_propagation =
        embedding_layer_forward_propagation->perceptron_forward_propagation;

    Tensor<type, 1> input_row(input_length);

//#pragma omp parallel for
    for(Index i = 0; i < batch_size; i++)
    {
        input_row = inputs_map.chip(i, 0);
        lookup_embedding(input_row, &perceptron_layer_forward_propagation, is_training);

        memcpy(embedding_layer_forward_propagation->outputs.data() + i*input_length*depth,
               perceptron_layer_forward_propagation.outputs.data(),
               static_cast<size_t>(input_length*depth*sizeof(type)));
    }

    if(positional_encoding)
    {
        Tensor<type, 3>& outputs = embedding_layer_forward_propagation->outputs;

        const Tensor<type, 2> positional_encoding_matrix = build_positional_encoding_matrix();
        #pragma omp parallel for
        for(Index i = 0; i < batch_size; i++)
            outputs.chip(i, 0) += positional_encoding_matrix;
    }
}


/*
void EmbeddingLayer::forward_propagate(type* inputs_data,
                                        const Tensor<Index, 1>& inputs_dimensions,
                                        Tensor<type, 1>& potential_parameters,
                                        LayerForwardPropagation* forward_propagation)
{
#ifdef OPENNN_DEBUG
    if(inputs_dimensions(1) != get_inputs_number())
    {
        ostringstream buffer;
        buffer << "OpenNN Exception:" << LOG << endl
               << "void forward_propagate(type*, const Tensor<Index, 1>&, Tensor<type, 1>&, LayerForwardPropagation*) final method.\n"
               << "Inputs columns number must be equal to " << get_inputs_number() << ", (inputs number).\n";

        throw invalid_argument(buffer.str());
    }

    check_size(potential_parameters, get_parameters_number(), LOG);
#endif

    const TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(), 1, neurons_number);

    const TensorMap<Tensor<type, 2>> potential_synaptic_weights(potential_parameters.data()+neurons_number, inputs_number, neurons_number);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
        = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation);
>>>>>>> f437e115fe9e567c3475cda88f60e74912a668c2

}


/// @todo
/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names vector of strings with the name of the layer inputs.
/// @param outputs_names vector of strings with the name of the layer outputs.

string EmbeddingLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    return string();
}


void EmbeddingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
}


void EmbeddingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
