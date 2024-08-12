//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "strings_utilities.h"
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
                               const Index& new_inputs_number,
                               const Index& new_depth,
                               const bool& new_positional_encoding) : Layer()
{
    set(new_inputs_dimension, new_inputs_number, new_depth, new_positional_encoding);

    layer_type = Type::Embedding;

    layer_name = "embedding_layer";
}


/// Returns the dimension (maximum value) of the input to the layer.

Index EmbeddingLayer::get_input_dimension() const
{
    return input_dimensions;
}


/// Returns the length of the input to the layer.

Index EmbeddingLayer::get_inputs_number() const
{
    return inputs_number;
}


/// Returns the embedding depth to be used in the layer.

Index EmbeddingLayer::get_depth() const
{
    return depth;
}

bool EmbeddingLayer::get_positional_encoding() const
{
    return positional_encoding;
}


dimensions EmbeddingLayer::get_output_dimensions() const
{
    return { inputs_number, depth };
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


Tensor<type, 1> EmbeddingLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    copy(/*execution::par,*/
        embedding_weights.data(),
        embedding_weights.data() + embedding_weights.size(),
        parameters.data());

    return parameters;
}


Index EmbeddingLayer::get_neurons_number() const
{
    return inputs_number * depth;
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
    input_dimensions = 0;

    inputs_number = 0;

    depth = 0;

    positional_encoding = false;

    embedding_weights.resize(0, 0);

    set_default();
}


/// Sets new input dimension, input length, embedding depth and activation function of the layer.
/// It also sets the rest of the members to their default values.

void EmbeddingLayer::set(const Index& new_inputs_dimension,
                         const Index& new_inputs_number,
                         const Index& new_depth,
                         const bool& new_positional_encoding)
{
    input_dimensions = new_inputs_dimension;

    inputs_number = new_inputs_number;

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
    input_dimensions = new_inputs_dimension;

    set_embedding_weights();
}


/// Sets a new input length in the layer.

void EmbeddingLayer::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


/// Sets a new embedding depth in the layer.

void EmbeddingLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_embedding_weights();
}

void EmbeddingLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


/// Sets the lookup table and randomizes its parameters.

void EmbeddingLayer::set_embedding_weights()
{
    embedding_weights.resize(input_dimensions, depth);

    set_parameters_random();
}


void EmbeddingLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    copy(/*execution::par,*/
        new_parameters.data() + index,
        new_parameters.data() + index + embedding_weights.size(),
        embedding_weights.data());
}


void EmbeddingLayer::set_parameters_random()
{
    /// @todo Avoid loops

    const type minimum = type(-0.05);
    const type maximum = type(0.05);

//    embedding_weights = Eigen::internal::random<Eigen::Tensor<type, 2>>(1, 1).array() * 0.4 - 0.2;

    // First row must be 0s because input value 0 is padding
    
    embedding_weights.chip(0, 0).setConstant(0);
    
    #pragma omp parallel for

    for(Index i = 1; i < embedding_weights.dimension(0); i++)
    {
        for(Index j = 0; j < embedding_weights.dimension(1); j++)
        {
            const type random = type(rand()/(RAND_MAX+1.0));

            embedding_weights(i, j) = minimum + (maximum - minimum)*random;
        }
    }
}

void EmbeddingLayer::set_parameters_constant(const type& value)
{
    embedding_weights.setConstant(value);
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void EmbeddingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void EmbeddingLayer::dropout(Tensor<type, 3>& outputs)
{
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for(Index i = 0; i < outputs.size(); i++)
    {
        random = calculate_random_uniform(type(0), type(1));

        if(random < dropout_rate)    outputs(i) = 0;
        else    outputs(i) *= scaling_factor;
    }
}


/*
/// Calculates one-hot encoding, of dimension = input_dimensions, of an input row (assuming all input values are integers)
/// @return Matrix of one-hot encodings of all values in input_row

Tensor<type, 2> EmbeddingLayer::one_hot_encode_row(const Tensor<type, 1>& input_row)
{
    Tensor<type, 2> one_hot_encoded_input_row(inputs_number, input_dimensions);
    one_hot_encoded_input_row.setZero();

    const Tensor<type, 0> max_input = input_row.maximum();

    if(max_input(0) >= type(input_dimensions))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
               << "void EmbeddingLayer::one_hot_encode_row(const Tensor<Index, 1>&)\n"
               << "All input values must be less than " << input_dimensions << " (" << max_input(0) << ").\n";
        throw invalid_argument(buffer.str());
    }

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        one_hot_encoded_input_row(i, Index(input_row(i))) = 1;

    return one_hot_encoded_input_row;
}
*/


void EmbeddingLayer::lookup_embedding(const Tensor<type, 2>& inputs, Tensor<type, 3>& outputs)
{
    const Index batch_size = inputs.dimension(0);

#pragma omp parallel for
    for(Index row = 0; row < batch_size; row++)
    {
        for(Index input_position = 0; input_position < inputs_number; input_position++)
        {
            outputs.chip(row, 0).chip(input_position, 0)
                = embedding_weights.chip(inputs(row, input_position), 0);
        }
    }
}


void EmbeddingLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                       LayerForwardPropagation* layer_forward_propagation,
                                       const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1]);

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation
        = static_cast<EmbeddingLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 3>& outputs = embedding_layer_forward_propagation->outputs;

    lookup_embedding(inputs, outputs);

    if(positional_encoding)
    {
        outputs.device(*thread_pool_device) = outputs * outputs.constant(sqrt(depth));

        const Tensor<type, 2>& positional_encoding = embedding_layer_forward_propagation->positional_encoding;
        
        for(Index batch_element = 0; batch_element < outputs.dimension(0); batch_element++)
        {
            outputs.chip(batch_element, 0).device(*thread_pool_device) += positional_encoding;
        }
    }

    if(dropout_rate > 0 && is_training)    dropout(outputs);
}


void EmbeddingLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                              const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                              LayerForwardPropagation* forward_propagation,
                                              LayerBackPropagation* back_propagation) const
{
    const Index batch_samples_number = inputs_pair(0).second[0];
    const Index inputs_number = inputs_pair(0).second[1];

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, batch_samples_number, inputs_number);

    if(deltas_pair.size() > 1)     add_deltas(deltas_pair);

    const TensorMap<Tensor<type, 3>> deltas(deltas_pair(0).first, batch_samples_number, inputs_number, deltas_pair(0).second[2]);

    // Back propagation

    EmbeddingLayerBackPropagation* embedding_layer_back_propagation = static_cast<EmbeddingLayerBackPropagation*>(back_propagation);

    Tensor<type, 2>& sample_deltas = embedding_layer_back_propagation->sample_deltas;
    Tensor<type, 2>& embedding_weights_derivatives = embedding_layer_back_propagation->embedding_weights_derivatives;
    
    embedding_weights_derivatives.setZero();
    
    for(Index i = 0; i < batch_samples_number; i++)
    {
        if(positional_encoding)
            sample_deltas.device(*thread_pool_device) = deltas.chip(i, 0) * sample_deltas.constant(sqrt(depth));
        else
            sample_deltas.device(*thread_pool_device) = deltas.chip(i, 0);

        for(Index j = 0; j < inputs_number; j++)
        {
            embedding_weights_derivatives.chip(Index(inputs(i, j)), 0).device(*thread_pool_device) += sample_deltas.chip(j, 0);
        }
    }
}


void EmbeddingLayer::add_deltas(const Tensor<pair<type*, dimensions>, 1>& deltas_pair) const
{
    TensorMap<Tensor<type, 3>> deltas(deltas_pair(0).first,
                                      deltas_pair(0).second[0],
                                      deltas_pair(0).second[1],
                                      deltas_pair(0).second[2]);
     
    for(Index i = 1; i < deltas_pair.size(); i++)
    {
        const TensorMap<Tensor<type, 3>> other_deltas(deltas_pair(i).first,
                                                      deltas_pair(i).second[0],
                                                      deltas_pair(i).second[1],
                                                      deltas_pair(i).second[2]);

        deltas.device(*thread_pool_device) += other_deltas;
    }
}


void EmbeddingLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                     const Index& index,
                                     Tensor<type, 1>& gradient) const
{
    const Index embedding_weights_number = get_parameters_number();

    const EmbeddingLayerBackPropagation* embedding_layer_back_propagation =
        static_cast<EmbeddingLayerBackPropagation*>(back_propagation);

    const type* embedding_weights_derivatives_data = embedding_layer_back_propagation->embedding_weights_derivatives.data();

    type* gradient_data = gradient.data();

    copy(/*execution::par,*/
        embedding_weights_derivatives_data,
        embedding_weights_derivatives_data + embedding_weights_number,
        gradient_data + index);
}


void EmbeddingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Embedding layer

    const tinyxml2::XMLElement* embedding_layer_element = document.FirstChildElement("EmbeddingLayer");

    if(!embedding_layer_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "EmbeddingLayer element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = embedding_layer_element->FirstChildElement("LayerName");

    if(!layer_name_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "LayerName element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(layer_name_element->GetText())
    {
        set_name(layer_name_element->GetText());
    }

    // Input dimension

    const tinyxml2::XMLElement* input_dimension_element = embedding_layer_element->FirstChildElement("InputDimension");

    if(!input_dimension_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "InputDimension element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(input_dimension_element->GetText())
    {
        set_input_dim(Index(stoi(input_dimension_element->GetText())));
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = embedding_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "InputsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));
    }

    // Embedding depth

    const tinyxml2::XMLElement* depth_element = embedding_layer_element->FirstChildElement("Depth");

    if(!depth_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "Depth element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(depth_element->GetText())
    {
        set_depth(Index(stoi(depth_element->GetText())));
    }

    // Positional encoding

    const tinyxml2::XMLElement* positional_encoding_element = embedding_layer_element->FirstChildElement("PositionalEncoding");

    if(!positional_encoding_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "PositionalEncoding element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(positional_encoding_element->GetText())
    {
        positional_encoding = string(positional_encoding_element->GetText()) == "true";
    }

    // Embedding weights

    const tinyxml2::XMLElement* parameters_element = embedding_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "Parameters element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();

        set_parameters(to_type_vector(parameters_string, " "));
    }
}

void EmbeddingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Embedding layer

    file_stream.OpenElement("EmbeddingLayer");

    // Layer name
    file_stream.OpenElement("LayerName");
    buffer.str("");
    buffer << layer_name;
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Input dimension
    file_stream.OpenElement("InputDimension");

    buffer.str("");
    buffer << get_input_dimension();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Inputs number

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << get_inputs_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Embedding depth

    file_stream.OpenElement("Depth");

    buffer.str("");
    buffer << get_depth();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Positional encoding

    file_stream.OpenElement("PositionalEncoding");

    buffer.str("");
    buffer << (positional_encoding ? "true" : "false");

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");

    buffer.str("");

    const Tensor<type, 1> parameters = get_parameters();
    const Index parameters_size = parameters.size();

    for(Index i = 0; i < parameters_size; i++)
    {
        buffer << parameters(i);

        if(i != (parameters_size - 1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Embedding layer (end tag)

    file_stream.CloseElement();
}



pair<type*, dimensions> EmbeddingLayerForwardPropagation::get_outputs_pair() const
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index inputs_number = embedding_layer->get_inputs_number();

    const Index depth = embedding_layer->get_depth();
    
    return pair<type*, dimensions>(outputs_data, { batch_samples_number, inputs_number, depth });
}


void EmbeddingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(new_layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = embedding_layer->get_inputs_number();

    const Index depth = embedding_layer->get_depth();

    // Outputs

    outputs.resize(batch_samples_number, inputs_number, depth);

    outputs_data = outputs.data();

    if(embedding_layer->get_positional_encoding())    build_positional_encoding_matrix();
}


void EmbeddingLayerForwardPropagation::build_positional_encoding_matrix()
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index inputs_number = embedding_layer->get_inputs_number();
    const Index depth = embedding_layer->get_depth();

    positional_encoding.resize(inputs_number, depth);

    positional_encoding.setZero();

    const type half_depth = type(depth) / 2;

    #pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
    {
        for(Index j = 0; j < Index(depth); j++)
        {
            if(j < Index(half_depth))
                positional_encoding(i, j) = sin((i) / pow(10000, (j) / half_depth));
            else
                positional_encoding(i, j) = cos((i) / pow(10000, (j - Index(half_depth)) / half_depth));
        }
    }


    built_positional_encoding_matrix = true;
}


void EmbeddingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(new_layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = embedding_layer->get_inputs_number();
    const Index depth = embedding_layer->get_depth();
    const Index input_dimension = embedding_layer->get_input_dimension();

    sample_deltas.resize(inputs_number, depth);
    embedding_weights_derivatives.resize(input_dimension, depth);

    inputs_derivatives.resize(0); // Always input layer
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
