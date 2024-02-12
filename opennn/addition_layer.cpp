//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


#include "addition_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates an empty AdditionLayer object.

AdditionLayer::AdditionLayer() : Layer()
{
    set_default();
}


/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty AdditionLayer object.
/// @param new_input_variables_dimensions A vector containing the new number of channels, rows and raw_variables for the input.

AdditionLayer::AdditionLayer(const Tensor<Index, 1>& new_input_variables_dimensions) : Layer()
{
    set_default();
}


/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty AdditionLayer object.
/// @param new_input_variables_dimensions A vector containing the desired number of rows and raw_variables for the input.
/// @param pool_dimensions A vector containing the desired number of rows and raw_variables for the pool.

AdditionLayer::AdditionLayer(const Tensor<Index, 1>& new_input_variables_dimensions, const Tensor<Index, 1>& pool_dimensions) : Layer()
{ 
    set(new_input_variables_dimensions, pool_dimensions);

    inputs_dimensions = new_input_variables_dimensions;

    set_default();
}


/// Returns the number of neurons the layer applies to an image.

Index AdditionLayer::get_neurons_number() const
{
    return get_outputs_rows_number() * get_outputs_raw_variables_number();
}


/// Returns the layer's outputs dimensions.

Tensor<Index, 1> AdditionLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(3);

    outputs_dimensions[0] = get_outputs_rows_number();
    outputs_dimensions[1] = get_outputs_raw_variables_number();
    outputs_dimensions[2] = inputs_dimensions[2];

    return outputs_dimensions;
}


/// Returns the number of inputs of the layer.

Index AdditionLayer::get_inputs_number() const
{
    return inputs_dimensions.size();
}



/// Returns the number of rows of the layer's input.

Index AdditionLayer::get_inputs_rows_number() const
{
    return inputs_dimensions[0];
}


/// Returns the number of raw_variables of the layer's input.

Index AdditionLayer::get_inputs_raw_variables_number() const
{
    return inputs_dimensions[1];
}


/// Returns the number of channels of the layers' input.

Index AdditionLayer::get_channels_number() const
{
    return inputs_dimensions[2];
}


/// Returns the number of rows of the layer's output.

Index AdditionLayer::get_outputs_rows_number() const
{
    return 0;
}


/// Returns the number of raw_variables of the layer's output.

Index AdditionLayer::get_outputs_raw_variables_number() const
{
    return 0;
}


/// Returns the number of parameters of the layer.

Index AdditionLayer::get_parameters_number() const
{
    return 0;
}


/// Returns the layer's parameters.

Tensor<type, 1> AdditionLayer::get_parameters() const
{
    return Tensor<type, 1>();
}


/// Returns the input_variables_dimensions.

Tensor<Index, 1> AdditionLayer::get_inputs_dimensions() const
{
    return inputs_dimensions;
}


void AdditionLayer::set(const Tensor<Index, 1>& new_input_variables_dimensions, const Tensor<Index, 1>& new_pool_dimensions)
{
    inputs_dimensions = new_input_variables_dimensions;

    set_default();
}

void AdditionLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}

/// Sets the number of rows of the layer's input.
/// @param new_input_rows_number The desired rows number.

void AdditionLayer::set_inputs_dimensions(const Tensor<Index, 1>& new_inputs_dimensions)
{
    inputs_dimensions = new_inputs_dimensions;
}


/// Sets the layer type to Layer::Pooling.

void AdditionLayer::set_default()
{
    layer_type = Layer::Type::Pooling;
}


void AdditionLayer::forward_propagate(const pair<type*, dimensions>&,
                                     LayerForwardPropagation* layer_forward_propagation,
                                     const bool& is_training)
{
    if(layer_forward_propagation == nullptr) cout << "NULL" << endl;

}


void AdditionLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                          LayerBackPropagation* next_back_propagation,
                                          LayerBackPropagation* this_layeer_back_propagation) const
{
}


/// Serializes the addition layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void AdditionLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Pooling layer

    file_stream.OpenElement("AdditionLayer");

    // Layer name

    file_stream.OpenElement("LayerName");

    buffer.str("");
    buffer << layer_name;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Image size

    file_stream.OpenElement("InputsVariablesDimensions");

    buffer.str("");
    buffer << "1 1 1";

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //Filters number

    file_stream.OpenElement("FiltersNumber");

    buffer.str("");
    buffer << 9;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    // Filters size

    file_stream.OpenElement("FiltersSize");

    buffer.str("");
    buffer << 9;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

//    file_stream.PushText(write_pooling_method().c_str());

    file_stream.CloseElement();

    //_______________________________________________________


    // Inputs variables dimensions

    file_stream.OpenElement("InputDimensions");

    buffer.str("");
    buffer << get_inputs_dimensions();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this convolutional layer object.
/// @param document TinyXML document containing the member data.

void AdditionLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Pooling layer

    const tinyxml2::XMLElement* pooling_layer_element = document.FirstChildElement("AdditionLayer");

    if(!pooling_layer_element)
    {
        buffer << "OpenNN Exception: AdditionLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling layer element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Pooling method element

    const tinyxml2::XMLElement* pooling_method_element = pooling_layer_element->FirstChildElement("PoolingMethod");

    if(!pooling_method_element)
    {
        buffer << "OpenNN Exception: AdditionLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling method element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string pooling_method_string = pooling_method_element->GetText();

//    set_pooling_method(pooling_method_string);

    // Input variables dimensions element

    const tinyxml2::XMLElement* input_variables_dimensions_element = pooling_layer_element->FirstChildElement("InputDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: AdditionLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling input variables dimensions element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

//    set_input_variables_dimenisons(input_variables_dimensions_string);
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
