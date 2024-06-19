//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a flatten layer.
/// This constructor also initializes the rest of the class members to their default values.

FlattenLayer::FlattenLayer() : Layer()
{
    layer_type = Layer::Type::Flatten;
}


FlattenLayer::FlattenLayer(const dimensions& new_inputs_dimensions) : Layer()
{
    inputs_dimensions = new_inputs_dimensions;

    set(inputs_dimensions);

    layer_type = Type::Flatten;
}


dimensions FlattenLayer::get_inputs_dimensions() const
{
    return inputs_dimensions;
}


void FlattenLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's kernels to an image.

Index FlattenLayer::get_outputs_number() const
{
    return inputs_dimensions[0] * inputs_dimensions[1] * inputs_dimensions[2];
}


dimensions FlattenLayer::get_outputs_dimensions() const
{
    return { inputs_dimensions[0] * inputs_dimensions[1] * inputs_dimensions[2] };
}


Index FlattenLayer::get_inputs_number() const
{
    return inputs_dimensions[0] * inputs_dimensions[1] * inputs_dimensions[2] * inputs_dimensions[3];
}


Index FlattenLayer::get_inputs_rows_number() const
{
    return inputs_dimensions[0];
}


Index FlattenLayer::get_inputs_raw_variables_number() const
{
    return inputs_dimensions[1];
}


Index FlattenLayer::get_inputs_channels_number() const
{
    return inputs_dimensions[2];
}


/// Returns the number of neurons

Index FlattenLayer::get_neurons_number() const
{
    return inputs_dimensions[0] * inputs_dimensions[1] * inputs_dimensions[2];
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (number of images (batch), number of channels, width, height).

void FlattenLayer::set(const dimensions& new_inputs_dimensions)
{
    layer_name = "flatten_layer";

    inputs_dimensions = new_inputs_dimensions;
}


void FlattenLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                     LayerForwardPropagation* layer_forward_propagation,
                                     const bool& is_training)
{
    const Index batch_samples_number = layer_forward_propagation->batch_samples_number;

    const Index neurons_number = get_neurons_number();

    FlattenLayerForwardPropagation* flatten_layer_forward_propagation =
            static_cast<FlattenLayerForwardPropagation*>(layer_forward_propagation);

    type* outputs_data = flatten_layer_forward_propagation->outputs.data();

    memcpy(outputs_data,
           inputs_pair(0).first,
           batch_samples_number*neurons_number*sizeof(type));

    const TensorMap<Tensor<type, 2>> outputs(inputs_pair(0).first, batch_samples_number, neurons_number);
}


void FlattenLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                            const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                            LayerForwardPropagation* forward_propagation,
                                            LayerBackPropagation* back_propagation) const
{
    const Index batch_samples_number = inputs_pair(0).second[0];
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 2>> deltas(deltas_pair(0).first, deltas_pair(0).second[0], deltas_pair(0).second[1]);

    // Back propagation

    FlattenLayerBackPropagation* flatten_layer_back_propagation =
        static_cast<FlattenLayerBackPropagation*>(back_propagation);

    Tensor<type, 4>& input_derivatives = flatten_layer_back_propagation->input_derivatives;

    memcpy(input_derivatives.data(),
           deltas_pair(0).first,
           static_cast<Index>(batch_samples_number * neurons_number * sizeof(type)));
}


/// Serializes the flatten layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void FlattenLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("FlattenLayer");

    file_stream.OpenElement("InputVariablesDimension");

    file_stream.OpenElement("InputHeight");
    buffer.str("");
    buffer << get_inputs_rows_number();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputWidth");
    buffer.str("");
    buffer << get_inputs_raw_variables_number();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputChannels");
    buffer.str("");
    buffer << get_inputs_channels_number();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this flatten layer object.
/// @param document TinyXML document containing the member data.

void FlattenLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* flatten_layer_element = document.FirstChildElement("FlattenLayer");

    if(!flatten_layer_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenLayer element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Flatten layer input variables dimenison

    const tinyxml2::XMLElement* input_variables_dimensions_element = flatten_layer_element->FirstChildElement("InputVariablesDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputVariablesDimensions element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Input height

    const tinyxml2::XMLElement* input_height_element = input_variables_dimensions_element->NextSiblingElement("InputHeight");

    if(!input_height_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputHeight element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const Index input_height = Index(atoi(input_height_element->GetText()));

    // Input width

    const tinyxml2::XMLElement* input_width_element = input_variables_dimensions_element->NextSiblingElement("InputWidth");

    if(!input_width_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputWidth element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const Index input_width = Index(atoi(input_width_element->GetText()));

    // Input channels number

    const tinyxml2::XMLElement* input_channels_number_element = input_variables_dimensions_element->NextSiblingElement("InputChannels");

    if(!input_channels_number_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputChannelsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const Index input_channels_number = Index(atoi(input_channels_number_element->GetText()));

//    Tensor<Index,1> inputsDimensionTensor(4);

//    inputsDimensionTensor.setValues({input_height, input_width, input_channels_number, 0});

//    set(inputsDimensionTensor);

    // @todo Change to dimensions
    /*
    Tensor<Index, 1> inputs_dimension_tensor(3);

    inputs_dimension_tensor.setValues({input_height, input_width, input_channels_number});

    set(inputs_dimension_tensor);
    */
}

pair<type*, dimensions> FlattenLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, neurons_number });
}

void FlattenLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const Index neurons_number = layer->get_neurons_number();

    outputs.resize(batch_samples_number, neurons_number);

    outputs_data = outputs.data();
}


void FlattenLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(layer);

    dimensions inputs_dimensions = flatten_layer->get_inputs_dimensions();

    input_derivatives.resize(batch_samples_number,
            inputs_dimensions[0],
            inputs_dimensions[1],
            inputs_dimensions[2]);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, inputs_dimensions[0], inputs_dimensions[1], inputs_dimensions[2] };
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
