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
    //set();

    layer_type = Type::Flatten;
}


FlattenLayer::FlattenLayer(const Tensor<Index, 1>& new_input_variables_dimensions) : Layer()
{
    input_variables_dimensions = new_input_variables_dimensions;
    set(input_variables_dimensions);

    layer_type = Type::Flatten;
}

void FlattenLayer::set_parameters(const Tensor<type, 1>&, const Index&)
{

}

Tensor<Index, 1> FlattenLayer::get_input_variables_dimensions() const
{
    return input_variables_dimensions;
}

/// @todo
/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's kernels to an image.

Tensor<Index, 1> FlattenLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(2);

    outputs_dimensions(0) = input_variables_dimensions(0) * input_variables_dimensions(1) * input_variables_dimensions(2);
    outputs_dimensions(1) = 1; //input_variables_dimensions(3); // Number of batches

    return outputs_dimensions;
}

/// @todo
Index FlattenLayer::get_inputs_number() const
{
    return input_variables_dimensions(0) * input_variables_dimensions(1) * input_variables_dimensions(2) * input_variables_dimensions(3);
}

Index FlattenLayer::get_input_height() const
{
    return input_variables_dimensions(0);
}

Index FlattenLayer::get_input_width() const
{
    return input_variables_dimensions(1);
}

Index FlattenLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions(2);
}

Index FlattenLayer::get_inputs_batch_number() const
{
    return input_variables_dimensions(3);
}

/// Returns the number of neurons

Index FlattenLayer::get_neurons_number() const
{
    return input_variables_dimensions(0)*input_variables_dimensions(1)*input_variables_dimensions(2);
}


Tensor<type, 1> FlattenLayer::get_parameters() const
{
    return Tensor<type,1>();
}


/// Returns the number of parameters of the layer.

Index FlattenLayer::get_parameters_number() const
{
    return 0;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.

/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (number of images (batch), number of channels, width, height).

void FlattenLayer::set(const Tensor<Index, 1>& new_inputs_dimensions)
{
    input_variables_dimensions = new_inputs_dimensions;

}


/// Obtain the connection between the convolutional and the conventional part
/// of a neural network. That is a matrix which links to the perceptron layer.
/// @param inputs 4d tensor(batch, channels, width, height)
/// @return result 2d tensor(batch, number of pixels)

Tensor<type, 2> FlattenLayer::calculate_outputs_2d(const Tensor<type, 4>& inputs)
{
    const Index batch = inputs.dimension(3);
    const Index channels = inputs.dimension(2);
    const Index heights = inputs.dimension(0);
    const Index width = inputs.dimension(1);

    const Eigen::array<Index, 2> new_dims{{batch, channels*width*heights}};

    const Tensor<type, 2> outputs = inputs.reshape(new_dims);

    return outputs;
}


void FlattenLayer::forward_propagate(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions, LayerForwardPropagation* forward_propagation)
{
    FlattenLayerForwardPropagation* flatten_layer_forward_propagation
            = static_cast<FlattenLayerForwardPropagation*>(forward_propagation);

#ifdef OPENNN_DEBUG

    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    if(outputs_dimensions[0] != flatten_layer_forward_propagation->outputs.dimension(0))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer::forward_propagate.\n"
               << "outputs_dimensions[0]" <<outputs_dimensions[0] <<"must be equal to" << flatten_layer_forward_propagation->outputs.dimension(0)<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs_dimensions[1] != flatten_layer_forward_propagation->outputs.dimension(1))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer::forward_propagate.\n"
               << "outputs_dimensions[1]" <<outputs_dimensions[1] <<"must be equal to" << flatten_layer_forward_propagation->outputs.dimension(1)<<".\n";

        throw invalid_argument(buffer.str());
    }

#endif

    if(inputs_dimensions.size() != 4)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) final.\n"
               << "Inputs rank must be equal to 4.\n";

        throw invalid_argument(buffer.str());
    }

    const Index batch = inputs_dimensions(3);
    const Index channels = inputs_dimensions(2);
    const Index heights = inputs_dimensions(0);
    const Index width = inputs_dimensions(1);

    const Eigen::array<Index, 2> new_dims{{batch, channels*width*heights}};

    TensorMap<Tensor<type, 4>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1), inputs_dimensions(2), inputs_dimensions(3));

    flatten_layer_forward_propagation->outputs = inputs.reshape(new_dims);
}


/// Serializes the flatten layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void FlattenLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("FlattenLayer");

    file_stream.OpenElement("InputVariablesDimension");

    file_stream.OpenElement("InputChannels");
    buffer.str("");
    buffer << get_inputs_channels_number();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputWidth");
    buffer.str("");
    buffer << get_input_width();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputHeight");
    buffer.str("");
    buffer << get_input_height();

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

        throw invalid_argument(buffer.str());
    }

    // Flatten layer input variables dimenison

    const tinyxml2::XMLElement* input_variables_dimensions_element = flatten_layer_element->FirstChildElement("InputVariablesDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputVariablesDimensions element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Input channels number

    const tinyxml2::XMLElement* input_channels_number_element = input_variables_dimensions_element->NextSiblingElement("InputChannels");

    if(!input_channels_number_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputChannelsNumber element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index input_channels_number = static_cast<Index>(atoi(input_channels_number_element->GetText()));

    // Input width

    const tinyxml2::XMLElement* input_width_element = input_variables_dimensions_element->NextSiblingElement("InputWidth");

    if(!input_width_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputWidth element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index input_width = static_cast<Index>(atoi(input_width_element->GetText()));

    // Input height

    const tinyxml2::XMLElement* input_height_element = input_variables_dimensions_element->NextSiblingElement("InputHeight");

    if(!input_height_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputHeight element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index input_height = static_cast<Index>(atoi(input_height_element->GetText()));

    Tensor<Index,1> inputsDimensionTensor(4);

    inputsDimensionTensor.setValues({0, input_channels_number, input_width, input_height});

    set(inputsDimensionTensor);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
