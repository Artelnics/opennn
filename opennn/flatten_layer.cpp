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


FlattenLayer::FlattenLayer(const Tensor<Index, 1>& new_inputs_dimensions) : Layer()
{
    inputs_dimensions = new_inputs_dimensions;

    set(inputs_dimensions);

    layer_type = Type::Flatten;
}


void FlattenLayer::set_parameters(const Tensor<type, 1>&, const Index&)
{
    return;
}


Tensor<Index, 1> FlattenLayer::get_inputs_dimensions() const
{
    return inputs_dimensions;
}


void FlattenLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// @todo
/// Returns a vector containing the number of channels, rows and raw_variables of the result of applying the layer's kernels to an image.

Index FlattenLayer::get_outputs_number() const
{
    return inputs_dimensions(0) * inputs_dimensions(1) * inputs_dimensions(2);
}


Tensor<Index, 1> FlattenLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(1);

    outputs_dimensions[0] = inputs_dimensions(0) * inputs_dimensions(1) * inputs_dimensions(2);

    return outputs_dimensions;
}

/// @todo
Index FlattenLayer::get_inputs_number() const
{
    return inputs_dimensions(0) * inputs_dimensions(1) * inputs_dimensions(2) * inputs_dimensions(3);
}


Index FlattenLayer::get_inputs_rows_number() const
{
    return inputs_dimensions(0);
}


Index FlattenLayer::get_inputs_raw_variables_number() const
{
    return inputs_dimensions(1);
}


Index FlattenLayer::get_inputs_channels_number() const
{
    return inputs_dimensions(2);
}


/// Returns the number of neurons

Index FlattenLayer::get_neurons_number() const
{
    return inputs_dimensions(0) * inputs_dimensions(1) * inputs_dimensions(2);
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

}


void FlattenLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                          LayerBackPropagation* next_back_propagation,
                                          LayerForwardPropagation*,
                                          LayerBackPropagation* this_back_propagation) const
{
    FlattenLayerBackPropagation* flatten_layer_back_propagation =
            static_cast<FlattenLayerBackPropagation*>(this_back_propagation);

    Layer::Type next_type = next_back_propagation->layer->get_type();

    switch(next_type)
    {
    case Type::Perceptron:
    {
        PerceptronLayerForwardPropagation* next_perceptron_layer_forward_propagation =
                static_cast<PerceptronLayerForwardPropagation*>(next_forward_propagation);

        PerceptronLayerBackPropagation* next_perceptron_layer_back_propagation =
                static_cast<PerceptronLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_perceptron_layer_forward_propagation,
                               next_perceptron_layer_back_propagation,
                               flatten_layer_back_propagation);
    }
        return;

    case Type::Probabilistic:
    {
        ProbabilisticLayerForwardPropagation* next_probabilistic_layer_forward_propagation =
                static_cast<ProbabilisticLayerForwardPropagation*>(next_forward_propagation);

        ProbabilisticLayerBackPropagation* next_probabilistic_layer_back_propagation =
                static_cast<ProbabilisticLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_probabilistic_layer_forward_propagation,
                               next_probabilistic_layer_back_propagation,
                               flatten_layer_back_propagation);
    }
        return;

    default:

        return;
    }
}


void FlattenLayer::calculate_hidden_delta(PerceptronLayerForwardPropagation* next_perceptron_layer_forward_propagation,
                                          PerceptronLayerBackPropagation* next_perceptron_layer_back_propagation,
                                          FlattenLayerBackPropagation* flatten_layer_back_propagation) const
{
    const PerceptronLayer* next_perceptron_layer
        = static_cast<PerceptronLayer*>(next_perceptron_layer_back_propagation->layer);

    const Tensor<type, 2>& next_synaptic_weights = next_perceptron_layer->get_synaptic_weights();

    const Tensor<type, 2>& next_error_combinations_derivatives = next_perceptron_layer_back_propagation->error_combinations_derivatives;

    Tensor<type, 2>& deltas = flatten_layer_back_propagation->deltas;

    deltas.device(*thread_pool_device) = (next_error_combinations_derivatives).contract(next_synaptic_weights, A_BT);
}


void FlattenLayer::calculate_hidden_delta(ProbabilisticLayerForwardPropagation* next_probabilistic_layer_forward_propagation,
                                          ProbabilisticLayerBackPropagation* next_probabilistic_layer_back_propagation,
                                          FlattenLayerBackPropagation* flatten_layer_back_propagation) const
{
    const ProbabilisticLayer* next_probabilistic_layer
        = static_cast<ProbabilisticLayer*>(next_probabilistic_layer_back_propagation->layer);

    const Tensor<type, 2>& next_synaptic_weights = next_probabilistic_layer->get_synaptic_weights();

    const Tensor<type, 2>& next_error_combinations_derivatives = next_probabilistic_layer_back_propagation->error_combinations_derivatives;

    Tensor<type, 2>& deltas = flatten_layer_back_propagation->deltas;

    deltas.device(*thread_pool_device) = (next_error_combinations_derivatives).contract(next_synaptic_weights, A_BT);
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

    Tensor<Index, 1> inputs_dimension_tensor(3);

    inputs_dimension_tensor.setValues({input_height, input_width, input_channels_number});

    set(inputs_dimension_tensor);
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

pair<type*, dimensions> FlattenLayerBackPropagation::get_deltas_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(deltas_data, { batch_samples_number, neurons_number });
}

void FlattenLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = new_layer->get_neurons_number();

    deltas.resize(batch_samples_number, neurons_number);

    deltas_data = deltas.data();
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
