//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "flatten_layer.h"

namespace opennn
{

FlattenLayer::FlattenLayer() : Layer()
{
    layer_type = Layer::Type::Flatten;
    name = "flatten_layer";
}


FlattenLayer::FlattenLayer(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);

    layer_type = Type::Flatten;
    name = "flatten_layer";
}


dimensions FlattenLayer::get_input_dimensions() const
{
    return input_dimensions;
}


Index FlattenLayer::get_outputs_number() const
{
    return input_dimensions[0] * input_dimensions[1] * input_dimensions[2];
}


dimensions FlattenLayer::get_output_dimensions() const
{
    return { input_dimensions[0] * input_dimensions[1] * input_dimensions[2] };
}


Index FlattenLayer::get_inputs_number() const
{
    return input_dimensions[0] * input_dimensions[1] * input_dimensions[2];
}


Index FlattenLayer::get_input_height() const
{
    return input_dimensions[0];
}


Index FlattenLayer::get_input_width() const
{
    return input_dimensions[1];
}


Index FlattenLayer::get_input_channels() const
{
    return input_dimensions[2];
}


Index FlattenLayer::get_neurons_number() const
{
    return input_dimensions[0]* input_dimensions[1] * input_dimensions[2];
}


void FlattenLayer::set(const dimensions& new_input_dimensions)
{
    name = "flatten_layer";

    input_dimensions = new_input_dimensions;
}


void FlattenLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     LayerForwardPropagation* layer_forward_propagation,
                                     const bool& is_training)
{
    const Index batch_samples_number = layer_forward_propagation->batch_samples_number;

    const Index neurons_number = get_neurons_number();

    FlattenLayerForwardPropagation* flatten_layer_forward_propagation =
            static_cast<FlattenLayerForwardPropagation*>(layer_forward_propagation);

    type* outputs_data = flatten_layer_forward_propagation->outputs.data();

    memcpy(outputs_data,
           input_pairs[0].first,
           batch_samples_number*neurons_number*sizeof(type));

    flatten_layer_forward_propagation->outputs = TensorMap<Tensor<type, 2>>(input_pairs[0].first, batch_samples_number, neurons_number);

}


void FlattenLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                            const vector<pair<type*, dimensions>>& deltas_pair,
                                            LayerForwardPropagation* forward_propagation,
                                            LayerBackPropagation* back_propagation) const
{
    const Index batch_samples_number = input_pairs[0].second[0];
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(deltas_pair[0]);

    // Back propagation

    FlattenLayerBackPropagation* flatten_layer_back_propagation =
        static_cast<FlattenLayerBackPropagation*>(back_propagation);

    Tensor<type, 4>& input_derivatives = flatten_layer_back_propagation->input_derivatives;

    memcpy(input_derivatives.data(),
           deltas_pair[0].first,
           Index(batch_samples_number * neurons_number * sizeof(type)));
}


void FlattenLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("FlattenLayer");

    file_stream.OpenElement("InputVariablesDimension");

    file_stream.OpenElement("InputHeight");
    file_stream.PushText(to_string(get_input_height()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputWidth");
    file_stream.PushText(to_string(get_input_width()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputChannels");
    file_stream.PushText(to_string(get_input_channels()).c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();

    file_stream.CloseElement();
}


void FlattenLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* flatten_layer_element = document.FirstChildElement("FlattenLayer");

    if(!flatten_layer_element)
        throw runtime_error("FlattenLayer element is nullptr.\n");

    // Flatten layer input variables dimenison

    const tinyxml2::XMLElement* input_dimensions_element = flatten_layer_element->FirstChildElement("InputDimensions");

    if(!input_dimensions_element)
        throw runtime_error("FlattenInputVariablesDimensions element is nullptr.\n");

    // Input height

    const tinyxml2::XMLElement* input_height_element = input_dimensions_element->NextSiblingElement("InputHeight");

    if(!input_height_element)
        throw runtime_error("FlattenInputHeight element is nullptr.\n");

    const Index input_height = Index(atoi(input_height_element->GetText()));

    // Input width

    const tinyxml2::XMLElement* input_width_element = input_dimensions_element->NextSiblingElement("InputWidth");

    if(!input_width_element)
        throw runtime_error("FlattenInputWidth element is nullptr.\n");

    const Index input_width = Index(atoi(input_width_element->GetText()));

    // Input channels number

    const tinyxml2::XMLElement* input_channels_number_element = input_dimensions_element->NextSiblingElement("InputChannels");

    if(!input_channels_number_element)
        throw runtime_error("FlattenInputChannelsNumber element is nullptr.\n");

    const Index input_channels = Index(atoi(input_channels_number_element->GetText()));

    set({input_height, input_width, input_channels, 0});
}


void FlattenLayer::print() const
{
    cout << "Flatten layer" << endl;

    cout << "Input dimensions: " << endl;
    print_dimensions(input_dimensions);

    cout << "Output dimensions: " << endl;
    print_dimensions(get_output_dimensions());
}


pair<type*, dimensions> FlattenLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return {outputs_data, {batch_samples_number, neurons_number}};
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

    const FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(layer);

    const dimensions input_dimensions = flatten_layer->get_input_dimensions();

    input_derivatives.resize(batch_samples_number,
                             input_dimensions[0],
                             input_dimensions[1],
                             input_dimensions[2]);
}


vector<pair<type*, dimensions>> FlattenLayerBackPropagation::get_input_derivative_pairs() const
{
    const FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(layer);

    const dimensions input_dimensions = flatten_layer->get_input_dimensions();

    return {{(type*)(input_derivatives.data()),
            {batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2]}}};
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
