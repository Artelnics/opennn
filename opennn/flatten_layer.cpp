//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#include "tensors.h"
#include "flatten_layer.h"

namespace opennn
{

FlattenLayer::FlattenLayer(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions FlattenLayer::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions FlattenLayer::get_output_dimensions() const
{
    return { input_dimensions[0] * input_dimensions[1] * input_dimensions[2] };
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


void FlattenLayer::set(const dimensions& new_input_dimensions)
{
    layer_type = Type::Flatten;

    set_name("flatten_layer");

    input_dimensions = new_input_dimensions;
}


void FlattenLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                     const bool&)
{
    const Index batch_samples_number = layer_forward_propagation->batch_samples_number;

    const Index outputs_number = get_outputs_number();

    FlattenLayerForwardPropagation* flatten_layer_forward_propagation =
            static_cast<FlattenLayerForwardPropagation*>(layer_forward_propagation.get());

    type* outputs_data = flatten_layer_forward_propagation->outputs.data();

    // cout<<"=========Flatten==========="<<endl;

    memcpy(outputs_data,
           input_pairs[0].first,
           batch_samples_number*outputs_number*sizeof(type));

    flatten_layer_forward_propagation->outputs = TensorMap<Tensor<type, 2>>(input_pairs[0].first, batch_samples_number, outputs_number);
}


void FlattenLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  const vector<pair<type*, dimensions>>& delta_pairs,
                                  unique_ptr<LayerForwardPropagation>&,
                                  unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_samples_number = input_pairs[0].second[0];
    const Index outputs_number = get_outputs_number();

    // Back propagation

    FlattenLayerBackPropagation* flatten_layer_back_propagation =
        static_cast<FlattenLayerBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& input_derivatives = flatten_layer_back_propagation->input_derivatives;

    memcpy(input_derivatives.data(),
           delta_pairs[0].first,
           Index(batch_samples_number * outputs_number * sizeof(type)));
}


void FlattenLayer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Flatten");

    add_xml_element(printer, "InputHeight", to_string(get_input_height()));
    add_xml_element(printer, "InputWidth", to_string(get_input_width()));
    add_xml_element(printer, "InputChannels", to_string(get_input_channels()));

    printer.CloseElement(); 
}


void FlattenLayer::from_XML(const XMLDocument& document)
{
    const XMLElement* flatten_layer_element = document.FirstChildElement("Flatten");

    if (!flatten_layer_element) 
        throw runtime_error("FlattenLayer element is nullptr.\n");

    const Index input_height = read_xml_index(flatten_layer_element, "InputHeight");
    const Index input_width = read_xml_index(flatten_layer_element, "InputWidth");
    const Index input_channels = read_xml_index(flatten_layer_element, "InputChannels");

    set({ input_height, input_width, input_channels });
}


void FlattenLayer::print() const
{
    cout << "Flatten layer" << endl;

    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);

    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


FlattenLayerForwardPropagation::FlattenLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> FlattenLayerForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {batch_samples_number, output_dimensions[0]}};
}


void FlattenLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_samples_number, output_dimensions[0]);
}


void FlattenLayerForwardPropagation::print() const
{
    cout << "Flatten Outputs:" << endl
         << outputs.dimensions() << endl;
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


void FlattenLayerBackPropagation::print() const
{
}


FlattenLayerBackPropagation::FlattenLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
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
