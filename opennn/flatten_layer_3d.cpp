//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   3D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "flatten_layer_3d.h"

namespace opennn
{

Flatten3d::Flatten3d(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions Flatten3d::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Flatten3d::get_output_dimensions() const
{
    return { input_dimensions[0] * input_dimensions[1] };
}


Index Flatten3d::get_input_height() const
{
    return input_dimensions[0];
}


Index Flatten3d::get_input_width() const
{
    return input_dimensions[1];
}


void Flatten3d::set(const dimensions& new_input_dimensions)
{
    name = "Flatten3d";

    set_label("flatten_layer");

    input_dimensions = new_input_dimensions;
}


void Flatten3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                  const bool&)
{
    const Index batch_size = layer_forward_propagation->batch_size;

    const Index outputs_number = get_outputs_number();

    Flatten3dForwardPropagation* flatten_layer_3d_forward_propagation =
        static_cast<Flatten3dForwardPropagation*>(layer_forward_propagation.get());

    flatten_layer_3d_forward_propagation->outputs = TensorMap<Tensor<type, 2>>(input_pairs[0].first, batch_size, outputs_number);
}


void Flatten3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>&,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_pairs[0].second[0];
    const Index outputs_number = get_outputs_number();

    // Back propagation

    Flatten3dBackPropagation* flatten_layer_3d_back_propagation =
        static_cast<Flatten3dBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_deltas = flatten_layer_3d_back_propagation->input_deltas;

    memcpy(input_deltas.data(),
           delta_pairs[0].first,
           (batch_size * outputs_number * sizeof(type)));
}


void Flatten3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Flatten3d");

    add_xml_element(printer, "InputHeight", to_string(get_input_height()));
    add_xml_element(printer, "InputWidth", to_string(get_input_width()));

    printer.CloseElement();
}


void Flatten3d::from_XML(const XMLDocument& document)
{
    const XMLElement* flatten_layer_3d_element = document.FirstChildElement("Flatten3d");

    if (!flatten_layer_3d_element)
        throw runtime_error("Flatten3d element is nullptr.\n");

    const Index input_height = read_xml_index(flatten_layer_3d_element, "InputHeight");
    const Index input_width = read_xml_index(flatten_layer_3d_element, "InputWidth");

    set({ input_height, input_width});
}


void Flatten3d::print() const
{
    cout << "Flatten3d layer" << endl;

    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);

    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


Flatten3dForwardPropagation::Flatten3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Flatten3dForwardPropagation::get_output_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
}


void Flatten3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_size, output_dimensions[0]);
}


void Flatten3dForwardPropagation::print() const
{
    cout << "Flatten3d Outputs:" << endl
         << outputs.dimensions() << endl;
}


void Flatten3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    if (!layer) return;

    const Flatten3d* flatten_layer_3d = static_cast<Flatten3d*>(layer);

    const dimensions input_dimensions = flatten_layer_3d->get_input_dimensions();

    input_deltas.resize(batch_size,
                             input_dimensions[0],
                             input_dimensions[1]);
}


void Flatten3dBackPropagation::print() const
{
}


Flatten3dBackPropagation::Flatten3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Flatten3dBackPropagation::get_input_derivative_pairs() const
{
    const Flatten3d* flatten_layer_3d = static_cast<Flatten3d*>(layer);

    const dimensions input_dimensions = flatten_layer_3d->get_input_dimensions();

    return {{(type*)(input_deltas.data()),
             {batch_size, input_dimensions[0], input_dimensions[1]}}};
}

REGISTER(Layer, Flatten3d, "Flatten3d")
REGISTER_FORWARD_PROPAGATION("Flatten3d", Flatten3dForwardPropagation);
REGISTER_BACK_PROPAGATION("Flatten3d", Flatten3dBackPropagation);

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
