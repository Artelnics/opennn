//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   3D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "flatten_layer_3d.h"

namespace opennn
{

Flatten3D::Flatten3D(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions Flatten3D::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Flatten3D::get_output_dimensions() const
{
    return { input_dimensions[0] * input_dimensions[1] };
}


Index Flatten3D::get_input_height() const
{
    return input_dimensions[0];
}


Index Flatten3D::get_input_width() const
{
    return input_dimensions[1];
}


void Flatten3D::set(const dimensions& new_input_dimensions)
{
    layer_type = Type::Flatten3D;

    set_name("flatten_layer");

    input_dimensions = new_input_dimensions;
}


void Flatten3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                  const bool&)
{
    const Index samples_number = layer_forward_propagation->batch_size;

    const Index outputs_number = get_outputs_number();

    FlattenLayer3DForwardPropagation* flatten_layer_3d_forward_propagation =
        static_cast<FlattenLayer3DForwardPropagation*>(layer_forward_propagation.get());

    flatten_layer_3d_forward_propagation->outputs = TensorMap<Tensor<type, 2>>(input_pairs[0].first, samples_number, outputs_number);
}


void Flatten3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>&,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_pairs[0].second[0];
    const Index outputs_number = get_outputs_number();

    // Back propagation

    FlattenLayer3DBackPropagation* flatten_layer_3d_back_propagation =
        static_cast<FlattenLayer3DBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_derivatives = flatten_layer_3d_back_propagation->input_derivatives;

    memcpy(input_derivatives.data(),
           delta_pairs[0].first,
           (batch_size * outputs_number * sizeof(type)));
}


void Flatten3D::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Flatten3D");

    add_xml_element(printer, "InputHeight", to_string(get_input_height()));
    add_xml_element(printer, "InputWidth", to_string(get_input_width()));

    printer.CloseElement();
}


void Flatten3D::from_XML(const XMLDocument& document)
{
    const XMLElement* flatten_layer_3d_element = document.FirstChildElement("Flatten3D");

    if (!flatten_layer_3d_element)
        throw runtime_error("Flatten3D element is nullptr.\n");

    const Index input_height = read_xml_index(flatten_layer_3d_element, "InputHeight");
    const Index input_width = read_xml_index(flatten_layer_3d_element, "InputWidth");

    set({ input_height, input_width});
}


void Flatten3D::print() const
{
    cout << "Flatten3D layer" << endl;

    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);

    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


FlattenLayer3DForwardPropagation::FlattenLayer3DForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> FlattenLayer3DForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
}


void FlattenLayer3DForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_size, output_dimensions[0]);
}


void FlattenLayer3DForwardPropagation::print() const
{
    cout << "Flatten3D Outputs:" << endl
         << outputs.dimensions() << endl;
}


void FlattenLayer3DBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    batch_size = new_batch_size;

    const Flatten3D* flatten_layer_3d = static_cast<Flatten3D*>(layer);

    const dimensions input_dimensions = flatten_layer_3d->get_input_dimensions();

    input_derivatives.resize(batch_size,
                             input_dimensions[0],
                             input_dimensions[1]);
}


void FlattenLayer3DBackPropagation::print() const
{
}


FlattenLayer3DBackPropagation::FlattenLayer3DBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> FlattenLayer3DBackPropagation::get_input_derivative_pairs() const
{
    const Flatten3D* flatten_layer_3d = static_cast<Flatten3D*>(layer);

    const dimensions input_dimensions = flatten_layer_3d->get_input_dimensions();

    return {{(type*)(input_derivatives.data()),
             {batch_size, input_dimensions[0], input_dimensions[1]}}};
}


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
