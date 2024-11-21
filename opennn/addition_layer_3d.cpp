//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#include "tensors.h"
#include "addition_layer_3d.h"

namespace opennn
{

AdditionLayer3D::AdditionLayer3D(const Index& new_inputs_number, const Index& new_inputs_depth) : Layer()
{
    set(new_inputs_number, new_inputs_depth);
}


Index AdditionLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index AdditionLayer3D::get_inputs_depth() const
{
    return inputs_depth;
}


dimensions AdditionLayer3D::get_output_dimensions() const
{
    return { inputs_number, inputs_depth };
}


void AdditionLayer3D::set(const Index& new_inputs_number, const Index& new_inputs_depth)
{
    inputs_number = new_inputs_number;

    inputs_depth = new_inputs_depth;

    name = "addition_layer_3d";

    layer_type = Type::Addition3D;
}


void AdditionLayer3D::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void AdditionLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    inputs_depth = new_inputs_depth;
}


void AdditionLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                        const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> input_1 = tensor_map_3(input_pairs[0]);
    
    const TensorMap<Tensor<type, 3>> input_2 = tensor_map_3(input_pairs[1]);

    AdditionLayer3DForwardPropagation* addition_layer_3d_forward_propagation =
        static_cast<AdditionLayer3DForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = addition_layer_3d_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = input_1 + input_2;    
}


void AdditionLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     const vector<pair<type*, dimensions>>& delta_pairs,
                                     unique_ptr<LayerForwardPropagation>& forward_propagation,
                                     unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Back propagation

    AdditionLayer3DBackPropagation* addition_layer_3d_back_propagation =
        static_cast<AdditionLayer3DBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_1_derivatives = addition_layer_3d_back_propagation->input_1_derivatives;

    Tensor<type, 3>& input_2_derivatives = addition_layer_3d_back_propagation->input_2_derivatives;

    input_1_derivatives.device(*thread_pool_device) = deltas;
    input_2_derivatives.device(*thread_pool_device) = deltas;
}


void AdditionLayer3D::from_XML(const XMLDocument& document)
{
    const auto* addition_layer_element = document.FirstChildElement("Addition3D");

    if (!addition_layer_element)
        throw runtime_error("Addition3D element is nullptr.\n");

    set_name(read_xml_string(addition_layer_element, "Name"));    
    set_inputs_number(read_xml_index(addition_layer_element, "InputsNumber"));
    set_inputs_depth(read_xml_index(addition_layer_element, "InputsDepth"));
}


void AdditionLayer3D::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Addition3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_inputs_number()));
    add_xml_element(printer, "InputsDepth", to_string(get_inputs_depth()));

    printer.CloseElement();
}


AdditionLayer3DForwardPropagation::AdditionLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> AdditionLayer3DForwardPropagation::get_outputs_pair() const
{
    AdditionLayer3D* addition_layer_3d = static_cast<AdditionLayer3D*>(layer);

    const Index inputs_number = addition_layer_3d->get_inputs_number();
    const Index inputs_depth = addition_layer_3d->get_inputs_depth();

    return {(type*)outputs.data(), {batch_samples_number, inputs_number, inputs_depth}};
}


void AdditionLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    AdditionLayer3D* addition_layer_3d = static_cast<AdditionLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = addition_layer_3d->get_inputs_number();
    const Index inputs_depth = addition_layer_3d->get_inputs_depth();

    outputs.resize(batch_samples_number, inputs_number, inputs_depth);
}


void AdditionLayer3DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


void AdditionLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    AdditionLayer3D* addition_layer_3d = static_cast<AdditionLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = addition_layer_3d->get_inputs_number();
    const Index inputs_depth = addition_layer_3d->get_inputs_depth();

    input_1_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
    input_2_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
}


void AdditionLayer3DBackPropagation::print() const
{
}


AdditionLayer3DBackPropagation::AdditionLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> AdditionLayer3DBackPropagation::get_input_derivative_pairs() const
{
    AdditionLayer3D* addition_layer_3d = static_cast<AdditionLayer3D*>(layer);

    const Index inputs_number = addition_layer_3d->get_inputs_number();
    const Index inputs_depth = addition_layer_3d->get_inputs_depth();

    return
    {{(type*)input_1_derivatives.data(), {batch_samples_number, inputs_number, inputs_depth}},
     {(type*)input_2_derivatives.data(), {batch_samples_number, inputs_number, inputs_depth}}};
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
