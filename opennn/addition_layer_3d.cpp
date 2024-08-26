//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//#include <cstdlib>

#include "addition_layer_3d.h"

namespace opennn
{


AdditionLayer3D::AdditionLayer3D() : Layer()
{
    set();

    layer_type = Type::Addition3D;
}


AdditionLayer3D::AdditionLayer3D(const Index& new_inputs_number, const Index& new_inputs_depth) : Layer()
{
    set(new_inputs_number, new_inputs_depth);

    layer_type = Type::Addition3D;

    name = "addition_layer_3d";
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


const bool& AdditionLayer3D::get_display() const
{
    return display;
}


void AdditionLayer3D::set()
{
    set_default();
}


void AdditionLayer3D::set(const Index& new_inputs_number, const Index& new_inputs_depth)
{
    inputs_number = new_inputs_number;

    inputs_depth = new_inputs_depth;

    set_default();
}


void AdditionLayer3D::set_default()
{
    name = "addition_layer_3d";

    display = true;

    layer_type = Type::Addition3D;
}


void AdditionLayer3D::set_name(const string& new_layer_name)
{
    name = new_layer_name;
}


void AdditionLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    inputs_depth = new_inputs_depth;
}


void AdditionLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


void AdditionLayer3D::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                        LayerForwardPropagation* layer_forward_propagation,
                                        const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> input_1(inputs_pair(0).first,
                                             inputs_pair(0).second[0],
                                             inputs_pair(0).second[1],
                                             inputs_pair(0).second[2]);

    const TensorMap<Tensor<type, 3>> input_2(inputs_pair(1).first,
                                             inputs_pair(1).second[0],
                                             inputs_pair(1).second[1],
                                             inputs_pair(1).second[2]);

    AdditionLayer3DForwardPropagation* addition_layer_3d_forward_propagation =
        static_cast<AdditionLayer3DForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 3>& outputs = addition_layer_3d_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = input_1 + input_2;
    
}


void AdditionLayer3D::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                               const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                               LayerForwardPropagation* forward_propagation,
                                               LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> deltas(deltas_pair(0).first,
                                            deltas_pair(0).second[0],
                                            deltas_pair(0).second[1],
                                            deltas_pair(0).second[2]);

    // Back propagation

    AdditionLayer3DBackPropagation* addition_layer_3d_back_propagation =
        static_cast<AdditionLayer3DBackPropagation*>(back_propagation);

    Tensor<type, 3>& input_1_derivatives = addition_layer_3d_back_propagation->input_1_derivatives;

    Tensor<type, 3>& input_2_derivatives = addition_layer_3d_back_propagation->input_2_derivatives;

    input_1_derivatives.device(*thread_pool_device) = deltas;
    input_2_derivatives.device(*thread_pool_device) = deltas;
}


void AdditionLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    // Addition layer

    const tinyxml2::XMLElement* addition_layer_element = document.FirstChildElement("AdditionLayer3D");

    if(!addition_layer_element)
        throw runtime_error("AdditionLayer3D element is nullptr.\n");

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = addition_layer_element->FirstChildElement("LayerName");

    if(!layer_name_element)
        throw runtime_error("LayerName element is nullptr.\n");

    if(layer_name_element->GetText())
    {
        set_name(layer_name_element->GetText());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = addition_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("InputsNumber element is nullptr.\n");

    if(inputs_number_element->GetText())
    {
        inputs_number = Index(stoi(inputs_number_element->GetText()));
    }

    // Inputs depth

    const tinyxml2::XMLElement* inputs_depth_element = addition_layer_element->FirstChildElement("InputsDepth");

    if(!inputs_depth_element)
        throw runtime_error("InputsDepth element is nullptr.\n");

    if(inputs_depth_element->GetText())
    {
        inputs_depth = Index(stoi(inputs_depth_element->GetText()));
    }
}


void AdditionLayer3D::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Addition layer

    file_stream.OpenElement("AdditionLayer3D");

    // Layer name

    file_stream.OpenElement("LayerName");   
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Inputs number

    file_stream.OpenElement("InputsNumber");
    file_stream.PushText(to_string(get_inputs_number()).c_str());
    file_stream.CloseElement();

    // Inputs depth

    file_stream.OpenElement("InputsDepth");
    file_stream.PushText(to_string(get_inputs_depth()).c_str());
    file_stream.CloseElement();

    // Addition layer (end tag)

    file_stream.CloseElement();
}



pair<type*, dimensions> AdditionLayer3DForwardPropagation::get_outputs_pair() const
{
    AdditionLayer3D* addition_layer_3d = static_cast<AdditionLayer3D*>(layer);

    const Index inputs_number = addition_layer_3d->get_inputs_number();
    const Index inputs_depth = addition_layer_3d->get_inputs_depth();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, inputs_number, inputs_depth });
}

void AdditionLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    AdditionLayer3D* addition_layer_3d = static_cast<AdditionLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = addition_layer_3d->get_inputs_number();
    const Index inputs_depth = addition_layer_3d->get_inputs_depth();

    outputs.resize(batch_samples_number, inputs_number, inputs_depth);

    outputs_data = outputs.data();
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

    inputs_derivatives.resize(2);
    inputs_derivatives(0).first = input_1_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, inputs_number, inputs_depth };

    inputs_derivatives(1).first = input_2_derivatives.data();
    inputs_derivatives(1).second = { batch_samples_number, inputs_number, inputs_depth };
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
