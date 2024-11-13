//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   4 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "scaling_layer_4d.h"

namespace opennn
{

ScalingLayer4D::ScalingLayer4D(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions ScalingLayer4D::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions ScalingLayer4D::get_output_dimensions() const
{
    return input_dimensions;
}


void ScalingLayer4D::set(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;

    set_min_max_range(type(0), type(255));

    layer_type = Type::Scaling4D;

    name = "scaling_layer_4d";
}


void ScalingLayer4D::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


bool ScalingLayer4D::is_empty() const
{
    const Index inputs_number = get_output_dimensions()[0];

    return inputs_number == 0;
}


void ScalingLayer4D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       const bool& is_training)
{
    ScalingLayer4DForwardPropagation* scaling_layer_forward_propagation =
        static_cast<ScalingLayer4DForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

    Tensor<type, 4>& outputs = scaling_layer_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs/type(255); 
}


void ScalingLayer4D::print() const
{
    cout << "Scaling layer 4D" << endl;

    print_vector(input_dimensions);
}


void ScalingLayer4D::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Scaling4D");

    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));

    printer.CloseElement();
}


void ScalingLayer4D::from_XML(const XMLDocument& document)
{
    const XMLElement* scaling_layer_element = document.FirstChildElement("Scaling4D");

    if(!scaling_layer_element)
        throw runtime_error("Scaling layer element is nullptr.\n");

    set(string_to_dimensions(read_xml_string(scaling_layer_element, "InputDimensions")));

}


ScalingLayer4DForwardPropagation::ScalingLayer4DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> ScalingLayer4DForwardPropagation::get_outputs_pair() const
{
    const ScalingLayer4D* scaling_layer_4d = static_cast<ScalingLayer4D*>(layer);

    const dimensions output_dimensions = scaling_layer_4d->get_output_dimensions();

    return {(type*)outputs.data(), {batch_samples_number, output_dimensions[0], output_dimensions[1], output_dimensions[2]}};
}


void ScalingLayer4DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_samples_number, output_dimensions[0], output_dimensions[1], output_dimensions[2]);
}


void ScalingLayer4DForwardPropagation::print() const
{
    cout << "Scaling Outputs:" << endl
         << outputs.dimensions() << endl;
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
