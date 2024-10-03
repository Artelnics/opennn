//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   4 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "scaling_layer_4d.h"
#include "strings_utilities.h"

namespace opennn
{

ScalingLayer4D::ScalingLayer4D() : Layer()
{    
    set();
}


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


Index ScalingLayer4D::get_inputs_number() const
{
    return input_dimensions[0]*input_dimensions[1]*input_dimensions[2];
}


Index ScalingLayer4D::get_neurons_number() const
{
    return 0;
}


const bool& ScalingLayer4D::get_display() const
{
    return display;
}


void ScalingLayer4D::set()
{
    set_default();
}


void ScalingLayer4D::set(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;

    set_default();
}


void ScalingLayer4D::set(const tinyxml2::XMLDocument& new_scaling_layer_document)
{
    set_default();

    from_XML(new_scaling_layer_document);
}


void ScalingLayer4D::set_default()
{
    set_min_max_range(type(0), type(255));

    display = true;

    layer_type = Type::Scaling4D;

    name = "scaling_layer";
}


void ScalingLayer4D::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


void ScalingLayer4D::set_display(const bool& new_display)
{
    display = new_display;
}


bool ScalingLayer4D::is_empty() const
{
    const Index inputs_number = get_neurons_number();

    return inputs_number == 0;
}


void ScalingLayer4D::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                     LayerForwardPropagation* forward_propagation,
                                     const bool& is_training)
{
    ScalingLayer4DForwardPropagation* scaling_layer_forward_propagation
            = static_cast<ScalingLayer4DForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(inputs_pair(0));
    
    Tensor<type, 4>& outputs = scaling_layer_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs/type(255); 
}


void ScalingLayer4D::print() const
{
    cout << "Scaling layer 4D" << endl;

    print_dimensions(input_dimensions);
}


void ScalingLayer4D::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    // Scaling layer

    file_stream.OpenElement("Scaling4D");

    // Scaling neurons number

    file_stream.OpenElement("NeuronsNumber");
    file_stream.PushText(to_string(neurons_number).c_str());
    file_stream.CloseElement();

    // Scaling neurons

    for(Index i = 0; i < neurons_number; i++)
    {
        // Scaling neuron

        file_stream.OpenElement("ScalingNeuron");
        file_stream.PushAttribute("Index", int(i+1));

        // Scaling neuron (end tag)

        file_stream.CloseElement();
    }

    // Scaling layer (end tag)

    file_stream.CloseElement();
}


void ScalingLayer4D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* scaling_layer_element = document.FirstChildElement("Scaling4D");

    if(!scaling_layer_element)
        throw runtime_error("Scaling layer element is nullptr.\n");

    // Scaling neurons number

    const tinyxml2::XMLElement* neurons_number_element = scaling_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("Scaling neurons number element is nullptr.\n");

    const Index neurons_number = Index(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        start_element = scaling_neuron_element;

        if(!scaling_neuron_element)
            throw runtime_error("Scaling neuron " + to_string(i+1) + " is nullptr.\n");

        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");


        // Scaling method

        const tinyxml2::XMLElement* scaling_method_element = scaling_neuron_element->FirstChildElement("Scaler");

        if(!scaling_method_element)
            throw runtime_error("Scaling method element " + to_string(i+1) + " is nullptr.\n");
    }

    // Display

    const tinyxml2::XMLElement* display_element = scaling_layer_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


pair<type*, dimensions> ScalingLayer4DForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, neurons_number, 1, 1 });
}


void ScalingLayer4DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_samples_number, output_dimensions[0], output_dimensions[1], output_dimensions[2]);

    outputs_data = outputs.data();
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
