//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


#include "addition_layer_3d.h"
#include "normalization_layer_3d.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.


AdditionLayer3D::AdditionLayer3D() : Layer()
{
    set();

    layer_type = Type::Addition3D;
}


/// Layer architecture constructor.
/// It creates a layer object with given numbers of inputs and perceptrons.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of perceptrons in the layer.

AdditionLayer3D::AdditionLayer3D(const Index& new_inputs_number, const Index& new_inputs_depth) : Layer()
{
    set(new_inputs_number, new_inputs_depth);

    layer_type = Type::Addition3D;

    layer_name = "addition_layer_3d";
}


/// Returns the number of inputs to the layer.

Index AdditionLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index AdditionLayer3D::get_inputs_depth() const
{
    return inputs_depth;
}


dimensions AdditionLayer3D::get_outputs_dimensions() const
{
    return { inputs_number, inputs_depth };
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& AdditionLayer3D::get_display() const
{
    return display;
}


/// Sets an empty layer, wihtout any perceptron.
/// It also sets the rest of the members to their default values.

void AdditionLayer3D::set()
{
    set_default();
}


/// Sets new numbers of inputs and perceptrons in the layer.
/// It also sets the rest of the members to their default values.
/// @param new_inputs_number Number of inputs.
/// @param new_neurons_number Number of perceptron neurons.

void AdditionLayer3D::set(const Index& new_inputs_number, const Index& new_inputs_depth)
{
    inputs_number = new_inputs_number;

    inputs_depth = new_inputs_depth;

    set_default();
}


/// Sets those members not related to the vector of perceptrons to their default value.
/// <ul>
/// <li> Display: True.
/// <li> layer_type: Perceptron_Layer.
/// <li> trainable: True.
/// </ul>

void AdditionLayer3D::set_default()
{
    layer_name = "addition_layer_3d";

    display = true;

    layer_type = Type::Addition3D;
}


void AdditionLayer3D::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new number of inputs in the layer.
/// It also initializes the new synaptic weights at random.
/// @param new_inputs_number Number of layer inputs.

void AdditionLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    inputs_depth = new_inputs_depth;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void AdditionLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


void AdditionLayer3D::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                        LayerForwardPropagation* layer_forward_propagation,
                                        const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> input_1(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1], inputs_pair(0).second[2]);
    const TensorMap<Tensor<type, 3>> input_2(inputs_pair(1).first, inputs_pair(1).second[0], inputs_pair(1).second[1], inputs_pair(1).second[2]);

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
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("AdditionLayer3D");

    if (!perceptron_layer_element)
    {
        buffer << "OpenNN Exception: AdditionLayer3D class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "AdditionLayer3D element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = perceptron_layer_element->FirstChildElement("LayerName");

    if (!layer_name_element)
    {
        buffer << "OpenNN Exception: AdditionLayer3D class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "LayerName element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if (layer_name_element->GetText())
    {
        set_name(layer_name_element->GetText());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

    if (!inputs_number_element)
    {
        buffer << "OpenNN Exception: AdditionLayer3D class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "InputsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if (inputs_number_element->GetText())
    {
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

    if (!neurons_number_element)
    {
        buffer << "OpenNN Exception: AdditionLayer3D class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "NeuronsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if (neurons_number_element->GetText())
    {
        set_neurons_number(Index(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

    if (!activation_function_element)
    {
        buffer << "OpenNN Exception: AdditionLayer3D class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "ActivationFunction element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

    if (!parameters_element)
    {
        buffer << "OpenNN Exception: AdditionLayer3D class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "Parameters element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if (parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();
        /*
        set_parameters(to_type_vector(parameters_string, ' '));
        */
    }
}


void AdditionLayer3D::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("AdditionLayer3D");

    // Layer name
    file_stream.OpenElement("LayerName");
    buffer.str("");
    buffer << layer_name;
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Inputs number
    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << get_inputs_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs number

    file_stream.OpenElement("NeuronsNumber");

    buffer.str("");
    buffer << get_neurons_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");

    buffer.str("");

    const Tensor<type, 1> parameters = get_parameters();
    const Index parameters_size = parameters.size();

    for (Index i = 0; i < parameters_size; i++)
    {
        buffer << parameters(i);

        if (i != (parameters_size - 1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Peceptron layer (end tag)

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
