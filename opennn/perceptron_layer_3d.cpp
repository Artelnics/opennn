//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer_3d.h"
#include "tensors.h"
#include "strings_utilities.h"

namespace opennn
{

Perceptron3d::Perceptron3d(const Index& new_sequence_length,
                           const Index& new_input_dimension,
                           const Index& new_output_dimension,
                           const Perceptron3d::Activation& new_activation_function,
                           const string& new_name) : Layer()
{
    layer_type = Type::Perceptron3D;

    set(new_sequence_length, new_input_dimension, new_output_dimension, new_activation_function, new_name);
}


Index Perceptron3d::get_sequence_length() const
{
    return sequence_length;
}


Index Perceptron3d::get_input_dimension() const
{
    return weights.dimension(0);
}


void Perceptron3d::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


Index Perceptron3d::get_output_dimension() const
{
    return biases.size();
}


dimensions Perceptron3d::get_output_dimensions() const
{
    return { sequence_length, get_output_dimension() };
}


Index Perceptron3d::get_parameters_number() const
{
    return biases.size() + weights.size();
}


type Perceptron3d::get_dropout_rate() const
{
    return dropout_rate;
}


Tensor<type, 1> Perceptron3d::get_parameters() const
{
    Tensor<type, 1> parameters(weights.size() + biases.size());

    Index index = 0;

    copy_to_vector(parameters, weights, index);
    copy_to_vector(parameters, biases, index);

    return parameters;
}


const Perceptron3d::Activation& Perceptron3d::get_activation_function() const
{
    return activation_function;
}


string Perceptron3d::get_activation_function_string() const
{
    switch(activation_function)
    {
    case Activation::HyperbolicTangent:
        return "HyperbolicTangent";

    case Activation::Linear:
        return "Linear";

    case Activation::RectifiedLinear:
        return "RectifiedLinear";
    }

    return string();
}


void Perceptron3d::set(const Index& new_sequence_length,
                       const Index& new_input_dimension, 
                       const Index& new_output_dimension,
                       const Perceptron3d::Activation& new_activation_function,
                       const string& new_name)
{
    sequence_length = new_sequence_length;

    biases.resize(new_output_dimension);

    weights.resize(new_input_dimension, new_output_dimension);

    set_parameters_glorot();

    activation_function = new_activation_function;

    name = new_name;

    layer_type = Type::Perceptron3D;

    dropout_rate = 0;
}


void Perceptron3d::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(weights, new_parameters, index);
    copy_from_vector(biases, new_parameters, index);
}


void Perceptron3d::set_activation_function(const Perceptron3d::Activation& new_activation_function)
{
    activation_function = new_activation_function;
}


void Perceptron3d::set_activation_function(const string& new_activation_function_name)
{
    if (new_activation_function_name == "HyperbolicTangent")
        activation_function = Activation::HyperbolicTangent;
    else if (new_activation_function_name == "Linear")
        activation_function = Activation::Linear;
    else if (new_activation_function_name == "RectifiedLinear")
        activation_function = Activation::RectifiedLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void Perceptron3d::set_parameters_constant(const type& value)
{
    biases.setConstant(value);
    weights.setConstant(value);
}


void Perceptron3d::set_parameters_random()
{
    set_random(biases);
    set_random(weights);
}


void Perceptron3d::set_parameters_glorot()
{
    biases.setZero();

    const type limit = sqrt(6 / type(get_input_dimension() + get_output_dimension()));

    set_random(weights, -limit, limit);
}


void Perceptron3d::calculate_combinations(const Tensor<type, 3>& inputs,
                                          Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(weights, contraction_indices);

    sum_matrices(thread_pool_device.get(), biases, combinations);
}


void Perceptron3d::dropout(Tensor<type, 3>& outputs) const 
{
    /*
    type* outputs_data = outputs.data();

    const Index samples_number = outputs.dimension(0);
    const Index inputs_number = outputs.dimension(1);
    const Index outputs_number = outputs.dimension(2);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        TensorMap<Tensor<type, 2>> matrix(outputs_data + neuron_index*batch_size*inputs_number,
                                          batch_size, inputs_number);

        random = get_random(type(0), type(1));

        if(random < dropout_rate)
            matrix.setZero();
        else
            matrix.device(*thread_pool_device) = matrix * scaling_factor;
    }
    */
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    #pragma omp parallel for

    for(Index i = 0; i < outputs.size(); i++)
        outputs(i) = (get_random_type(type(0), type(1)) < dropout_rate)
            ? 0 
            : outputs(i) * scaling_factor;
}


void Perceptron3d::calculate_activations(Tensor<type, 3>& activations, Tensor<type, 3>& activation_derivatives) const
{
    switch(activation_function)
    {
    case Activation::Linear: linear(activations, activation_derivatives); return;
    case Activation::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;
    case Activation::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;
    case Activation::Logistic: logistic(activations, activation_derivatives); return;
    default: return;
    }
}


void Perceptron3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                     const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    Perceptron3dForwardPropagation* perceptron_layer_3d_forward_propagation =
        static_cast<Perceptron3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = perceptron_layer_3d_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if(is_training)
    {
        if(dropout_rate > type(0))
            dropout(outputs);

        Tensor<type, 3>& activation_derivatives = perceptron_layer_3d_forward_propagation->activation_derivatives;

        calculate_activations(outputs, activation_derivatives);

        cout << "Perceptron layer outputs dimensions: " << outputs.dimensions() << endl;
        cout << "Perceptron layer activation_derivatives dimensions: " << activation_derivatives.dimensions() << endl;

        cout << "Outputs:\n" << outputs.chip(0,2) << endl;

    }
    else
    {
        calculate_activations(outputs, empty);
    }
}


void Perceptron3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  const vector<pair<type*, dimensions>>& delta_pairs,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Forward propagation

    const Perceptron3dForwardPropagation* perceptron_layer_3d_forward_propagation =
            static_cast<Perceptron3dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& activation_derivatives = perceptron_layer_3d_forward_propagation->activation_derivatives;

    // Back propagation

    Perceptron3dBackPropagation* perceptron_3d_back_propagation =
            static_cast<Perceptron3dBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& combination_derivatives = perceptron_3d_back_propagation->combination_derivatives;

    Tensor<type, 1>& bias_derivatives = perceptron_3d_back_propagation->bias_derivatives;
    Tensor<type, 2>& weight_derivatives = perceptron_3d_back_propagation->weight_derivatives;

    Tensor<type, 3>& input_derivatives = perceptron_3d_back_propagation->input_derivatives;

    combination_derivatives.device(*thread_pool_device) 
        = deltas * activation_derivatives;

    bias_derivatives.device(*thread_pool_device)
        = combination_derivatives.sum(sum_dimensions);

    weight_derivatives.device(*thread_pool_device)
        = inputs.contract(combination_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = combination_derivatives.contract(weights, single_contraction_indices);
}


void Perceptron3d::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                   Index& index,
                                   Tensor<type, 1>& gradient) const
{
    Perceptron3dBackPropagation* perceptron_back_propagation =
        static_cast<Perceptron3dBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, perceptron_back_propagation->weight_derivatives, index);
    copy_to_vector(gradient, perceptron_back_propagation->bias_derivatives, index);
}


void Perceptron3d::from_XML(const XMLDocument& document)
{
    const XMLElement* perceptron_layer_element = document.FirstChildElement("Perceptron3D");

    if(!perceptron_layer_element)
        throw runtime_error("Perceptron3D element is nullptr.\n");

    const Index new_sequence_length = read_xml_index(perceptron_layer_element, "InputsNumber");
    const Index new_input_dimension = read_xml_index(perceptron_layer_element, "InputsDepth");
    const Index new_output_dimension = read_xml_index(perceptron_layer_element, "NeuronsNumber");

    set(new_sequence_length, new_input_dimension, new_output_dimension);

    set_name(read_xml_string(perceptron_layer_element, "Name"));
    set_activation_function(read_xml_string(perceptron_layer_element, "Activation"));

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(perceptron_layer_element, "Parameters"), " "), index);
}


void Perceptron3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Perceptron3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_sequence_length()));
    add_xml_element(printer, "InputsDepth", to_string(get_input_dimension()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimension()));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


void Perceptron3dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


Perceptron3dForwardPropagation::Perceptron3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Perceptron3dForwardPropagation::get_outputs_pair() const
{
    Perceptron3d* perceptron_layer_3d = static_cast<Perceptron3d*>(layer);

    const Index sequence_length = perceptron_layer_3d->get_sequence_length();
    const Index output_dimension = perceptron_layer_3d->get_output_dimension();

    return { (type*)outputs.data(), { batch_size, sequence_length, output_dimension } };
}


void Perceptron3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Perceptron3d* perceptron_layer_3d = static_cast<Perceptron3d*>(layer);

    batch_size = new_batch_size;

    const Index output_dimension = perceptron_layer_3d->get_output_dimension();

    const Index sequence_length = perceptron_layer_3d->get_sequence_length();

    outputs.resize(batch_size, sequence_length, output_dimension);

    activation_derivatives.resize(batch_size, sequence_length, output_dimension);
}


void Perceptron3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Perceptron3d* perceptron_layer_3d = static_cast<Perceptron3d*>(layer);

    batch_size = new_batch_size;

    const Index output_dimension = perceptron_layer_3d->get_output_dimension();
    const Index sequence_length = perceptron_layer_3d->get_sequence_length();
    const Index input_dimension = perceptron_layer_3d->get_input_dimension();

    bias_derivatives.resize(output_dimension);
    weight_derivatives.resize(input_dimension, output_dimension);
    combination_derivatives.resize(batch_size, sequence_length, output_dimension);
    input_derivatives.resize(batch_size, sequence_length, input_dimension);
}


void Perceptron3dBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << weight_derivatives << endl;
}


Perceptron3dBackPropagation::Perceptron3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Perceptron3dBackPropagation::get_input_derivative_pairs() const
{
    Perceptron3d* perceptron_layer_3d = static_cast<Perceptron3d*>(layer);

    const Index sequence_length = perceptron_layer_3d->get_sequence_length();
    const Index inputs_depth = perceptron_layer_3d->get_input_dimension();

    return {{(type*)(input_derivatives.data()),
            {batch_size, sequence_length, inputs_depth}}};
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
