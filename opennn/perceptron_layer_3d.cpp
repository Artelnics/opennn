//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "strings_utilities.h"
#include "perceptron_layer_3d.h"
#include "tensors.h"

namespace opennn
{

PerceptronLayer3D::PerceptronLayer3D(const Index& new_inputs_number,
                                     const Index& new_inputs_depth,
                                     const Index& new_neurons_number,
                                     const PerceptronLayer3D::ActivationFunction& new_activation_function) : Layer()
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number, new_activation_function);

    layer_type = Type::Perceptron3D;

    name = "perceptron_layer_3d";
}


Index PerceptronLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index PerceptronLayer3D::get_inputs_depth() const
{
    return synaptic_weights.dimension(0);
}


void PerceptronLayer3D::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


Index PerceptronLayer3D::get_neurons_number() const
{
    return biases.size();
}


dimensions PerceptronLayer3D::get_output_dimensions() const
{
    return { inputs_number, get_neurons_number() };
}


Index PerceptronLayer3D::get_biases_number() const
{
    return biases.size();
}


Index PerceptronLayer3D::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


Index PerceptronLayer3D::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


type PerceptronLayer3D::get_dropout_rate() const
{
    return dropout_rate;
}


Tensor<type, 1> PerceptronLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    memcpy(parameters.data(), synaptic_weights.data(), synaptic_weights.size()*sizeof(type));

    memcpy(parameters.data() + synaptic_weights.size(), biases.data(), biases.size()*sizeof(type));

    return parameters;
}


const PerceptronLayer3D::ActivationFunction& PerceptronLayer3D::get_activation_function() const
{
    return activation_function;
}


string PerceptronLayer3D::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case ActivationFunction::Linear:
        return "Linear";

    case ActivationFunction::RectifiedLinear:
        return "RectifiedLinear";
    }

    return string();
}


const bool& PerceptronLayer3D::get_display() const
{
    return display;
}


void PerceptronLayer3D::set(const Index& new_inputs_number, 
                            const Index& new_inputs_depth, 
                            const Index& new_neurons_number,
                            const PerceptronLayer3D::ActivationFunction& new_activation_function)
{
    inputs_number = new_inputs_number;

    biases.resize(new_neurons_number);

    synaptic_weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_glorot();

    activation_function = new_activation_function;

    name = "perceptron_layer_3d";

    display = true;

    layer_type = Type::Perceptron3D;

    dropout_rate = 0;
}


void PerceptronLayer3D::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void PerceptronLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_depth, neurons_number);
}


void PerceptronLayer3D::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_depth = get_inputs_depth();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_depth, new_neurons_number);
}


void PerceptronLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(synaptic_weights.data(), new_parameters.data() + index, synaptic_weights.size()*sizeof(type));

        #pragma omp section
        memcpy(biases.data(), new_parameters.data() + index + synaptic_weights.size(), biases.size()*sizeof(type));
    }
}


void PerceptronLayer3D::set_activation_function(const PerceptronLayer3D::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void PerceptronLayer3D::set_activation_function(const string& new_activation_function_name)
{
    if (new_activation_function_name == "HyperbolicTangent")
        activation_function = ActivationFunction::HyperbolicTangent;
    else if (new_activation_function_name == "Linear")
        activation_function = ActivationFunction::Linear;
    else if (new_activation_function_name == "RectifiedLinear")
        activation_function = ActivationFunction::RectifiedLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void PerceptronLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


void PerceptronLayer3D::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void PerceptronLayer3D::set_parameters_random()
{
    set_random(biases);

    set_random(synaptic_weights);
}


void PerceptronLayer3D::set_parameters_glorot()
{
    biases.setZero();

    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    const type minimum = -limit;
    const type maximum = limit;

    #pragma omp parallel for
    for(Index i = 0; i < synaptic_weights.size(); i++)
        synaptic_weights(i) = minimum + (maximum - minimum)*type(rand() / (RAND_MAX + 1.0));
}


void PerceptronLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
                                               Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, contraction_indices);

    sum_matrices(thread_pool_device, biases, combinations);
}


void PerceptronLayer3D::dropout(Tensor<type, 3>& outputs) const 
{
    /*
    type* outputs_data = outputs.data();

    const Index batch_samples_number = outputs.dimension(0);
    const Index inputs_number = outputs.dimension(1);
    const Index outputs_number = outputs.dimension(2);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        TensorMap<Tensor<type, 2>> matrix(outputs_data + neuron_index*batch_samples_number*inputs_number,
                                          batch_samples_number, inputs_number);

        random = calculate_random_uniform(type(0), type(1));

        if(random < dropout_rate)
            matrix.setZero();
        else
            matrix.device(*thread_pool_device) = matrix * scaling_factor;
    }
    */
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    #pragma omp parallel for

    for(Index i = 0; i < outputs.size(); i++)
        outputs(i) = (calculate_random_uniform(type(0), type(1)) < dropout_rate) 
            ? 0 
            : outputs(i) * scaling_factor;
}


void PerceptronLayer3D::calculate_activations(Tensor<type, 3>& activations, Tensor<type, 3>& activation_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activation_derivatives); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

    default: return;
    }
}


void PerceptronLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                          const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    PerceptronLayer3DForwardPropagation* perceptron_layer_3d_forward_propagation =
        static_cast<PerceptronLayer3DForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = perceptron_layer_3d_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if(is_training)
    {
        if(dropout_rate > type(0))
            dropout(outputs);

        Tensor<type, 3>& activation_derivatives = perceptron_layer_3d_forward_propagation->activation_derivatives;

        calculate_activations(outputs, activation_derivatives);
    }
    else
    {
        calculate_activations(outputs, empty);
    }
}


void PerceptronLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       const vector<pair<type*, dimensions>>& delta_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Forward propagation

    const PerceptronLayer3DForwardPropagation* perceptron_layer_3d_forward_propagation =
            static_cast<PerceptronLayer3DForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& activation_derivatives = perceptron_layer_3d_forward_propagation->activation_derivatives;

    // Back propagation

    PerceptronLayer3DBackPropagation* perceptron_layer_3d_back_propagation =
            static_cast<PerceptronLayer3DBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& combinations_derivatives = perceptron_layer_3d_back_propagation->combinations_derivatives;

    Tensor<type, 1>& biases_derivatives = perceptron_layer_3d_back_propagation->biases_derivatives;
    Tensor<type, 2>& synaptic_weights_derivatives = perceptron_layer_3d_back_propagation->synaptic_weights_derivatives;

    Tensor<type, 3>& input_derivatives = perceptron_layer_3d_back_propagation->input_derivatives;

    combinations_derivatives.device(*thread_pool_device) 
        = deltas * activation_derivatives;

    biases_derivatives.device(*thread_pool_device)
        = combinations_derivatives.sum(sum_dimensions);

    synaptic_weights_derivatives.device(*thread_pool_device)
        = inputs.contract(combinations_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = combinations_derivatives.contract(synaptic_weights, single_contraction_indices);
}


void PerceptronLayer3D::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    for(Index i = 1; i < static_cast<Index>(delta_pairs.size()); i++)
        deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
}


void PerceptronLayer3D::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                      const Index& index,
                                      Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    PerceptronLayer3DBackPropagation* perceptron_layer_back_propagation =
        static_cast<PerceptronLayer3DBackPropagation*>(back_propagation.get());

    const type* synaptic_weights_derivatives_data = perceptron_layer_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = perceptron_layer_back_propagation->biases_derivatives.data();
    type* gradient_data = gradient.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void PerceptronLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer3D");

    if(!perceptron_layer_element)
        throw runtime_error("PerceptronLayer3D element is nullptr.\n");

    set_name(read_xml_string(perceptron_layer_element, "Name"));
    set_inputs_number(read_xml_index(perceptron_layer_element, "InputsNumber"));
    set_inputs_depth(read_xml_index(perceptron_layer_element, "InputsDepth"));
    set_neurons_number(read_xml_index(perceptron_layer_element, "NeuronsNumber"));
    set_activation_function(read_xml_string(perceptron_layer_element, "ActivationFunction"));
    set_parameters(to_type_vector(read_xml_string(perceptron_layer_element, "Parameters"), " "));
}


void PerceptronLayer3D::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("PerceptronLayer3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_inputs_number()));
    add_xml_element(printer, "InputsDepth", to_string(get_inputs_depth()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_neurons_number()));
    add_xml_element(printer, "ActivationFunction", write_activation_function());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


void PerceptronLayer3DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


PerceptronLayer3DForwardPropagation::PerceptronLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> PerceptronLayer3DForwardPropagation::get_outputs_pair() const
{
    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();

    return { (type*)outputs.data(), { batch_samples_number, inputs_number, neurons_number } };
}


void PerceptronLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();

    outputs.resize(batch_samples_number, inputs_number, neurons_number);

    activation_derivatives.resize(batch_samples_number, inputs_number, neurons_number);
}


void PerceptronLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();
    const Index inputs_number = perceptron_layer_3d->get_inputs_number();
    const Index inputs_depth = perceptron_layer_3d->get_inputs_depth();

    biases_derivatives.resize(neurons_number);

    synaptic_weights_derivatives.resize(inputs_depth, neurons_number);

    combinations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
}


void PerceptronLayer3DBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << biases_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << synaptic_weights_derivatives << endl;
}


PerceptronLayer3DBackPropagation::PerceptronLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> PerceptronLayer3DBackPropagation::get_input_derivative_pairs() const
{
    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();
    const Index inputs_depth = perceptron_layer_3d->get_inputs_depth();

    return {{(type*)(input_derivatives.data()),
            {batch_samples_number, inputs_number, inputs_depth}}};
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
