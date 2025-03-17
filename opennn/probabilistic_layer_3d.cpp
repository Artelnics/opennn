//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "probabilistic_layer_3d.h"
#include "strings_utilities.h"

namespace opennn
{

ProbabilisticLayer3D::ProbabilisticLayer3D(const Index& new_inputs_number, 
                                           const Index& new_inputs_depth, 
                                           const Index& new_neurons_number,
                                           const string& new_name)
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number, new_name);
}


Index ProbabilisticLayer3D::get_inputs_number_xxx() const
{
    return inputs_number_xxx;
}


Index ProbabilisticLayer3D::get_inputs_depth() const
{
    return weights.dimension(0);
}


Index ProbabilisticLayer3D::get_neurons_number() const
{
    return biases.size();
}


dimensions ProbabilisticLayer3D::get_output_dimensions() const
{
    return { inputs_number_xxx, get_neurons_number() };
}


const ProbabilisticLayer3D::ActivationFunction& ProbabilisticLayer3D::get_activation_function() const
{
    return activation_function;
}


string ProbabilisticLayer3D::get_activation_function_string() const
{
    switch (activation_function)
    {
    case ActivationFunction::Competitive:
        return "Competitive";
    case ActivationFunction::Softmax:
        return "Softmax";
    default:
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


string ProbabilisticLayer3D::get_activation_function_text() const
{
    switch (activation_function)
    {
    case ActivationFunction::Competitive:
        return "competitive";
    case ActivationFunction::Softmax:
        return "softmax";
    default:
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


Index ProbabilisticLayer3D::get_parameters_number() const
{
    return biases.size() + weights.size();
}


Tensor<type, 1> ProbabilisticLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(weights.size() + biases.size());

    memcpy(parameters.data(), weights.data(), weights.size()*sizeof(type));

    memcpy(parameters.data() + weights.size(), biases.data(), biases.size()*sizeof(type));

    return parameters;
}


void ProbabilisticLayer3D::set(const Index& new_inputs_number, 
                               const Index& new_inputs_depth, 
                               const Index& new_neurons_number, 
                               const string& new_name)
{
    inputs_number_xxx = new_inputs_number;

    biases.resize(new_neurons_number);

    weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_glorot();

    name = new_name;

    layer_type = Layer::Type::Probabilistic3D;

    activation_function = ActivationFunction::Softmax;
}


void ProbabilisticLayer3D::set_inputs_number(const Index new_inputs_number)
{
    inputs_number_xxx = new_inputs_number;
}


void ProbabilisticLayer3D::set_input_dimensions(const dimensions& new_input_dimensions)
{
/*
    inputs_number = new_inputs_number;
*/
}


void ProbabilisticLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    weights.resize(new_inputs_depth, neurons_number);
}


void ProbabilisticLayer3D::set_output_dimensions(const dimensions& new_output_dimensions)
{
/*
    const Index inputs_depth = get_inputs_depth();

    biases.resize(new_neurons_number);

    weights.resize(inputs_depth, new_neurons_number);
*/
    const Index inputs_depth = get_inputs_depth();
    const Index neurons_number = new_output_dimensions[0];

    biases.resize(neurons_number);

    weights.resize(inputs_depth, neurons_number);
}


void ProbabilisticLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = biases.size();
    const Index weights_number = weights.size();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(weights.data(), new_parameters.data() + index, weights_number*sizeof(type));

        #pragma omp section
        memcpy(biases.data(), new_parameters.data() + index + weights_number, biases_number*sizeof(type));
    }
}


void ProbabilisticLayer3D::set_activation_function(const ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void ProbabilisticLayer3D::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Competitive")
        set_activation_function(ActivationFunction::Competitive);
    else if(new_activation_function == "Softmax")
        set_activation_function(ActivationFunction::Softmax);
    else
        throw runtime_error("Unknown probabilistic method: " + new_activation_function + ".\n");
}


void ProbabilisticLayer3D::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    weights.setConstant(value);
}


void ProbabilisticLayer3D::set_parameters_random()
{
    set_random(biases);

    set_random(weights);
}


void ProbabilisticLayer3D::set_parameters_glorot()
{
    biases.setZero();
    
    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    const type minimum = -limit;
    const type maximum = limit;
    
    #pragma omp parallel for

    for(Index i = 0; i < weights.size(); i++)
        weights(i) = get_random_type(minimum, maximum);
}


void ProbabilisticLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
                                                  Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(weights, contraction_indices);
    sum_matrices(thread_pool_device.get(), biases, combinations);
}


void ProbabilisticLayer3D::calculate_activations(Tensor<type, 3>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Softmax: softmax(activations); 
        return;

    default: 
        return;
    }
}


void ProbabilisticLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                                             const bool&)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    ProbabilisticLayer3DForwardPropagation* probabilistic_layer_3d_forward_propagation =
        static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.get());
    
    Tensor<type, 3>& outputs = probabilistic_layer_3d_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    calculate_activations(outputs);
}


void ProbabilisticLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          const vector<pair<type*, dimensions>>&,
                                          unique_ptr<LayerForwardPropagation>& forward_propagation,
                                          unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    // Forward propagation

    ProbabilisticLayer3DForwardPropagation* probabilistic_layer_3d_forward_propagation =
            static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& outputs = probabilistic_layer_3d_forward_propagation->outputs;

    // Back propagation

    ProbabilisticLayer3DBackPropagation* probabilistic_layer_3d_back_propagation =
            static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.get());

    const Tensor<type, 2>& targets = probabilistic_layer_3d_back_propagation->targets;
    Tensor<type, 2>& mask = probabilistic_layer_3d_back_propagation->mask;
    bool& built_mask = probabilistic_layer_3d_back_propagation->built_mask;

    Tensor<type, 3>& combination_derivatives = probabilistic_layer_3d_back_propagation->combination_derivatives;

    Tensor<type, 1>& bias_derivatives = probabilistic_layer_3d_back_propagation->bias_derivatives;
    Tensor<type, 2>& synaptic_weight_derivatives = probabilistic_layer_3d_back_propagation->synaptic_weight_derivatives;

    Tensor<type, 3>& input_derivatives = probabilistic_layer_3d_back_propagation->input_derivatives;

    if(!built_mask)
    {
        mask.device(*thread_pool_device) = (targets != targets.constant(0)).cast<type>();
 
        const Tensor<type, 0> mask_sum = mask.sum();
        
        mask.device(*thread_pool_device) = mask / mask_sum(0);
        
        built_mask = true;
    }

    calculate_combinations_derivatives(outputs, targets, mask, combination_derivatives);

    bias_derivatives.device(*thread_pool_device) 
        = combination_derivatives.sum(sum_dimensions);

    synaptic_weight_derivatives.device(*thread_pool_device) 
        = inputs.contract(combination_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = combination_derivatives.contract(weights, single_contraction_indices);
}


void ProbabilisticLayer3D::calculate_combinations_derivatives(const Tensor<type, 3>& outputs, 
                                                              const Tensor<type, 2>& targets,
                                                              const Tensor<type, 2>& mask,
                                                              Tensor<type, 3>& combination_derivatives) const
{
    const Index batch_samples_number = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    combination_derivatives.device(*thread_pool_device) = outputs;

    #pragma omp parallel for collapse(2)

    for(Index i = 0; i < batch_samples_number; i++)
        for(Index j = 0; j < outputs_number; j++)
            combination_derivatives(i, j, Index(targets(i, j)))--;

    multiply_matrices(thread_pool_device.get(), combination_derivatives, mask);
}


void ProbabilisticLayer3D::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                           const Index& index,
                                           Tensor<type, 1>& gradient) const
{
    const Index biases_number = biases.size();
    const Index weights_number = weights.size();

    const ProbabilisticLayer3DBackPropagation* probabilistic_layer_3d_back_propagation =
        static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.get());

    const type* weight_derivatives_data = probabilistic_layer_3d_back_propagation->synaptic_weight_derivatives.data();
    const type* biases_derivatives_data = probabilistic_layer_3d_back_propagation->bias_derivatives.data();

    type* gradient_data = gradient.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, weight_derivatives_data, weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void ProbabilisticLayer3D::from_XML(const XMLDocument& document)
{
    const XMLElement* probabilistic_layer_element = document.FirstChildElement("Probabilistic3D");

    if(!probabilistic_layer_element)
        throw runtime_error("Probabilistic3D element is nullptr.\n");

    set_name(read_xml_string(probabilistic_layer_element, "Name"));
    set_inputs_number(read_xml_index(probabilistic_layer_element, "InputsNumber"));
    set_inputs_depth(read_xml_index(probabilistic_layer_element, "InputsDepth"));
    set_output_dimensions({read_xml_index(probabilistic_layer_element, "NeuronsNumber")});
    set_activation_function(read_xml_string(probabilistic_layer_element, "ActivationFunction"));
    set_parameters(to_type_vector(read_xml_string(probabilistic_layer_element, "Parameters"), " "));

}


void ProbabilisticLayer3D::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Probabilistic3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_inputs_number_xxx()));
    add_xml_element(printer, "InputsDepth", to_string(get_inputs_depth()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_neurons_number()));
    add_xml_element(printer, "ActivationFunction", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


ProbabilisticLayer3DForwardPropagation::ProbabilisticLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> ProbabilisticLayer3DForwardPropagation::get_outputs_pair() const
{
    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();

    return {(type*)outputs.data(), {batch_samples_number, inputs_number, neurons_number}};
}


void ProbabilisticLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    
    outputs.resize(batch_samples_number, inputs_number, neurons_number);
}


void ProbabilisticLayer3DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


void ProbabilisticLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    targets.resize(batch_samples_number, inputs_number);
    mask.resize(batch_samples_number, inputs_number);

    bias_derivatives.resize(neurons_number);

    synaptic_weight_derivatives.resize(inputs_depth, neurons_number);

    combination_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
}


void ProbabilisticLayer3DBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << synaptic_weight_derivatives << endl;
}


ProbabilisticLayer3DBackPropagation::ProbabilisticLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> ProbabilisticLayer3DBackPropagation::get_input_derivative_pairs() const
{
    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    return {{(type*)(input_derivatives.data()), {batch_samples_number, inputs_number, inputs_depth}} };
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
