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

Probabilistic3d::Probabilistic3d(const Index& new_inputs_number, 
                                 const Index& new_inputs_depth, 
                                 const Index& new_neurons_number,
                                 const string& new_name) : Layer()
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number, new_name);
}


Index Probabilistic3d::get_inputs_number_xxx() const
{
    return inputs_number_xxx;
}


Index Probabilistic3d::get_inputs_depth() const
{
    return weights.dimension(0);
}


Index Probabilistic3d::get_neurons_number() const
{
    return biases.size();
}


dimensions Probabilistic3d::get_output_dimensions() const
{
    return { inputs_number_xxx, get_neurons_number() };
}


const Probabilistic3d::Activation& Probabilistic3d::get_activation_function() const
{
    return activation_function;
}


string Probabilistic3d::get_activation_function_string() const
{
    switch (activation_function)
    {
    case Activation::Competitive:
        return "Competitive";
    case Activation::Softmax:
        return "Softmax";
    default:
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


string Probabilistic3d::get_activation_function_text() const
{
    switch (activation_function)
    {
    case Activation::Competitive:
        return "competitive";
    case Activation::Softmax:
        return "softmax";
    default:
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


Index Probabilistic3d::get_parameters_number() const
{
    return biases.size() + weights.size();
}


void Probabilistic3d::get_parameters(Tensor<type, 1>& parameters) const
{
    parameters.resize(weights.size() + biases.size());

    Index index = 0;

    copy_to_vector(parameters, weights, index);
    copy_to_vector(parameters, biases, index);
}


void Probabilistic3d::set(const Index& new_inputs_number, 
                               const Index& new_inputs_depth, 
                               const Index& new_neurons_number, 
                               const string& new_name)
{
    inputs_number_xxx = new_inputs_number;

    biases.resize(new_neurons_number);

    weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_glorot();
    // set_parameters_random();

    name = new_name;

    layer_type = Layer::Type::Probabilistic3d;

    activation_function = Activation::Softmax;
}


void Probabilistic3d::set_inputs_number(const Index new_inputs_number)
{
    inputs_number_xxx = new_inputs_number;
}



void Probabilistic3d::set_input_dimensions(const dimensions& new_input_dimensions)
{
    // @todo
}


void Probabilistic3d::set_inputs_depth(const Index& new_inputs_depth)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    weights.resize(new_inputs_depth, neurons_number);
}


void Probabilistic3d::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_depth = get_inputs_depth();
    const Index neurons_number = new_output_dimensions[0];

    biases.resize(neurons_number);

    weights.resize(inputs_depth, neurons_number);
}


void Probabilistic3d::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(weights, new_parameters, index);
    copy_from_vector(biases, new_parameters, index);
}


void Probabilistic3d::set_activation_function(const Activation& new_activation_function)
{
    activation_function = new_activation_function;
}


void Probabilistic3d::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Competitive")
        set_activation_function(Activation::Competitive);
    else if(new_activation_function == "Softmax")
        set_activation_function(Activation::Softmax);
    else
        throw runtime_error("Unknown probabilistic method: " + new_activation_function + ".\n");
}


void Probabilistic3d::set_parameters_random()
{
    set_random(biases);

    set_random(weights);
}


void Probabilistic3d::set_parameters_glorot()
{
    biases.setZero();
    
    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    set_random(weights, -limit, limit);
}


void Probabilistic3d::calculate_combinations(const Tensor<type, 3>& inputs,
                                             Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(weights, axes(2,0))
     + biases.reshape(array<Index, 3>{1, 1, biases.dimension(0)})
             .broadcast(array<Index, 3>{combinations.dimension(0), combinations.dimension(1), 1});

    //sum_matrices(thread_pool_device.get(), biases, combinations);
}


void Probabilistic3d::calculate_activations(Tensor<type, 3>& activations) const
{
    switch(activation_function)
    {
    case Activation::Softmax: softmax(activations); 
        return;

    default: 
        return;
    }
}


void Probabilistic3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        const bool&)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_pairs[0]);

    Probabilistic3DForwardPropagation* this_forward_propagation =
        static_cast<Probabilistic3DForwardPropagation*>(forward_propagation.get());
    
    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    calculate_activations(outputs);
}


void Probabilistic3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          const vector<pair<type*, dimensions>>&,
                                          unique_ptr<LayerForwardPropagation>& forward_propagation,
                                          unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_pairs[0]);
    const Index samples_number = inputs.dimension(0);

    // Forward propagation

    Probabilistic3DForwardPropagation* this_forward_propagation =
            static_cast<Probabilistic3DForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    // Back propagation

    Probabilistic3dBackPropagation* probabilistic_3d_back_propagation =
            static_cast<Probabilistic3dBackPropagation*>(back_propagation.get());

    const Tensor<type, 2>& targets = probabilistic_3d_back_propagation->targets;

    Tensor<type, 2>& mask = probabilistic_3d_back_propagation->mask;

    bool& built_mask = probabilistic_3d_back_propagation->built_mask;

    Tensor<type, 3>& combination_deltas = probabilistic_3d_back_propagation->combination_deltas;

    Tensor<type, 1>& bias_deltas = probabilistic_3d_back_propagation->bias_deltas;
    Tensor<type, 2>& weight_deltas = probabilistic_3d_back_propagation->weight_deltas;
    Tensor<type, 3>& input_deltas = probabilistic_3d_back_propagation->input_deltas;

    // if(!built_mask)
    // {
        mask.device(*thread_pool_device) = (targets != targets.constant(0)).cast<type>();
 
        const Tensor<type, 0> mask_sum = mask.sum();
        
        mask.device(*thread_pool_device) = mask / type(samples_number)/*mask_sum(0)*/;
        
        built_mask = true;
    // }

    calculate_combination_deltas(outputs, targets, mask, combination_deltas);

    bias_deltas.device(*thread_pool_device)
        = combination_deltas.sum(array<Index, 2>({0,1}));

    weight_deltas.device(*thread_pool_device)
        = inputs.contract(combination_deltas, axes(0,0,1,1));

    input_deltas.device(*thread_pool_device)
        = combination_deltas.contract(weights, axes(2,1));
}


void Probabilistic3d::calculate_combination_deltas(const Tensor<type, 3>& outputs,
                                                   const Tensor<type, 2>& targets,
                                                   const Tensor<type, 2>& mask,
                                                   Tensor<type, 3>& combination_deltas) const
{
    const Index batch_size = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    combination_deltas = outputs;

    #pragma omp parallel for collapse(2)

    for(Index i = 0; i < batch_size; i++)
        for(Index j = 0; j < outputs_number; j++)
            combination_deltas(i, j, Index(targets(i, j)))--;

    multiply_matrices(thread_pool_device.get(), combination_deltas, mask);
}


void Probabilistic3d::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                           Index& index,
                                           Tensor<type, 1>& gradient) const
{
    const Probabilistic3dBackPropagation* probabilistic_3d_back_propagation =
        static_cast<Probabilistic3dBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, probabilistic_3d_back_propagation->weight_deltas, index);
    copy_to_vector(gradient, probabilistic_3d_back_propagation->bias_deltas, index);
}


void Probabilistic3d::from_XML(const XMLDocument& document)
{
    const XMLElement* probabilistic_layer_element = document.FirstChildElement("Probabilistic3d");

    if(!probabilistic_layer_element)
        throw runtime_error("Probabilistic3d element is nullptr.\n");

    const Index new_inputs_number = read_xml_index(probabilistic_layer_element, "InputsNumber");
    const Index new_inputs_depth = read_xml_index(probabilistic_layer_element, "InputsDepth");
    const Index new_neurons_number = read_xml_index(probabilistic_layer_element, "NeuronsNumber");

    set(new_inputs_number, new_inputs_depth, new_neurons_number);

    set_name(read_xml_string(probabilistic_layer_element, "Name"));
    set_activation_function(read_xml_string(probabilistic_layer_element, "Activation"));

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(probabilistic_layer_element, "Parameters"), " "), index);
}


void Probabilistic3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Probabilistic3d");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_inputs_number_xxx()));
    add_xml_element(printer, "InputsDepth", to_string(get_inputs_depth()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_neurons_number()));
    add_xml_element(printer, "Activation", get_activation_function_string());

    Tensor<type, 1> parameters;
    get_parameters(parameters);

    add_xml_element(printer, "Parameters", tensor_to_string(parameters));

    printer.CloseElement();
}


Probabilistic3DForwardPropagation::Probabilistic3DForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Probabilistic3DForwardPropagation::get_outputs_pair() const
{
    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();

    return {(type*)outputs.data(), {batch_size, inputs_number, neurons_number}};
}


void Probabilistic3DForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

    batch_size = new_batch_size;

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    
    outputs.resize(batch_size, inputs_number, neurons_number);
}


void Probabilistic3DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


void Probabilistic3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

    batch_size = new_batch_size;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    targets.resize(batch_size, inputs_number);
    mask.resize(batch_size, inputs_number);

    bias_deltas.resize(neurons_number);

    weight_deltas.resize(inputs_depth, neurons_number);

    input_deltas.resize(batch_size, inputs_number, inputs_depth);
}


void Probabilistic3dBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_deltas << endl
         << "Synaptic weights derivatives:" << endl
         << weight_deltas << endl;
}


Probabilistic3dBackPropagation::Probabilistic3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Probabilistic3dBackPropagation::get_input_derivative_pairs() const
{
    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    return {{(type*)(input_deltas.data()), {batch_size, inputs_number, inputs_depth}} };
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
