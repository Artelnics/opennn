//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
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


void Probabilistic3d::set(const Index& new_inputs_number, 
                               const Index& new_inputs_depth,
                               const Index& new_neurons_number,
                               const string& new_label)
{
    inputs_number_xxx = new_inputs_number;

    biases.resize(new_neurons_number);

    weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_glorot();
    // set_parameters_random();

    label = new_label;

    name = "Probabilistic3d";

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


vector<pair<type *, Index> > Probabilistic3d::get_parameter_pairs() const
{
    return {{(type*)(biases.data()), biases.size()},
            {(type*)(weights.data()), weights.size()}};
}


void Probabilistic3d::calculate_combinations(const Tensor<type, 3>& inputs,
                                             Tensor<type, 3>& combinations) const
{
    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index input_features = inputs.dimension(2);
    const Index output_features = combinations.dimension(2);

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < sequence_length; ++j)
            for (Index k = 0; k < output_features; ++k)
            {
                type sum = biases(k);

                for (Index l = 0; l < input_features; ++l)
                    sum += inputs(i, j, l) * weights(l, k);

                combinations(i, j, k) = sum;
            }
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

    Probabilistic3dForwardPropagation* this_forward_propagation =
        static_cast<Probabilistic3dForwardPropagation*>(forward_propagation.get());
    
    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    calculate_activations(outputs);
}


void Probabilistic3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          const vector<pair<type*, dimensions>>& delta_pairs,
                                          unique_ptr<LayerForwardPropagation>& forward_propagation,
                                          unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_pairs[0]);
    const Index samples_number = inputs.dimension(0);

    if (delta_pairs.size() > 1) add_deltas(delta_pairs);

    // Forward propagation

    Probabilistic3dForwardPropagation* this_forward_propagation =
            static_cast<Probabilistic3dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    // Back propagation

    Probabilistic3dBackPropagation* probabilistic_3d_back_propagation =
            static_cast<Probabilistic3dBackPropagation*>(back_propagation.get());

    const Tensor<type, 2>& targets = probabilistic_3d_back_propagation->targets;

    Tensor<type, 2>& mask = probabilistic_3d_back_propagation->mask;

    Tensor<type, 3>& combination_deltas = probabilistic_3d_back_propagation->combination_deltas;
    Tensor<type, 1>& bias_deltas = probabilistic_3d_back_propagation->bias_deltas;
    Tensor<type, 2>& weight_deltas = probabilistic_3d_back_propagation->weight_deltas;
    Tensor<type, 3>& input_deltas = probabilistic_3d_back_propagation->input_deltas;

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index input_features = inputs.dimension(2);
    const Index output_features = combination_deltas.dimension(2);

    // Mask

    #pragma omp parallel for
    for (Index i = 0; i < mask.size(); ++i)
        mask.data()[i] = (targets.data()[i] != type(0)) ? type(1) : type(0);

    const Tensor<type, 0> mask_sum_tensor = mask.sum();
    const type mask_sum = mask_sum_tensor(0);
    const type norm_factor = (mask_sum > 0) ? type(1.0) / mask_sum : type(1.0) / type(samples_number);

    #pragma omp parallel for
    for (Index i = 0; i < mask.size(); ++i)
        mask.data()[i] *= norm_factor;

    // Deltas

    calculate_combination_deltas(outputs, targets, mask, combination_deltas);

    // Biases

    bias_deltas.setZero();

    #pragma omp parallel
    {
        Tensor<type, 1> private_bias_deltas(output_features); private_bias_deltas.setZero();

        #pragma omp for
        for (Index i = 0; i < batch_size; ++i)
            for (Index j = 0; j < sequence_length; ++j)
                for (Index k = 0; k < output_features; ++k)
                    private_bias_deltas(k) += combination_deltas(i, j, k);

        #pragma omp critical
        {
            bias_deltas += private_bias_deltas;
        }
    }

    // Weights

    #pragma omp parallel for
    for (Index i = 0; i < input_features; ++i)
        for (Index j = 0; j < output_features; ++j)
        {
            type sum = 0;

            for (Index k = 0; k < batch_size; ++k)
                for (Index l = 0; l < sequence_length; ++l)
                    sum += inputs(k, l, i) * combination_deltas(k, l, j);

            weight_deltas(i, j) = sum;
        }

    // Previous layer

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < sequence_length; ++j)
            for (Index k = 0; k < input_features; ++k)
            {
                type sum = 0;

                for (Index l = 0; l < output_features; ++l)
                    sum += combination_deltas(i, j, l) * weights(k, l);

                input_deltas(i, j, k) = sum;
            }
}


void Probabilistic3d::calculate_combination_deltas(const Tensor<type, 3>& outputs,
                                                   const Tensor<type, 2>& targets,
                                                   const Tensor<type, 2>& mask,
                                                   Tensor<type, 3>& combination_deltas) const
{
    const Index batch_size = outputs.dimension(0);
    const Index seq_len = outputs.dimension(1);
    const Index features = outputs.dimension(2);

    combination_deltas = outputs;

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < seq_len; ++j)
        {
            const Index target_class_index = static_cast<Index>(targets(i, j));

            if (target_class_index >= 0 && target_class_index < features)
                combination_deltas(i, j, target_class_index) -= type(1);
        }

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < seq_len; ++j)
        {
            const type mask_value = mask(i, j);

            for (Index k = 0; k < features; ++k)
                combination_deltas(i, j, k) *= mask_value;
        }
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

    set_label(read_xml_string(probabilistic_layer_element, "Label"));
    set_activation_function(read_xml_string(probabilistic_layer_element, "Activation"));
    string_to_tensor<type, 1>(read_xml_string(probabilistic_layer_element, "Biases"), biases);
    string_to_tensor<type, 2>(read_xml_string(probabilistic_layer_element, "Weights"), weights);
}


void Probabilistic3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Probabilistic3d");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputsNumber", to_string(get_inputs_number_xxx()));
    add_xml_element(printer, "InputsDepth", to_string(get_inputs_depth()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_neurons_number()));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "Biases", tensor_to_string<type, 1>(biases));
    add_xml_element(printer, "Weights", tensor_to_string<type, 2>(weights));

    printer.CloseElement();
}


Probabilistic3dForwardPropagation::Probabilistic3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Probabilistic3dForwardPropagation::get_output_pair() const
{
    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();

    return {(type*)outputs.data(), {batch_size, inputs_number, neurons_number}};
}


void Probabilistic3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

    batch_size = new_batch_size;

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number_xxx();
    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    
    outputs.resize(batch_size, inputs_number, neurons_number);
}


void Probabilistic3dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


void Probabilistic3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    Probabilistic3d* probabilistic_layer_3d = static_cast<Probabilistic3d*>(layer);

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


vector<pair<type*, Index>> Probabilistic3dBackPropagation::get_parameter_delta_pairs() const
{
    return {
        {(type*)bias_deltas.data(), bias_deltas.size()},
        {(type*)weight_deltas.data(), weight_deltas.size()}
    };
}

REGISTER(Layer, Probabilistic3d, "Probabilistic3d")
REGISTER(LayerForwardPropagation, Probabilistic3dForwardPropagation, "Probabilistic3d")
REGISTER(LayerBackPropagation, Probabilistic3dBackPropagation, "Probabilistic3d")

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
