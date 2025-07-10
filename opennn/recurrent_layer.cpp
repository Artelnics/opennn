//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "recurrent_layer.h"

namespace opennn
{

Recurrent::Recurrent(const dimensions& new_input_dimensions,
                     const dimensions& new_output_dimensions) : Layer()
{
    set(new_input_dimensions, new_output_dimensions);
}


dimensions Recurrent::get_input_dimensions() const
{
    return { time_steps, input_weights.dimension(0) };
}


dimensions Recurrent::get_output_dimensions() const
{
    return { biases.size() };
}


Index Recurrent::get_timesteps() const
{
    return time_steps;
}


void Recurrent::get_parameters(Tensor<type, 1>& parameters) const
{
    const Index parameters_number = get_parameters_number();

    parameters.resize(parameters_number);

    Index index = 0;

    copy_to_vector(parameters, biases, index);
    copy_to_vector(parameters, input_weights, index);
    copy_to_vector(parameters, recurrent_weights, index);
}


string Recurrent::get_activation_function() const
{
    return activation_function;
}


void Recurrent::set(const dimensions& new_input_dimensions, const dimensions& new_output_dimensions)
{
    time_steps = new_input_dimensions[0];

    biases.resize(new_output_dimensions[0]);

    input_weights.resize(new_input_dimensions[0], new_output_dimensions[0]);

    recurrent_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);

    set_parameters_random();

    label = "recurrent_layer";

    name = "Recurrent";
}


void Recurrent::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const Index outputs_number = get_outputs_number();

    input_weights.resize(new_input_dimensions[0], outputs_number);
}


void Recurrent::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_output_dimensions[0]);

    input_weights.resize(inputs_number, new_output_dimensions[0]);

    recurrent_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);
}


void Recurrent::set_timesteps(const Index& new_timesteps)
{
    time_steps = new_timesteps;
}


void Recurrent::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(biases, new_parameters, index);
    copy_from_vector(input_weights, new_parameters, index);
    copy_from_vector(recurrent_weights, new_parameters, index);
}



void Recurrent::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Logistic"
    || new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "ExponentialLinear")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}


void Recurrent::calculate_combinations(const Tensor<type, 2>& inputs,
                                       Tensor<type, 2>& combinations) const
{
    const Index samples_number = inputs.dimension(0);

    // Compute the new hidden state: h_t = tanh(W_x * x_t + W_h * h_t + b)
    combinations = inputs.contract(input_weights, axes(1,0))
                   + previous_hidden_states.contract(recurrent_weights, axes(1,0))
                   + biases.reshape(Eigen::DSizes<Index,2>{1, biases.dimension(0)})
                         .broadcast(array<Index,2>{samples_number, 1});
}


void Recurrent::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool& is_training)
{
    const Index batch_size = input_pairs[0].second[0];
    const Index time_steps = input_pairs[0].second[1];
    const Index input_size = input_pairs[0].second[2];

    TensorMap<Tensor<type, 3>> inputs(input_pairs[0].first, batch_size, time_steps, input_size);

    RecurrentForwardPropagation* recurrent_forward =
        static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

    Tensor<type, 2>& outputs = recurrent_forward->outputs;
    Tensor<type, 3>& activation_derivatives = recurrent_forward->activation_derivatives;
    Tensor<type, 2>& current_activations_derivatives = recurrent_forward->current_activations_derivatives;
    Tensor<type, 3>& hidden_states = recurrent_forward->hidden_states;

    outputs.resize(batch_size, input_size);

    for(Index t = 0; t < time_steps; t++)
    {
        if (t == 0)
            outputs = inputs.chip(t, 1).contract(input_weights, axes(1,0))
                      + biases
                            .reshape(DSizes<Index, 2>{1, biases.dimension(0)})
                            .broadcast(array<Index, 2>({ batch_size, 1 }));
        else
        {
            previous_hidden_states = hidden_states.chip(t-1,1);

            calculate_combinations(inputs.chip(t, 1),outputs);
        }

        current_activations_derivatives = activation_derivatives.chip(t, 1);

        calculate_activations(activation_function, outputs, current_activations_derivatives);

        activation_derivatives.chip(t,1) = current_activations_derivatives;

        hidden_states.chip(t,1) = outputs;
    }
}


void Recurrent::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>& forward_propagation,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_pairs[0].second[0];
    const Index time_steps = input_pairs[0].second[1];
    const Index input_size = input_pairs[0].second[2];
    const Index output_size = get_outputs_number();

    TensorMap<Tensor<type, 3>> inputs(input_pairs[0].first, batch_size, time_steps, input_size);
    TensorMap<Tensor<type, 2>> deltas(delta_pairs[0].first, batch_size, output_size);

    RecurrentForwardPropagation* recurrent_forward =
        static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

    RecurrentBackPropagation* recurrent_backward =
        static_cast<RecurrentBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& hidden_states = recurrent_forward->hidden_states;

    Tensor<type, 2>& current_deltas = recurrent_backward->current_deltas;
    Tensor<type, 3>& input_deltas = recurrent_backward->input_deltas;
    Tensor<type, 2>& input_weight_deltas = recurrent_backward->input_weight_deltas;
    Tensor<type, 2>& recurrent_weight_deltas = recurrent_backward->recurrent_weight_deltas;
    Tensor<type, 1>& bias_deltas = recurrent_backward->bias_deltas;
    Tensor<type, 2>& combination_deltas = recurrent_backward->combination_deltas;
    Tensor<type, 2>& current_combinations_derivatives = recurrent_backward->current_combinations_derivatives;

    Tensor<type, 3>& activation_derivatives = recurrent_forward->activation_derivatives;

    for(Index t = time_steps - 1; t >= 0; --t)
    {
        if (t == time_steps - 1)
            current_deltas = deltas;
        else
            current_deltas += current_combinations_derivatives;

        combination_deltas.device(*thread_pool_device) =
            current_deltas * activation_derivatives.chip(t,1);

        // Need

        input_weight_deltas += inputs.chip(t,1).contract(combination_deltas, axes(0,0));

        if(t > 0)
        {
            recurrent_weight_deltas.device(*thread_pool_device) +=
                hidden_states.chip(t-1,1)
                    .contract(combination_deltas, axes(0,0));

            current_combinations_derivatives.device(*thread_pool_device) =
                combination_deltas.contract(recurrent_weights, axes(1,0));
        }
        else
            current_combinations_derivatives.setZero();

        bias_deltas.device(*thread_pool_device) += combination_deltas.sum(array<Index, 1>({ 0 }));

        input_deltas.chip(t,1).device(*thread_pool_device) =
            combination_deltas.contract(
                input_weights.shuffle(array<Index,2>{{1,0}}),
                axes(1,0));
    }
}

void Recurrent::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                Index& index,
                                Tensor<type, 1>& gradient) const
{
    RecurrentBackPropagation* recurrent_back_propagation =
        static_cast<RecurrentBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, recurrent_back_propagation->bias_deltas, index);
    copy_to_vector(gradient, recurrent_back_propagation->input_weight_deltas, index);
    copy_to_vector(gradient, recurrent_back_propagation->recurrent_weight_deltas, index);
}


string Recurrent::get_expression(const vector<string>& input_names,
                                 const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
    {
        const Tensor<type, 1> weights_column =  recurrent_weights.chip(j,1);

        buffer << output_names[j] << " = " << activation_function << "( " << biases(j) << " +";

        for(size_t i = 0; i < input_names.size() - 1; i++)
            buffer << " (" << input_names[i] << "*" << weights_column(i) << ") +";

        buffer << " (" << input_names[input_names.size() - 1] << "*" << weights_column[input_names.size() - 1] << "));\n";
    }

    return buffer.str();
}


void Recurrent::print() const
{
    cout << "Recurrent layer" << endl
         << "Input dimensions: " << get_input_dimensions()[0] << endl
         << "Output dimensions: " << get_output_dimensions()[0] << endl
         << "Biases dimensions: " << biases.dimensions() << endl
         << "Input weights dimensions: " << input_weights.dimensions() << endl
         << "Recurrent weights dimensions: " << recurrent_weights.dimensions() << endl;

    cout << "Biases:" << endl;
    cout << biases << endl;
    cout << "Input weights:" << endl;
    cout << input_weights << endl;
    cout << "Recurrent weights:" << endl;
    cout << recurrent_weights << endl;
    cout << "Activation function:" << endl;
    cout << activation_function << endl;
}


void Recurrent::from_XML(const XMLDocument& document)
{
    const XMLElement* recurrent_layer_element = document.FirstChildElement("Recurrent");

    if(!recurrent_layer_element)
        throw runtime_error("Recurrent layer element is nullptr.\n");

    set_label(read_xml_string(recurrent_layer_element,"Label"));
    set_input_dimensions({ read_xml_index(recurrent_layer_element, "InputsNumber") });
    set_output_dimensions({ read_xml_index(recurrent_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(recurrent_layer_element, "Activation"));

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(recurrent_layer_element, "Parameters"), " "), index);

}


void Recurrent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Recurrent");

    add_xml_element(printer, "Label", get_label());
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", activation_function);

    Tensor<type, 1> parameters;
    get_parameters(parameters);

    add_xml_element(printer, "Parameters", tensor_to_string(parameters));

    printer.CloseElement();
}


RecurrentForwardPropagation::RecurrentForwardPropagation(const Index& new_batch_size, Layer* new_layer) : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> RecurrentForwardPropagation::get_output_pair() const
{
    const Index outputs_number = layer->get_outputs_number();

    return {(type*)outputs.data(), {{batch_size, outputs_number}}};
}


void RecurrentForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size=new_batch_size;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index time_steps = layer->get_input_dimensions()[1];

    current_inputs.resize(batch_size, time_steps, inputs_number);

    current_activations_derivatives.resize(batch_size, outputs_number);

    activation_derivatives.resize(batch_size, time_steps, outputs_number);

    outputs.resize(batch_size, outputs_number);
    outputs.setZero();

    hidden_states.resize(batch_size, time_steps, outputs_number);
    hidden_states.setZero();
}


void RecurrentForwardPropagation::print() const
{
}


void RecurrentBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    if (!layer) return;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index time_steps = layer->get_input_dimensions()[1];

    combinations_bias_deltas.resize(outputs_number, outputs_number);
    combinations_input_weight_deltas.resize(inputs_number, outputs_number, outputs_number);
    combinations_recurrent_weight_deltas.resize(outputs_number, outputs_number, outputs_number);
    combination_deltas.resize(batch_size, outputs_number);
    current_combinations_derivatives.resize(batch_size, outputs_number);
    bias_deltas.resize(outputs_number);
    input_weight_deltas.resize(inputs_number, outputs_number);
    recurrent_weight_deltas.resize(outputs_number, outputs_number);
    input_deltas.resize(batch_size, time_steps, inputs_number);

    input_weight_deltas.setZero();
    recurrent_weight_deltas.setZero();
    bias_deltas.setZero();
    input_deltas.setZero();
    current_combinations_derivatives.setZero();
    current_deltas.setZero();
    combination_deltas.setZero();
}


void RecurrentBackPropagation::print() const
{

}


RecurrentBackPropagation::RecurrentBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> RecurrentBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_deltas.data()), {batch_size, inputs_number}}};
}

vector<pair<type*, Index>> RecurrentBackPropagation::get_parameter_delta_pairs() const
{
    return {
        {(type*)bias_deltas.data(), bias_deltas.size()},
        {(type*)input_weight_deltas.data(), input_weight_deltas.size()},
        {(type*)recurrent_weight_deltas.data(), recurrent_weight_deltas.size()}
    };
}

REGISTER(Layer, Recurrent, "Recurrent")
REGISTER(LayerForwardPropagation, RecurrentForwardPropagation, "Recurrent")
REGISTER(LayerBackPropagation, RecurrentBackPropagation, "Recurrent")

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
