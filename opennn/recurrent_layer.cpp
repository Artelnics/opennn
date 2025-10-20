//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
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
    return input_dimensions;
}


dimensions Recurrent::get_output_dimensions() const
{
    return { biases.size() };
}


vector<ParameterView > Recurrent::get_parameter_views() const
{
    return {
        {(type*)biases.data(), biases.size()},
        {(type*)input_weights.data(), input_weights.size()},
        {(type*)recurrent_weights.data(), recurrent_weights.size()}
    };
}


string Recurrent::get_activation_function() const
{
    return activation_function;
}


void Recurrent::set(const dimensions& new_input_dimensions, const dimensions& new_output_dimensions)
{
    set_input_dimensions(new_input_dimensions);
    set_output_dimensions(new_output_dimensions);

    const Index inputs_number = new_input_dimensions[1];
    const Index outputs_number = new_output_dimensions[0];

    biases.resize(outputs_number);

    input_weights.resize(inputs_number, outputs_number);

    recurrent_weights.resize(outputs_number, outputs_number);

    set_parameters_random();

    label = "recurrent_layer";

    name = "Recurrent";
}


void Recurrent::set_input_dimensions(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 2)
        throw runtime_error("Input dimensions rank is not 2 for Recurrent (time_steps, inputs).");

    input_dimensions = new_input_dimensions;

    const Index inputs_number = input_dimensions[1];
    const Index outputs_number = get_outputs_number();

    input_weights.resize(inputs_number, outputs_number);
}


void Recurrent::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = input_weights.dimension(0);
    const Index outputs_number = new_output_dimensions[0];

    biases.resize(outputs_number);
    input_weights.resize(inputs_number, outputs_number);
    recurrent_weights.resize(outputs_number, outputs_number);
}


void Recurrent::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Logistic"
    || new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "ScaledExponentialLinear")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}


void Recurrent::calculate_combinations(const Tensor<type, 2>& inputs,
                                       const Tensor<type, 2>& previous_hidden_states,
                                       Tensor<type, 2>& combinations) const
{
    const Index batch_size = inputs.dimension(0);

    // Compute the new hidden state: h_t = tanh(W_x * x_t + W_h * h_t + b)
    combinations = inputs.contract(input_weights, axes(1,0))
                   + previous_hidden_states.contract(recurrent_weights, axes(1,0))
                   + biases.reshape(Eigen::DSizes<Index,2>{1, biases.dimension(0)})
                         .broadcast(array<Index,2>{batch_size, 1});
}


void Recurrent::forward_propagate(const vector<TensorView>& input_views,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool&)
{
    const Index batch_size = input_views[0].dims[0];
    const Index past_time_steps = input_views[0].dims[1];
    const Index input_size = input_views[0].dims[2];

    TensorMap<Tensor<type, 3>> inputs(input_views[0].data, batch_size, past_time_steps, input_size);

    RecurrentForwardPropagation* recurrent_forward =
        static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

    Tensor<type, 2>& outputs = recurrent_forward->outputs;
    Tensor<type, 3>& activation_derivatives = recurrent_forward->activation_derivatives;
    Tensor<type, 2>& current_activation_derivatives = recurrent_forward->current_activation_derivatives;
    Tensor<type, 3>& hidden_states = recurrent_forward->hidden_states;

    const Index output_size = input_weights.dimension(1);

    Tensor<type, 2> previous_hidden_states(batch_size, output_size);
    previous_hidden_states.setZero();
    for(Index time_step = 0; time_step < past_time_steps; time_step++)
    {
        // Compute the new hidden state: h_t = tanh(W_x * x_t + W_h * h_t + b)
        outputs.device(*thread_pool_device)
            = inputs.chip(time_step, 1).contract(input_weights, axes(1,0))
              + previous_hidden_states.contract(recurrent_weights, axes(1,0))
              + biases.reshape(Eigen::DSizes<Index,2>{1, output_size}).broadcast(array<Index,2>{batch_size, 1});

        //calculate_combinations(inputs.chip(t, 1), previous_hidden_state, outputs);

        current_activation_derivatives.device(*thread_pool_device) =
            activation_derivatives.chip(time_step, 1);

        calculate_activations(activation_function, outputs, current_activation_derivatives);

        activation_derivatives.chip(time_step, 1) = current_activation_derivatives;

        hidden_states.chip(time_step, 1) = outputs;

        previous_hidden_states = outputs;
    }
}


void Recurrent::back_propagate(const vector<TensorView>& input_views,
                               const vector<TensorView>& delta_views,
                               unique_ptr<LayerForwardPropagation>& forward_propagation,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_views[0].dims[0];
    const Index past_time_steps = input_views[0].dims[1];
    const Index input_size = input_views[0].dims[2];
    const Index output_size = get_outputs_number();

    Tensor<type, 2> initial_hidden_states(batch_size, output_size);
    initial_hidden_states.setZero();

    Tensor<type, 2> previous_hidden_states(batch_size, output_size);

    TensorMap<Tensor<type, 3>> inputs(input_views[0].data, batch_size, past_time_steps, input_size);
    TensorMap<Tensor<type, 2>> deltas(delta_views[0].data, batch_size, output_size);

    RecurrentForwardPropagation* recurrent_forward =
        static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

    RecurrentBackPropagation* recurrent_backward =
        static_cast<RecurrentBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& hidden_states = recurrent_forward->hidden_states;

    Tensor<type, 3>& input_deltas = recurrent_backward->input_deltas;
    Tensor<type, 2>& input_weight_deltas = recurrent_backward->input_weight_deltas;
    Tensor<type, 2>& recurrent_weight_deltas = recurrent_backward->recurrent_weight_deltas;
    Tensor<type, 1>& bias_deltas = recurrent_backward->bias_deltas;
    Tensor<type, 2>& current_combination_deltas = recurrent_backward->current_combination_deltas;

    Tensor<type, 3>& activation_derivatives = recurrent_forward->activation_derivatives;

    input_weight_deltas.setZero();
    recurrent_weight_deltas.setZero();
    bias_deltas.setZero();
    current_combination_deltas.setZero();

    Tensor<type, 2> combination_deltas(batch_size, output_size);

    Tensor<type, 2>& current_deltas = recurrent_backward->current_deltas;

    for(Index time_step = past_time_steps - 1; time_step >= 0; --time_step)
    {             
        if (time_step == past_time_steps - 1)
            current_deltas = deltas;
        else
            current_deltas = current_combination_deltas;

        combination_deltas.device(*thread_pool_device) =
            current_deltas * activation_derivatives.chip(time_step, 1);

        // Need

        input_weight_deltas.device(*thread_pool_device) +=
            inputs.chip(time_step, 1).contract(combination_deltas, axes(0,0));

        if (time_step == 0)
            previous_hidden_states.device(*thread_pool_device) = initial_hidden_states;
        else
            previous_hidden_states.device(*thread_pool_device) = hidden_states.chip(time_step - 1, 1);

        recurrent_weight_deltas.device(*thread_pool_device) +=
            previous_hidden_states.contract(combination_deltas, axes(0,0));

        bias_deltas.device(*thread_pool_device) +=
            combination_deltas.sum(array<Index, 1>({ 0 }));

        if(time_step == 0)
            current_combination_deltas.setZero();
        else
            current_combination_deltas.device(*thread_pool_device)
                = combination_deltas.contract(recurrent_weights.shuffle(array<Index,2>{{1,0}}), axes(1,0));

        input_deltas.chip(time_step, 1).device(*thread_pool_device)
            = combination_deltas.contract(input_weights.shuffle(array<Index,2>{{1,0}}), axes(1,0));
    }
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
         << "Time steps: " << get_input_dimensions()[0] << endl
         << "Input dimensions: " << get_input_dimensions()[1] << endl
         << "Output dimensions: " << get_output_dimensions()[0] << endl
         << "Biases dimensions: " << biases.dimensions() << endl
         << "Input weights dimensions: " << input_weights.dimensions() << endl
         << "Recurrent weights dimensions: " << recurrent_weights.dimensions() << endl;

    cout << "Biases:" << endl
         << biases << endl
         << "Input weights:" << endl
         << input_weights << endl
         << "Recurrent weights:" << endl
         << recurrent_weights << endl
         << "Activation function: " << activation_function << endl
         << "Total parameters: " << biases.size() + input_weights.size() + recurrent_weights.size() << endl;
}


void Recurrent::from_XML(const XMLDocument& document)
{
    const XMLElement* recurrent_layer_element = document.FirstChildElement("Recurrent");

    if(!recurrent_layer_element)
        throw runtime_error("Recurrent layer element is nullptr.\n");

    set_label(read_xml_string(recurrent_layer_element,"Label"));
    set_input_dimensions(string_to_dimensions(read_xml_string(recurrent_layer_element, "InputDimensions")));
    set_output_dimensions({ read_xml_index(recurrent_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(recurrent_layer_element, "Activation"));
    string_to_tensor<type, 1>(read_xml_string(recurrent_layer_element, "Biases"), biases);
    string_to_tensor<type, 2>(read_xml_string(recurrent_layer_element, "InputWeights"), input_weights);
    string_to_tensor<type, 2>(read_xml_string(recurrent_layer_element, "RecurrentWeights"), recurrent_weights);
}


void Recurrent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Recurrent");

    add_xml_element(printer, "Label", get_label());
    add_xml_element(printer, "InputDimensions", dimensions_to_string(get_input_dimensions()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "Biases", tensor_to_string<type, 1>(biases));
    add_xml_element(printer, "InputWeights", tensor_to_string<type, 2>(input_weights));
    add_xml_element(printer, "RecurrentWeights", tensor_to_string<type, 2>(recurrent_weights));

    printer.CloseElement();
}


RecurrentForwardPropagation::RecurrentForwardPropagation(const Index& new_batch_size, Layer* new_layer) : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView RecurrentForwardPropagation::get_output_pair() const
{
    const Index outputs_number = layer->get_outputs_number();

    return {(type*)outputs.data(), {{batch_size, outputs_number}}};
}


void RecurrentForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size=new_batch_size;

    layer = new_layer;
    if(layer == nullptr)
        throw runtime_error("recurrrent layer is nullptr");

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[1];
    const Index past_time_steps = layer->get_input_dimensions()[0];

    current_inputs.resize(batch_size, past_time_steps, inputs_number);

    current_activation_derivatives.resize(batch_size, outputs_number);

    activation_derivatives.resize(batch_size, past_time_steps, outputs_number);

    outputs.resize(batch_size, outputs_number);
    outputs.setZero();

    hidden_states.resize(batch_size, past_time_steps, outputs_number);
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
    const Index inputs_number = layer->get_input_dimensions()[1];
    const Index past_time_steps = layer->get_input_dimensions()[0];

    combinations_bias_deltas.resize(outputs_number, outputs_number);
    combinations_input_weight_deltas.resize(inputs_number, outputs_number, outputs_number);
    combinations_recurrent_weight_deltas.resize(outputs_number, outputs_number, outputs_number);
    combination_deltas.resize(batch_size, outputs_number);
    current_combination_deltas.resize(batch_size, outputs_number);
    bias_deltas.resize(outputs_number);
    input_weight_deltas.resize(inputs_number, outputs_number);
    recurrent_weight_deltas.resize(outputs_number, outputs_number);
    input_deltas.resize(batch_size, past_time_steps, inputs_number);

    input_weight_deltas.setZero();
    recurrent_weight_deltas.setZero();
    bias_deltas.setZero();
    input_deltas.setZero();
    current_combination_deltas.setZero();
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


vector<TensorView> RecurrentBackPropagation::get_input_derivative_views() const
{
    const Index past_time_steps = layer->get_input_dimensions()[0];
    const Index inputs_number = layer->get_input_dimensions()[1];

    return {{(type*)(input_deltas.data()), {batch_size, past_time_steps, inputs_number}}};
}


vector<ParameterView> RecurrentBackPropagation::get_parameter_delta_views() const
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
