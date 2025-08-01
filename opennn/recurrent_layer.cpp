//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
//#include "strings_utilities.h"
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
    return {input_weights.dimension(0), past_time_steps};
}


dimensions Recurrent::get_output_dimensions() const
{
    return { biases.size() };
}


Index Recurrent::get_timesteps() const
{
    return past_time_steps;
}


vector<pair<type *, Index> > Recurrent::get_parameter_pairs() const
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
    const Index inputs_number = get_inputs_number() - past_time_steps + 1;

    biases.resize(new_output_dimensions[0]);

    input_weights.resize(inputs_number, new_output_dimensions[0]);

    recurrent_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);
}


void Recurrent::set_timesteps(const Index& new_timesteps)
{
    past_time_steps = new_timesteps;
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


void Recurrent::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool& is_training)
{
    const Index batch_size = input_pairs[0].second[0];
    const Index past_time_steps = input_pairs[0].second[2];
    const Index input_size = input_pairs[0].second[1];
    const Index output_size = get_outputs_number();

    TensorMap<Tensor<type, 3>> inputs(input_pairs[0].first, batch_size, past_time_steps, input_size);

    RecurrentForwardPropagation* recurrent_forward =
        static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

    Tensor<type, 2>& outputs = recurrent_forward->outputs;
    Tensor<type, 3>& activation_derivatives = recurrent_forward->activation_derivatives;
    Tensor<type, 3>& hidden_states = recurrent_forward->hidden_states;

    Tensor<type, 2> previous_hidden_states(batch_size, output_size);
    previous_hidden_states.setZero();

    for (Index time_step = 0; time_step < past_time_steps; time_step++)
    {
        #pragma omp parallel for
        for (Index i = 0; i < batch_size; ++i)
            for (Index j = 0; j < output_size; ++j)
            {
                type sum = biases(j);

                for (Index k = 0; k < input_size; ++k)
                    sum += inputs(i, time_step, k) * input_weights(k, j);

                for (Index k = 0; k < output_size; ++k)
                    sum += previous_hidden_states(i, k) * recurrent_weights(k, j);

                outputs(i, j) = sum;
            }

        Tensor<type, 2> current_activation_derivatives = activation_derivatives.chip(time_step, 1);

        calculate_activations(activation_function, outputs, current_activation_derivatives);

        hidden_states.chip(time_step, 1) = outputs;
        previous_hidden_states = outputs;
    }
}


void Recurrent::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>& forward_propagation,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_pairs[0].second[0];
    const Index past_time_steps = input_pairs[0].second[2];
    const Index input_size = input_pairs[0].second[1];
    const Index output_size = get_outputs_number();

    TensorMap<Tensor<type, 3>> inputs(input_pairs[0].first, batch_size, past_time_steps, input_size);
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

    Tensor<type, 3>& activation_derivatives = recurrent_forward->activation_derivatives;

    input_weight_deltas.setZero();
    recurrent_weight_deltas.setZero();
    bias_deltas.setZero();

    Tensor<type, 2> next_hidden_state_delta(batch_size, output_size);
    next_hidden_state_delta.setZero();

    for (Index time_step = past_time_steps - 1; time_step >= 0; --time_step)
    {
        if (time_step == past_time_steps - 1)
            current_deltas = deltas;
        else
            current_deltas = next_hidden_state_delta;

        TensorMap<Tensor<type, 2>> current_act_derivs(activation_derivatives.data() + time_step * batch_size * output_size,
                                                      batch_size, output_size);

        #pragma omp parallel for
        for (Index i = 0; i < current_deltas.size(); ++i)
            current_deltas.data()[i] *= current_act_derivs.data()[i];

        TensorMap<Tensor<type, 2>> current_input(inputs.data() + time_step * batch_size * input_size,
                                                 batch_size, input_size);

        TensorMap<Tensor<type, 2>> prev_hidden((time_step == 0)
                                                   ? Tensor<type, 2>(batch_size, output_size).setZero().data()
                                                   : hidden_states.data() + (time_step - 1) * batch_size * output_size,
                                               batch_size, output_size);

        #pragma omp parallel for
        for (Index i = 0; i < input_size; ++i)
            for (Index j = 0; j < output_size; ++j)
            {
                type sum = 0;

                for (Index k = 0; k < batch_size; ++k)
                    sum += current_input(k, i) * current_deltas(k, j);

                #pragma omp atomic
                input_weight_deltas(i, j) += sum;
            }

        #pragma omp parallel for
        for (Index i = 0; i < output_size; ++i)
            for (Index j = 0; j < output_size; ++j)
            {
                type sum = 0;

                for (Index k = 0; k < batch_size; ++k)
                    sum += prev_hidden(k, i) * current_deltas(k, j);

                #pragma omp atomic
                recurrent_weight_deltas(i, j) += sum;
            }

        #pragma omp parallel for
        for (Index j = 0; j < output_size; ++j)
        {
            type sum = 0;

            for (Index i = 0; i < batch_size; ++i)
                sum += current_deltas(i, j);

            #pragma omp atomic
            bias_deltas(j) += sum;
        }

        if (time_step > 0)
        {
            #pragma omp parallel for
            for (Index i = 0; i < batch_size; ++i)
                for (Index j = 0; j < output_size; ++j)
                {
                    type sum = 0;

                    for (Index k = 0; k < output_size; ++k)
                        sum += current_deltas(i, k) * recurrent_weights(j, k);

                    next_hidden_state_delta(i, j) = sum;
                }
        }

        TensorMap<Tensor<type, 2>> current_input_delta(input_deltas.data() + time_step * batch_size * input_size,
                                                       batch_size, input_size);

        #pragma omp parallel for
        for (Index i = 0; i < batch_size; ++i)
            for (Index j = 0; j < input_size; ++j)
            {
                type sum = 0;

                for (Index k = 0; k < output_size; ++k)
                    sum += current_deltas(i, k) * input_weights(j, k);

                current_input_delta(i, j) = sum;
            }
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
         << "Time steps: " << get_input_dimensions()[1] << endl
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
    cout << "Activation function: " << activation_function << endl;
    cout << "Total parameters: " << biases.size() + input_weights.size() + recurrent_weights.size() << endl;
}


void Recurrent::from_XML(const XMLDocument& document)
{
    const XMLElement* recurrent_layer_element = document.FirstChildElement("Recurrent");

    if(!recurrent_layer_element)
        throw runtime_error("Recurrent layer element is nullptr.\n");

    set_timesteps(3);
    set_label(read_xml_string(recurrent_layer_element,"Label"));
    set_input_dimensions({ read_xml_index(recurrent_layer_element, "InputsNumber") });
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
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
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


pair<type*, dimensions> RecurrentForwardPropagation::get_output_pair() const
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
        throw std::runtime_error("recurrrent layer is nullptr");

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index past_time_steps = layer->get_input_dimensions()[1];

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
    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index past_time_steps = layer->get_input_dimensions()[1];

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
