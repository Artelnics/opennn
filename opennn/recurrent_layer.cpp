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

Recurrent::Recurrent(const Shape& new_input_shape,
                     const Shape& new_output_shape) : Layer()
{
    set(new_input_shape, new_output_shape);
}


Shape Recurrent::get_input_shape() const
{
    return input_shape;
}


Shape Recurrent::get_output_shape() const
{
    return { biases.size() };
}


vector<TensorView*> Recurrent::get_parameter_views()
{  
    return {&biases, &input_weights, &recurrent_weights};
}


string Recurrent::get_activation_function() const
{
    return activation_function;
}


void Recurrent::set(const Shape& new_input_shape, const Shape& new_output_shape)
{
    if(new_input_shape.size() != 2)
        throw runtime_error("Input shape rank is not 2 for Recurrent (time_steps, inputs).");

    input_shape = new_input_shape;

    const Index inputs_number = new_input_shape[1];
    const Index outputs_number = new_output_shape[0];

    biases.shape = {outputs_number};
    input_weights.shape = {inputs_number, outputs_number};
    recurrent_weights.shape = {outputs_number, outputs_number};

    label = "recurrent_layer";
    name = "Recurrent";
}


void Recurrent::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.size() != 2)
        throw runtime_error("Input shape rank is not 2 for Recurrent (time_steps, inputs).");

    input_shape = new_input_shape;

    const Index inputs_number = input_shape[1];
    const Index outputs_number = get_outputs_number();

    input_weights.shape = {inputs_number, outputs_number};
}


void Recurrent::set_output_shape(const Shape& new_output_shape)
{
    const Index inputs_number = input_weights.shape[0];
    const Index outputs_number = new_output_shape[0];

    biases.shape = {outputs_number};
    input_weights.shape = {inputs_number, outputs_number};
    recurrent_weights.shape = {outputs_number, outputs_number};
}


void Recurrent::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Sigmoid"
    || new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "ScaledExponentialLinear")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}


void Recurrent::forward_propagate(const vector<TensorView>& input_views,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  bool is_training)
{
    const Index batch_size = input_views[0].shape[0];
    const Index past_time_steps = input_views[0].shape[1];
    const Index input_size = input_views[0].shape[2];
    const Index output_size = biases.shape[0];

    const VectorMap biases_vec = vector_map(biases);
    const MatrixMap input_weights_mat = matrix_map(input_weights);
    const MatrixMap recurrent_weights_mat = matrix_map(recurrent_weights);

    RecurrentForwardPropagation* recurrent_fp = static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

    TensorMap3 input_sequences = tensor_map<3>(input_views[0]);
    TensorMap3 all_hidden_states = tensor_map<3>(recurrent_fp->hidden_states);
    TensorMap3 all_activation_derivatives = tensor_map<3>(recurrent_fp->activation_derivatives);

    MatrixR current_hidden_state(batch_size, output_size);
    current_hidden_state.setZero();

    MatrixR step_input(batch_size, input_size);
    MatrixR previous_hidden_state(batch_size, output_size);

    for(Index t = 0; t < past_time_steps; t++)
    {
        MatrixMap(step_input.data(), batch_size, input_size) = input_sequences.chip(t, 1);

        calculate_combinations<2>(
            MatrixMap(step_input.data(), batch_size, input_size),
            input_weights_mat,
            biases_vec,
            MatrixMap(current_hidden_state.data(), batch_size, output_size)
            );

        if(t > 0)
        {
            MatrixMap(previous_hidden_state.data(), batch_size, output_size) = all_hidden_states.chip(t - 1, 1);

            current_hidden_state.noalias() += previous_hidden_state * recurrent_weights_mat;
        }

        if(is_training)
        {
            MatrixR step_derivatives(batch_size, output_size);

            calculate_activations<2>(
                activation_function,
                MatrixMap(current_hidden_state.data(), batch_size, output_size),
                MatrixMap(step_derivatives.data(), batch_size, output_size)
                );

            all_activation_derivatives.chip(t, 1) = MatrixMap(step_derivatives.data(), batch_size, output_size);
        }
        else
            calculate_activations<2>(
                activation_function,
                MatrixMap(current_hidden_state.data(), batch_size, output_size),
                MatrixMap(empty_2.data(), empty_2.dimensions())
                );

        all_hidden_states.chip(t, 1) = MatrixMap(current_hidden_state.data(), batch_size, output_size);
    }

    MatrixMap outputs_mat = matrix_map(recurrent_fp->outputs);
    MatrixMap(outputs_mat.data(), batch_size, output_size) = all_hidden_states.chip(past_time_steps - 1, 1);
}


void Recurrent::back_propagate(const vector<TensorView>& input_views,
                               const vector<TensorView>& output_gradient_views,
                               unique_ptr<LayerForwardPropagation>& forward_propagation,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_views[0].shape[0];
    const Index past_time_steps = input_views[0].shape[1];
    const Index neurons_number = biases.shape[0];

    TensorMap3 input_sequences = tensor_map<3>(input_views[0]);
    MatrixMap external_output_gradients = matrix_map(output_gradient_views[0]);

    RecurrentForwardPropagation* recurrent_forward_propagation = static_cast<RecurrentForwardPropagation*>(forward_propagation.get());
    RecurrentBackPropagation* recurrent_back_propagation = static_cast<RecurrentBackPropagation*>(back_propagation.get());

    MatrixMap input_to_hidden_weights_map = matrix_map(input_weights);
    MatrixMap hidden_to_hidden_weights_map = matrix_map(recurrent_weights);

    VectorMap bias_gradients_map = vector_map(recurrent_back_propagation->bias_gradients);
    MatrixMap input_weight_gradients_map = matrix_map(recurrent_back_propagation->input_weight_gradients);
    MatrixMap recurrent_weight_gradients_map = matrix_map(recurrent_back_propagation->recurrent_weight_gradients);
    TensorMap3 input_gradients_map = tensor_map<3>(recurrent_back_propagation->input_gradients[0]);

    bias_gradients_map.setZero();
    input_weight_gradients_map.setZero();
    recurrent_weight_gradients_map.setZero();

    TensorMap3 all_hidden_states = tensor_map<3>(recurrent_forward_propagation->hidden_states);
    TensorMap3 all_activation_derivatives = tensor_map<3>(recurrent_forward_propagation->activation_derivatives);

    Tensor2 current_error_delta(batch_size, neurons_number);
    current_error_delta.setZero();

    Tensor2 error_delta_from_next_step(batch_size, neurons_number);
    error_delta_from_next_step.setZero();

    for(Index t = past_time_steps - 1; t >= 0; t--)
    {
        if(t == past_time_steps - 1)
            current_error_delta.device(get_device()) = external_output_gradients;
        else
            current_error_delta.device(get_device()) = error_delta_from_next_step;

        Tensor2 der_eval = all_activation_derivatives.chip(t, 1);
        current_error_delta.device(get_device()) = current_error_delta * der_eval;

        Tensor2 input_eval = input_sequences.chip(t, 1);
        input_weight_gradients_map.device(get_device()) += input_eval.contract(current_error_delta, axes(0, 0));

        if(t > 0)
        {
            Tensor2 h_prev_eval = all_hidden_states.chip(t - 1, 1);
            recurrent_weight_gradients_map.device(get_device()) += h_prev_eval.contract(current_error_delta, axes(0, 0));
        }

        bias_gradients_map.device(get_device()) += current_error_delta.sum(array_1(0));

        if(!recurrent_back_propagation->is_first_layer)
            input_gradients_map.chip(t, 1).device(get_device()) = current_error_delta.contract(input_to_hidden_weights_map, axes(1, 1));

        if (t > 0)
            error_delta_from_next_step.device(get_device()) = current_error_delta.contract(hidden_to_hidden_weights_map, axes(1, 1));
    }
}


string Recurrent::get_expression(const vector<string>& feature_names,
                                 const vector<string>& output_names) const
{
    const Index time_steps = input_shape[0];
    const Index inputs_number = input_shape[1];
    const Index outputs_number = get_outputs_number();

    const vector<string> final_feature_names = feature_names.empty()
                                                   ? get_default_feature_names()
                                                   : feature_names;

    const vector<string> final_output_names = output_names.empty()
                                                  ? get_default_output_names()
                                                  : output_names;

    VectorMap biases_map = vector_map(biases);
    MatrixMap input_to_hidden_weights_map = matrix_map(input_weights);
    MatrixMap hidden_to_hidden_weights_map = matrix_map(recurrent_weights);

    ostringstream buffer;
    buffer.precision(10);

    for(Index time_step = 0; time_step < time_steps; time_step++)
    {
        for(Index j = 0; j < outputs_number; j++)
        {
            string current_variable_name;

            if(time_step == time_steps - 1)
            {
                if(j < static_cast<Index>(final_output_names.size()))
                    current_variable_name = final_output_names[j];
                else
                    current_variable_name = "recurrent_output_" + to_string(j);
            }
            else
                current_variable_name = "recurrent_hidden_step_" + to_string(time_step) + "_neuron_" + to_string(j);

            buffer << current_variable_name << " = " << activation_function << "( " << biases_map(j);

            for(Index i = 0; i < inputs_number; i++)
            {
                Index feature_index = (time_step * inputs_number) + i;

                if(feature_index < static_cast<Index>(final_feature_names.size()))
                    buffer << " + (" << final_feature_names[feature_index] << "*" << input_to_hidden_weights_map(i, j) << ")";
            }

            if(time_step > 0)
            {
                for(Index previous_j = 0; previous_j < outputs_number; previous_j++)
                {
                    string previous_variable_name = "recurrent_hidden_step_" + to_string(time_step - 1) + "_neuron_" + to_string(previous_j);
                    buffer << " + (" << previous_variable_name << "*" << hidden_to_hidden_weights_map(previous_j, j) << ")";
                }
            }

            buffer << " );\n";
        }
    }

    return buffer.str();
}


void Recurrent::print() const
{
/*
    cout << "Recurrent layer" << endl
         << "Time steps: " << get_input_shape()[0] << endl
         << "Input shape: " << get_input_shape()[1] << endl
         << "Output shape: " << get_output_shape()[0] << endl
         << "Biases shape: " << biases.shape << endl
         << "Input weights shape: " << input_weights.shape << endl
         << "Recurrent weights shape: " << recurrent_weights.shape << endl
         << "Total parameters: " << biases.size() + input_weights.size() + recurrent_weights.size() << endl;
*/
}


void Recurrent::from_XML(const XMLDocument& document)
{
    const XMLElement* recurrent_layer_element = document.FirstChildElement("Recurrent");

    if(!recurrent_layer_element)
        throw runtime_error("Recurrent layer element is nullptr.\n");

    set_label(read_xml_string(recurrent_layer_element,"Label"));
    set_input_shape(string_to_shape(read_xml_string(recurrent_layer_element, "InputDimensions")));
    set_output_shape({ read_xml_index(recurrent_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(recurrent_layer_element, "Activation"));
}


void Recurrent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Recurrent");

    add_xml_element(printer, "Label", get_label());
    add_xml_element(printer, "InputDimensions", shape_to_string(get_input_shape()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_shape()[0]));
    add_xml_element(printer, "Activation", activation_function);

    printer.CloseElement();
}


RecurrentForwardPropagation::RecurrentForwardPropagation(const Index new_batch_size, Layer* new_layer) : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void RecurrentForwardPropagation::initialize()
{
    if(layer == nullptr)
        throw runtime_error("Recurrrent layer is nullptr");

    const Index batch = batch_size;
    const Index outputs_num = layer->get_outputs_number();
    const Index steps = layer->get_input_shape()[0];

    outputs.shape = {batch, outputs_num};
    hidden_states.shape = {batch, steps, outputs_num};
    activation_derivatives.shape = {batch, steps, outputs_num};
}


vector<TensorView *> RecurrentForwardPropagation::get_workspace_views()
{
    return {&outputs, &hidden_states, &activation_derivatives};
}


void RecurrentForwardPropagation::print() const
{
    cout << "Recurrent forward propagation" << endl
         << "Batch size: " << batch_size << endl
         << "Output shape: " << outputs.shape << endl
         << "Hidden states shape: " << hidden_states.shape << endl
         << "Activation derivatives shape: " << activation_derivatives.shape << endl;
}


void RecurrentBackPropagation::initialize()
{
    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_shape()[1];
    const Index time_steps = layer->get_input_shape()[0];

    bias_gradients.shape = {outputs_number};
    input_weight_gradients.shape = {inputs_number, outputs_number};
    recurrent_weight_gradients.shape = {outputs_number, outputs_number};

    const Shape full_input_shape = { batch_size, time_steps, inputs_number };

    input_gradients_memory.resize(1);
    input_gradients_memory[0].resize(full_input_shape.count());
    input_gradients.resize(1);
    input_gradients[0].data = input_gradients_memory[0].data();
    input_gradients[0].shape = full_input_shape;
}


void RecurrentBackPropagation::print() const
{
    cout << "Recurrent back propagation" << endl
         << "Batch size: " << batch_size << endl
         << "Input gradients number: " << input_gradients_memory.size() << endl
         << "Bias gradients shape: " << bias_gradients.shape << endl
         << "Input weight gradients shape: " << input_weight_gradients.shape << endl
         << "Recurrent weight gradients shape: " << recurrent_weight_gradients.shape << endl;
}


RecurrentBackPropagation::RecurrentBackPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<TensorView*> RecurrentBackPropagation::get_gradient_views()
{
    return {&bias_gradients, &input_weight_gradients, &recurrent_weight_gradients};
}


REGISTER(Layer, Recurrent, "Recurrent")
REGISTER(LayerForwardPropagation, RecurrentForwardPropagation, "Recurrent")
REGISTER(LayerBackPropagation, RecurrentBackPropagation, "Recurrent")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
