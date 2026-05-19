//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "recurrent_layer.h"
#include "loss.h"

namespace opennn
{

Recurrent::Recurrent(const Shape& new_input_shape,
                     const Shape& new_output_shape)
    : Layer(LayerType::Recurrent)
{
    set(new_input_shape, new_output_shape);
}

Shape Recurrent::get_output_shape() const
{
    return { biases.size() };
}

vector<TensorSpec> Recurrent::get_parameter_specs() const
{
    return {
        {biases.shape,            compute_dtype},
        {input_weights.shape,     compute_dtype},
        {recurrent_weights.shape, compute_dtype},
    };
}

vector<TensorSpec> Recurrent::get_forward_specs(Index batch_size) const
{
    const Index outputs_number = get_outputs_number();

    return {
        {{batch_size, outputs_number},             compute_dtype},
        {{batch_size, time_steps, outputs_number}, compute_dtype},
        {{batch_size, time_steps, outputs_number}, compute_dtype},
    };
}

vector<TensorSpec> Recurrent::get_backward_specs(Index batch_size) const
{
    return {{{batch_size, time_steps, input_features}, compute_dtype}};
}

void Recurrent::set(const Shape& new_input_shape, const Shape& new_output_shape)
{
    if (new_input_shape.rank != 2)
        throw runtime_error("Input shape rank is not 2 for Recurrent (time_steps, inputs).");

    time_steps     = new_input_shape[0];
    input_features = new_input_shape[1];

    const Index outputs_number = new_output_shape.dim_or_zero(0);

    biases.shape            = {outputs_number};
    input_weights.shape     = {input_features, outputs_number};
    recurrent_weights.shape = {outputs_number, outputs_number};

    set_label("recurrent_layer");
}

void Recurrent::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 2)
        throw runtime_error("Input shape rank is not 2 for Recurrent (time_steps, inputs).");

    time_steps = new_input_shape[0];
    input_features = new_input_shape[1];

    const Index outputs_number = get_outputs_number();

    input_weights.shape = {input_features, outputs_number};
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
    if (new_activation_function != "Sigmoid"
        && new_activation_function != "Tanh"
        && new_activation_function != "Identity"
        && new_activation_function != "ReLU")
        throw runtime_error(format("Unknown activation function: {}", new_activation_function));

    activation_function = new_activation_function;
}

void Recurrent::forward_propagate(ForwardPropagation& /*forward_propagation*/, size_t /*index*/, bool /*is_training*/) noexcept
{
/*
    const Index batch_size = forward_propagation->inputs[0].shape[0];
    const Index past_time_steps = forward_propagation->inputs[0].shape[1];
    const Index input_features = forward_propagation->inputs[0].shape[2];
    const Index output_size = biases.shape[0];

    const VectorMap biases_map = biases.as_vector();
    const MatrixMap input_weights_map = input_weights.as_matrix();
    const MatrixMap recurrent_weights_map = recurrent_weights.as_matrix();

    TensorMap3 input_sequences = forward_propagation->inputs[0].as_tensor<3>();
    TensorMap3 all_hidden_states = recurrent_forward_propagation->hidden_states.as_tensor<3>();
    TensorMap3 all_activation_derivatives = recurrent_forward_propagation->activation_derivatives.as_tensor<3>();

    MatrixR current_hidden_state = MatrixR::Zero(batch_size, output_size);

    MatrixR step_input(batch_size, input_features);
    MatrixR previous_hidden_state(batch_size, output_size);
    MatrixR step_derivatives(batch_size, output_size);

    for (Index t = 0; t < past_time_steps; ++t)
    {
        TensorMap2 step_input_tensor(step_input.data(), batch_size, input_features);
        TensorMap2 current_hidden_tensor(current_hidden_state.data(), batch_size, output_size);

        step_input_tensor = input_sequences.chip(t, 1);

        calculate_combinations<2>(
            step_input_tensor,
            input_weights_map,
            biases_map,
            current_hidden_tensor
            );

        if (t > 0)
        {
            TensorMap2(previous_hidden_state.data(), batch_size, output_size) = all_hidden_states.chip(t - 1, 1);

            current_hidden_state.noalias() += previous_hidden_state * recurrent_weights_map;
        }

        if (is_training)
        {
            TensorMap2 step_derivatives_tensor(step_derivatives.data(), batch_size, output_size);

            calculate_activations<2>(
                activation_function,
                current_hidden_tensor,
                step_derivatives_tensor
                );

            all_activation_derivatives.chip(t, 1) = step_derivatives_tensor;
        }
        else
        {
            calculate_activations<2>(
                activation_function,
                current_hidden_tensor,
                TensorMap2(empty_2.data(), empty_2.dimensions())
                );
        }

        all_hidden_states.chip(t, 1) = current_hidden_tensor;
    }

    if (recurrent_forward_propagation->outputs.data)
    {
        TensorMap2 outputs_map(recurrent_forward_propagation->outputs.data, batch_size, output_size);
        outputs_map = all_hidden_states.chip(past_time_steps - 1, 1);
    }

    MatrixMap outputs_map = recurrent_forward_propagation->outputs.as_matrix();
    TensorMap2(outputs_map.data(), batch_size, output_size) = all_hidden_states.chip(past_time_steps - 1, 1);
*/

}

void Recurrent::back_propagate(ForwardPropagation& /*forward_propagation*/,
                               BackPropagation& /*back_propagation*/,
                               size_t /*index*/) const noexcept
{
/*
    const Index batch_size = forward_propagation->inputs[0].shape[0];
    const Index past_time_steps = forward_propagation->inputs[0].shape[1];
    const Index input_features = forward_propagation->inputs[0].shape[2];
    const Index output_features = biases.shape[0];

    const MatrixMap W_in = input_weights.as_matrix();
    const MatrixMap W_rec = recurrent_weights.as_matrix();
    const MatrixMap external_output_deltas = back_propagation->output_deltas[0].as_matrix();

    VectorMap bias_gradients = recurrent_bp->bias_gradients.as_vector();
    MatrixMap input_weight_gradients = recurrent_bp->input_weight_gradients.as_matrix();
    MatrixMap recurrent_weight_gradients = recurrent_bp->recurrent_weight_gradients.as_matrix();

    TensorMap3 input_sequence_gradient = recurrent_bp->input_deltas[0].as_tensor<3>();

    bias_gradients.setZero();
    input_weight_gradients.setZero();
    recurrent_weight_gradients.setZero();

    RecurrentForwardPropagation* recurrent_fp = static_cast<RecurrentForwardPropagation*>(forward_propagation.get());
    TensorMap3 input_sequences = forward_propagation->inputs[0].as_tensor<3>();
    TensorMap3 all_hidden_states = recurrent_fp->hidden_states.as_tensor<3>();
    TensorMap3 all_activation_derivatives = recurrent_fp->activation_derivatives.as_tensor<3>();

    MatrixR output_delta(batch_size, output_features);
    MatrixR next_step_gradient = MatrixR::Zero(batch_size, output_features);

    MatrixR step_input(batch_size, input_features);
    MatrixR step_derivatives(batch_size, output_features);
    MatrixR step_prev_hidden(batch_size, output_features);

    for (Index t = past_time_steps - 1; t >= 0; t--)
    {
        if (t == past_time_steps - 1)
            output_delta = external_output_deltas;
        else
            output_delta = next_step_gradient;

        TensorMap2(step_derivatives.data(), batch_size, output_features) = all_activation_derivatives.chip(t, 1);
        output_delta.array() *= step_derivatives.array();

        TensorMap2(step_input.data(), batch_size, input_features) = input_sequences.chip(t, 1);
        input_weight_gradients.noalias() += step_input.transpose() * output_delta;

        if (t > 0)
        {
            TensorMap2(step_prev_hidden.data(), batch_size, output_features) = all_hidden_states.chip(t - 1, 1);
            recurrent_weight_gradients.noalias() += step_prev_hidden.transpose() * output_delta;
        }

        bias_gradients.noalias() += output_delta.colwise().sum();

        if (!is_first_layer)
        {
            MatrixR input_step_gradient = output_delta * W_in.transpose();

            input_sequence_gradient.chip(t, 1) = TensorMap2(input_step_gradient.data(), batch_size, input_features);
        }

        if (t > 0)
            next_step_gradient.noalias() = output_delta * W_rec.transpose();
    }
*/
}

void Recurrent::read_JSON_body(const Json* recurrent_layer_element)
{
    set_activation_function(read_json_string(recurrent_layer_element, "Activation"));
}

void Recurrent::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "Activation", activation_function);
}

string Recurrent::write_expression(const vector<string>& feature_names,
                                   const vector<string>& output_names) const
{
    const Shape input_shape_local = get_input_shape();
    const Index time_steps_local = input_shape_local[0];
    const Index inputs_number = input_shape_local[1];
    const Index outputs_number = get_outputs_number();

    VectorMap biases_map = get_biases().as_vector();
    MatrixMap input_to_hidden_weights_map = get_input_weights().as_matrix();
    MatrixMap hidden_to_hidden_weights_map = get_recurrent_weights().as_matrix();

    const string& activation_function_local = get_activation_function();

    ostringstream buffer;
    buffer.precision(10);

    for (Index time_step = 0; time_step < time_steps_local; ++time_step)
    {
        for (Index j = 0; j < outputs_number; ++j)
        {
            string current_variable_name;

            if (time_step == time_steps_local - 1)
                current_variable_name = (j < ssize(output_names))
                    ? output_names[j]
                    : format("recurrent_output_{}", j);
            else
                current_variable_name = format("recurrent_hidden_step_{}_neuron_{}", time_step, j);

            buffer << current_variable_name << " = " << activation_function_local << "( " << biases_map(j);

            for (Index i = 0; i < inputs_number; ++i)
            {
                const Index feature_index = (time_step * inputs_number) + i;

                if (feature_index < ssize(feature_names))
                    buffer << " + (" << feature_names[feature_index] << "*" << input_to_hidden_weights_map(i, j) << ")";
            }

            if (time_step > 0)
            {
                for (Index previous_j = 0; previous_j < outputs_number; ++previous_j)
                {
                    string previous_variable_name = format("recurrent_hidden_step_{}_neuron_{}", time_step - 1, previous_j);
                    buffer << " + (" << previous_variable_name << "*" << hidden_to_hidden_weights_map(previous_j, j) << ")";
                }
            }

            buffer << " );\n";
        }
    }

    return buffer.str();
}

REGISTER(Layer, Recurrent, "Recurrent")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
