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
                     const Shape& new_output_shape) : Layer()
{
    set(new_input_shape, new_output_shape);
}

// Getters

Shape Recurrent::get_output_shape() const
{
    return { biases.size() };
}

vector<Shape> Recurrent::get_parameter_shapes() const
{
    return {biases.shape, input_weights.shape, recurrent_weights.shape};
}

// Setters

void Recurrent::set(const Shape& new_input_shape, const Shape& new_output_shape)
{
    if(new_input_shape.rank() != 2)
        throw runtime_error("Input shape rank is not 2 for Recurrent (time_steps, inputs).");

    time_steps = new_input_shape[0];
    input_features = new_input_shape[1];

    const Index inputs_number = input_features;
    const Index outputs_number = new_output_shape[0];

    biases.shape = {outputs_number};
    input_weights.shape = {inputs_number, outputs_number};
    recurrent_weights.shape = {outputs_number, outputs_number};

    label = "recurrent_layer";
    name = "Recurrent";
    layer_type = LayerType::Recurrent;
}

void Recurrent::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank() != 2)
        throw runtime_error("Input shape rank is not 2 for Recurrent (time_steps, inputs).");

    time_steps = new_input_shape[0];
    input_features = new_input_shape[1];

    const Index inputs_number = input_features;
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
    string normalized_activation_function = new_activation_function;

    if(normalized_activation_function == "Logistic")
        normalized_activation_function = "Sigmoid";

    if(normalized_activation_function == "Sigmoid"
        || normalized_activation_function == "HyperbolicTangent"
        || normalized_activation_function == "Linear"
        || normalized_activation_function == "RectifiedLinear"
        || normalized_activation_function == "ScaledExponentialLinear")
        activation_function = normalized_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}

// Forward / back propagation

void Recurrent::forward_propagate(ForwardPropagation& forward_propagation, size_t index, bool is_training) noexcept
{
/*
    const Index batch_size = forward_propagation->inputs[0].shape[0];
    const Index past_time_steps = forward_propagation->inputs[0].shape[1];
    const Index input_features = forward_propagation->inputs[0].shape[2];
    const Index output_size = biases.shape[0];

    const VectorMap biases_map = vector_map(biases);
    const MatrixMap input_weights_map = matrix_map(input_weights);
    const MatrixMap recurrent_weights_map = matrix_map(recurrent_weights);

    TensorMap3 input_sequences = tensor_map<3>(forward_propagation->inputs[0]);
    TensorMap3 all_hidden_states = tensor_map<3>(recurrent_forward_propagation->hidden_states);
    TensorMap3 all_activation_derivatives = tensor_map<3>(recurrent_forward_propagation->activation_derivatives);

    MatrixR current_hidden_state = MatrixR::Zero(batch_size, output_size);

    MatrixR step_input(batch_size, input_features);
    MatrixR previous_hidden_state(batch_size, output_size);
    MatrixR step_derivatives(batch_size, output_size);

    for(Index t = 0; t < past_time_steps; ++t)
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

        if(t > 0)
        {
            TensorMap2(previous_hidden_state.data(), batch_size, output_size) = all_hidden_states.chip(t - 1, 1);

            current_hidden_state.noalias() += previous_hidden_state * recurrent_weights_map;
        }

        if(is_training)
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

    if(recurrent_forward_propagation->outputs.data)
    {
        TensorMap2 outputs_map(recurrent_forward_propagation->outputs.data, batch_size, output_size);
        outputs_map = all_hidden_states.chip(past_time_steps - 1, 1);
    }

    MatrixMap outputs_map = matrix_map(recurrent_forward_propagation->outputs);
    TensorMap2(outputs_map.data(), batch_size, output_size) = all_hidden_states.chip(past_time_steps - 1, 1);
*/

}

void Recurrent::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t index) const noexcept
{
/*
    const Index batch_size = forward_propagation->inputs[0].shape[0];
    const Index past_time_steps = forward_propagation->inputs[0].shape[1];
    const Index input_features = forward_propagation->inputs[0].shape[2];
    const Index output_features = biases.shape[0];

    const MatrixMap W_in = matrix_map(input_weights);
    const MatrixMap W_rec = matrix_map(recurrent_weights);
    const MatrixMap external_output_gradients = matrix_map(back_propagation->output_gradients[0]);

    VectorMap bias_gradients = vector_map(recurrent_bp->bias_gradients);
    MatrixMap input_weight_gradients = matrix_map(recurrent_bp->input_weight_gradients);
    MatrixMap recurrent_weight_gradients = matrix_map(recurrent_bp->recurrent_weight_gradients);

    TensorMap3 input_sequence_gradient = tensor_map<3>(recurrent_bp->input_gradients[0]);

    bias_gradients.setZero();
    input_weight_gradients.setZero();
    recurrent_weight_gradients.setZero();

    RecurrentForwardPropagation* recurrent_fp = static_cast<RecurrentForwardPropagation*>(forward_propagation.get());
    TensorMap3 input_sequences = tensor_map<3>(forward_propagation->inputs[0]);
    TensorMap3 all_hidden_states = tensor_map<3>(recurrent_fp->hidden_states);
    TensorMap3 all_activation_derivatives = tensor_map<3>(recurrent_fp->activation_derivatives);

    MatrixR output_gradient(batch_size, output_features);
    MatrixR next_step_gradient = MatrixR::Zero(batch_size, output_features);

    MatrixR step_input(batch_size, input_features);
    MatrixR step_derivatives(batch_size, output_features);
    MatrixR step_prev_hidden(batch_size, output_features);

    for(Index t = past_time_steps - 1; t >= 0; t--)
    {
        if(t == past_time_steps - 1)
            output_gradient = external_output_gradients;
        else
            output_gradient = next_step_gradient;

        TensorMap2(step_derivatives.data(), batch_size, output_features) = all_activation_derivatives.chip(t, 1);
        output_gradient.array() *= step_derivatives.array();

        TensorMap2(step_input.data(), batch_size, input_features) = input_sequences.chip(t, 1);
        input_weight_gradients.noalias() += step_input.transpose() * output_gradient;

        if(t > 0)
        {
            TensorMap2(step_prev_hidden.data(), batch_size, output_features) = all_hidden_states.chip(t - 1, 1);
            recurrent_weight_gradients.noalias() += step_prev_hidden.transpose() * output_gradient;
        }

        bias_gradients.noalias() += output_gradient.colwise().sum();

        if(!is_first_layer)
        {
            MatrixR input_step_gradient = output_gradient * W_in.transpose();

            input_sequence_gradient.chip(t, 1) = TensorMap2(input_step_gradient.data(), batch_size, input_features);
        }

        if(t > 0)
            next_step_gradient.noalias() = output_gradient * W_rec.transpose();
    }
*/
}

// Serialization

void Recurrent::from_XML(const XmlDocument& document)
{
    const XmlElement* recurrent_layer_element = get_xml_root(document, "Recurrent");

    set_label(read_xml_string(recurrent_layer_element,"Label"));
    set_input_shape(string_to_shape(read_xml_string(recurrent_layer_element, "InputDimensions")));
    set_output_shape({ read_xml_index(recurrent_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(recurrent_layer_element, "Activation"));
}

void Recurrent::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Recurrent");

    write_xml_properties(printer, {
        {"Label", get_label()},
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"NeuronsNumber", to_string(get_output_shape()[0])},
        {"Activation", activation_function}
    });

    printer.close_element();
}

REGISTER(Layer, Recurrent, "Recurrent")
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
