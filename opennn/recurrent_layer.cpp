//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "recurrent_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Recurrent::Recurrent(const Shape& new_input_shape,
                     const Shape& new_output_shape,
                     const string& new_activation_function,
                     const string& new_label)
    : Layer(LayerType::Recurrent)
{
    operators = {&recurrent_op};

    set(new_input_shape, new_output_shape, new_activation_function, new_label);
}

Shape Recurrent::get_output_shape() const
{
    return {output_features};
}

vector<TensorSpec> Recurrent::get_forward_specs(Index batch_size) const
{
    const Index T = get_time_steps();
    const Shape full = Shape{batch_size}.append(get_output_shape());

    return {
        {full,                              compute_dtype}, // Output (last hidden state)
        {{batch_size, T, output_features}, compute_dtype}, // AllHiddenStates (per-step scratch)
        {{batch_size, T, output_features}, compute_dtype}, // AllActivationDerivatives
    };
}

vector<TensorSpec> Recurrent::get_backward_specs(Index batch_size) const
{
    if (!is_trainable) return {};
    return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
}

void Recurrent::configure_operators()
{
    recurrent_op.set(get_input_features(),
                     output_features,
                     get_time_steps(),
                     recurrent_op.activation_function);

    recurrent_op.input_slots         = {Input};
    recurrent_op.output_slots        = {Output, AllHiddenStates, AllActivationDerivatives};
    recurrent_op.output_delta_slots  = {OutputDelta};
    recurrent_op.input_delta_slots   = {InputDelta};
}

void Recurrent::set(const Shape& new_input_shape,
                    const Shape& new_output_shape,
                    const string& new_activation_function,
                    const string& new_label)
{
    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        return;
    }

    check_rank(new_input_shape, {2}, "Recurrent", "input");
    check_rank(new_output_shape, {1}, "Recurrent", "output");

    input_shape     = new_input_shape;
    output_features = new_output_shape[0];

    set_activation_function(new_activation_function);
    set_label(new_label);

    configure_operators();
}

void Recurrent::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {2}, "Recurrent", "input");
    input_shape = new_input_shape;
    configure_operators();
}

void Recurrent::set_output_shape(const Shape& new_output_shape)
{
    output_features = new_output_shape.dim_or_zero(0);
    configure_operators();
}

void Recurrent::set_activation_function(const string& new_activation_function)
{
    const ActivationOp::Function function = ActivationOp::from_string(new_activation_function);

    using enum ActivationOp::Function;
    if (function != Identity && function != Sigmoid && function != Tanh && function != ReLU)
        throw runtime_error(format("Recurrent: unsupported activation function \"{}\".", new_activation_function));

    recurrent_op.activation_function = function;
}

string Recurrent::write_expression(const vector<string>& feature_names,
                                   const vector<string>& output_names) const
{
    if (!recurrent_op.biases.data
        || !recurrent_op.input_weights.data
        || !recurrent_op.recurrent_weights.data) return "";

    const Index T = get_time_steps();
    const Index inputs_number = get_input_features();
    const Index outputs_number = get_output_features();

    const VectorMap biases_map            = recurrent_op.biases.as_vector();
    const MatrixMap input_to_hidden       = recurrent_op.input_weights.as_matrix();
    const MatrixMap hidden_to_hidden      = recurrent_op.recurrent_weights.as_matrix();

    const string& activation_name = ActivationOp::to_string(recurrent_op.activation_function);

    ostringstream buffer;
    buffer.precision(10);

    for (Index time_step = 0; time_step < T; ++time_step)
    {
        for (Index j = 0; j < outputs_number; ++j)
        {
            string current_variable_name;

            if (time_step == T - 1)
                current_variable_name = (j < ssize(output_names))
                    ? output_names[j]
                    : format("recurrent_output_{}", j);
            else
                current_variable_name = format("recurrent_hidden_step_{}_neuron_{}", time_step, j);

            buffer << current_variable_name << " = " << activation_name << "( " << biases_map(j);

            for (Index i = 0; i < inputs_number; ++i)
            {
                const Index feature_index = (time_step * inputs_number) + i;

                if (feature_index < ssize(feature_names))
                    buffer << " + (" << feature_names[feature_index] << "*" << input_to_hidden(i, j) << ")";
            }

            if (time_step > 0)
            {
                for (Index previous_j = 0; previous_j < outputs_number; ++previous_j)
                {
                    const string previous_variable_name = format("recurrent_hidden_step_{}_neuron_{}", time_step - 1, previous_j);
                    buffer << " + (" << previous_variable_name << "*" << hidden_to_hidden(previous_j, j) << ")";
                }
            }

            buffer << " );\n";
        }
    }

    return buffer.str();
}

void Recurrent::read_JSON_body(const Json* recurrent_layer_element)
{
    set_activation_function(read_json_string(recurrent_layer_element, "Activation"));
}

void Recurrent::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "Activation", ActivationOp::to_string(recurrent_op.activation_function));
}

REGISTER(Layer, Recurrent, "Recurrent")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
