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

vector<TensorSpec> Recurrent::get_forward_specs(Index batch_size) const
{
    const Shape state_history {batch_size, time_steps, output_features};
    const Shape output_shape = return_sequences ? state_history : Shape{batch_size, output_features};

    return {
        {state_history, compute_dtype},  // Forward::HiddenStates          (batch, time, out)
        {state_history, compute_dtype},  // Forward::ActivationDerivatives (batch, time, out)
        {output_shape,  compute_dtype},  // Forward::Output                — last
    };
}

vector<TensorSpec> Recurrent::get_backward_specs(Index batch_size) const
{
    if (!is_trainable) return {};

    const Shape input_delta_shape = Shape{batch_size}.append(get_input_shape());
    const Shape step_in_shape  {batch_size, input_features};
    const Shape step_out_shape {batch_size, output_features};

    return {
        {input_delta_shape, compute_dtype},  // InputDeltaSlot
        {step_in_shape,     compute_dtype},  // StepInputScratchSlot
        {step_out_shape,    compute_dtype},  // StepPrevHScratchSlot
        {step_out_shape,    compute_dtype},  // DeltaScratchSlot
        {step_out_shape,    compute_dtype},  // NextCarryScratchSlot
        {step_in_shape,     compute_dtype},  // StepInDeltaScratchSlot
    };
}

void Recurrent::configure_operators()
{
    recurrent_op.set(input_features, time_steps, output_features,
                     recurrent_op.activation, compute_dtype);

    recurrent_op.return_sequences = return_sequences;
    recurrent_op.input_slots  = {Input};
    recurrent_op.output_slots = {Output, HiddenStates, ActivationDerivatives};
}

void Recurrent::set_return_sequences(bool value)
{
    if (return_sequences == value) return;
    return_sequences = value;
    configure_operators();
}

void Recurrent::set(const Shape& new_input_shape,
                    const Shape& new_output_shape,
                    const string& new_activation_function,
                    const string& new_label)
{
    if (new_input_shape.empty() && new_output_shape.empty())
    {
        time_steps      = 0;
        input_features  = 0;
        output_features = 0;
        return;
    }

    check_rank(new_input_shape,  {2}, "Recurrent", "input");
    check_rank(new_output_shape, {1}, "Recurrent", "output");

    time_steps      = new_input_shape[0];
    input_features  = new_input_shape[1];
    output_features = new_output_shape[0];

    set_activation_function(new_activation_function);
    set_label(new_label);

    configure_operators();
}

void Recurrent::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {2}, "Recurrent", "input");
    time_steps     = new_input_shape[0];
    input_features = new_input_shape[1];
    configure_operators();
}

void Recurrent::set_output_shape(const Shape& new_output_shape)
{
    check_rank(new_output_shape, {1, 2}, "Recurrent", "output");
    output_features = new_output_shape[new_output_shape.rank - 1];
    configure_operators();
}

void Recurrent::set_activation_function(const string& name)
{
    const ActivationOp::Function fn = ActivationOp::from_string(name);
    if (fn == ActivationOp::Function::Softmax)
        throw runtime_error("Recurrent: Softmax activation is not supported (use Tanh, Sigmoid, ReLU or Identity).");
    recurrent_op.activation = fn;
}

void Recurrent::read_JSON_body(const Json* recurrent_layer_element)
{
    set_activation_function(read_json_string(recurrent_layer_element, "Activation"));
    set_return_sequences(read_json_bool(recurrent_layer_element, "ReturnSequences"));
}

void Recurrent::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "Activation", get_activation_function());
    add_json_field(printer, "ReturnSequences", to_string(return_sequences));
}

string Recurrent::write_expression(const vector<string>& feature_names,
                                   const vector<string>& output_names) const
{
    if (parameters.size() < 3 || !parameters[0].data || !parameters[1].data || !parameters[2].data)
        return {};

    VectorMap biases_map        = parameters[0].as_vector();
    MatrixMap input_w_map       = parameters[1].as_matrix();
    MatrixMap recurrent_w_map   = parameters[2].as_matrix();

    const string& activation_name = ActivationOp::to_string(recurrent_op.activation);

    auto step_var = [&](Index t, Index j) -> string {
        const string internal = format("recurrent_hidden_step_{}_neuron_{}", t, j);
        if (return_sequences)
        {
            const Index linear = t * output_features + j;
            if (linear < ssize(output_names)) return output_names[linear];
            return internal;
        }
        if (t == time_steps - 1)
        {
            if (j < ssize(output_names)) return output_names[j];
            return format("recurrent_output_{}", j);
        }
        return internal;
    };

    ostringstream buffer;
    buffer.precision(10);

    for (Index time_step = 0; time_step < time_steps; ++time_step)
    {
        for (Index j = 0; j < output_features; ++j)
        {
            const string current_var = step_var(time_step, j);
            buffer << current_var << " = " << activation_name << "( " << biases_map(j);

            for (Index i = 0; i < input_features; ++i)
            {
                const Index feature_index = time_step * input_features + i;
                if (feature_index < ssize(feature_names))
                    buffer << " + (" << feature_names[feature_index] << "*" << input_w_map(i, j) << ")";
            }

            if (time_step > 0)
                for (Index prev_j = 0; prev_j < output_features; ++prev_j)
                    buffer << " + (" << step_var(time_step - 1, prev_j)
                           << "*" << recurrent_w_map(prev_j, j) << ")";

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
