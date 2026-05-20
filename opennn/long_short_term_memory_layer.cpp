//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "long_short_term_memory_layer.h"

namespace opennn
{

LongShortTermMemory::LongShortTermMemory(const Shape& new_input_shape,
                                         const Shape& new_output_shape,
                                         const string& new_activation_function,
                                         const string& new_recurrent_activation_function,
                                         const string& new_label)
    : Layer(LayerType::LongShortTermMemory)
{
    operators = {&lstm_op};

    set(new_input_shape,
        new_output_shape,
        new_activation_function,
        new_recurrent_activation_function,
        new_label);
}

vector<TensorSpec> LongShortTermMemory::get_forward_specs(Index batch_size) const
{
    const Index T = get_time_steps();
    const Shape output_shape = Shape{batch_size}.append(get_output_shape());
    const Shape sequence_shape{batch_size, T, output_features};

    return {
        {output_shape,    compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
    };
}

vector<TensorSpec> LongShortTermMemory::get_backward_specs(Index batch_size) const
{
    if (!is_trainable) return {};

    const Shape input_delta_shape = Shape{batch_size}.append(get_input_shape());
    const Shape scratch_shape{batch_size, output_features};

    return {
        {input_delta_shape, compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
    };
}

void LongShortTermMemory::configure_operators()
{
    lstm_op.set(get_input_features(),
                output_features,
                get_time_steps(),
                lstm_op.activation_function,
                lstm_op.recurrent_activation_function);

    using enum LongShortTermMemoryOp::ForwardSlot;
    using enum LongShortTermMemoryOp::BackwardSlot;

    lstm_op.input_slots = {InputSlot};
    lstm_op.output_slots = {
        OutputSlot,
        ForgetGateSlot,
        InputGateSlot,
        CandidateGateSlot,
        OutputGateSlot,
        CellStateSlot,
        HiddenStateSlot,
        CellActivationSlot
    };
    lstm_op.output_delta_slots = {OutputDeltaSlot};
    lstm_op.input_delta_slots = {InputDeltaSlot};
}

void LongShortTermMemory::set(const Shape& new_input_shape,
                              const Shape& new_output_shape,
                              const string& new_activation_function,
                              const string& new_recurrent_activation_function,
                              const string& new_label)
{
    set_label(new_label);
    set_activation_function(new_activation_function);
    set_recurrent_activation_function(new_recurrent_activation_function);

    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        configure_operators();
        return;
    }

    check_rank(new_input_shape, {2}, "LongShortTermMemory", "input");
    check_rank(new_output_shape, {1}, "LongShortTermMemory", "output");

    input_shape = new_input_shape;
    output_features = new_output_shape[0];

    configure_operators();
}

void LongShortTermMemory::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {2}, "LongShortTermMemory", "input");
    input_shape = new_input_shape;
    configure_operators();
}

void LongShortTermMemory::set_output_shape(const Shape& new_output_shape)
{
    check_rank(new_output_shape, {1}, "LongShortTermMemory", "output");
    output_features = new_output_shape[0];
    configure_operators();
}

void LongShortTermMemory::set_activation_function(const string& new_activation_function)
{
    const ActivationOp::Function function = ActivationOp::from_string(new_activation_function);

    using enum ActivationOp::Function;
    if (function != Identity && function != Sigmoid && function != Tanh && function != ReLU)
        throw runtime_error(format("LongShortTermMemory: unsupported activation function \"{}\".", new_activation_function));

    lstm_op.activation_function = function;
}

void LongShortTermMemory::set_recurrent_activation_function(const string& new_recurrent_activation_function)
{
    const ActivationOp::Function function = ActivationOp::from_string(new_recurrent_activation_function);

    using enum ActivationOp::Function;
    if (function != Identity && function != Sigmoid && function != Tanh && function != ReLU)
        throw runtime_error(format("LongShortTermMemory: unsupported recurrent activation function \"{}\".",
                                   new_recurrent_activation_function));

    lstm_op.recurrent_activation_function = function;
}

void LongShortTermMemory::read_JSON_body(const Json* lstm_layer_element)
{
    set_activation_function(read_json_string(lstm_layer_element, "Activation"));
    set_recurrent_activation_function(read_json_string(lstm_layer_element, "RecurrentActivation"));
    configure_operators();
}

void LongShortTermMemory::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "Activation", ActivationOp::to_string(lstm_op.activation_function));
    add_json_field(printer, "RecurrentActivation", ActivationOp::to_string(lstm_op.recurrent_activation_function));
}

string LongShortTermMemory::write_expression(const vector<string>&,
                                             const vector<string>& output_names) const
{
    return output_names.empty()
        ? "long_short_term_memory_output = LongShortTermMemory(input_sequence);\n"
        : format("{} = LongShortTermMemory(input_sequence);\n", output_names.front());
}

REGISTER(Layer, LongShortTermMemory, "LongShortTermMemory")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
