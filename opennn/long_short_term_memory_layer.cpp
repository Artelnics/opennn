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
    const Shape sequence_shape{batch_size, T, output_features};
    const Shape output_shape = return_sequences
        ? sequence_shape
        : Shape{batch_size, output_features};

    return {
        {sequence_shape,  compute_dtype}, // ForgetGateSlot
        {sequence_shape,  compute_dtype}, // InputGateSlot
        {sequence_shape,  compute_dtype}, // CandidateGateSlot
        {sequence_shape,  compute_dtype}, // OutputGateSlot
        {sequence_shape,  compute_dtype}, // CellStateSlot
        {sequence_shape,  compute_dtype}, // HiddenStateSlot
        {sequence_shape,  compute_dtype}, // CellActivationSlot
        {output_shape,    compute_dtype}, // OutputSlot (principal output, last)
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

    lstm_op.return_sequences = return_sequences;

    using enum LongShortTermMemoryOp::ForwardSlot;
    using enum LongShortTermMemoryOp::BackwardSlot;

    lstm_op.input_slots = {InputSlot};
    lstm_op.output_slots = {
        ForgetGateSlot,
        InputGateSlot,
        CandidateGateSlot,
        OutputGateSlot,
        CellStateSlot,
        HiddenStateSlot,
        CellActivationSlot,
        OutputSlot
    };
    lstm_op.output_delta_slots = {OutputDeltaSlot};
    lstm_op.input_delta_slots = {InputDeltaSlot};
}

void LongShortTermMemory::set_return_sequences(bool value)
{
    if (return_sequences == value) return;
    return_sequences = value;
    configure_operators();
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

string LongShortTermMemory::write_expression(const vector<string>& feature_names,
                                             const vector<string>& output_names) const
{
    if (parameters.size() < 12) return {};
    for (Index p = 0; p < 12; ++p)
        if (!parameters[p].data) return {};

    VectorMap bf = parameters[0].as_vector();
    VectorMap bi = parameters[1].as_vector();
    VectorMap bg = parameters[2].as_vector();
    VectorMap bo = parameters[3].as_vector();

    MatrixMap Wf = parameters[4].as_matrix();
    MatrixMap Wi = parameters[5].as_matrix();
    MatrixMap Wg = parameters[6].as_matrix();
    MatrixMap Wo = parameters[7].as_matrix();

    MatrixMap Uf = parameters[8].as_matrix();
    MatrixMap Ui = parameters[9].as_matrix();
    MatrixMap Ug = parameters[10].as_matrix();
    MatrixMap Uo = parameters[11].as_matrix();

    const string act = ActivationOp::to_string(lstm_op.activation_function);
    const string ract = ActivationOp::to_string(lstm_op.recurrent_activation_function);

    const Index T = get_time_steps();
    const Index F = get_input_features();
    const Index H = output_features;

    auto h_name = [&](Index t, Index j) -> string {
        const string internal = format("lstm_h_{}_{}", t, j);
        if (return_sequences)
        {
            const Index linear = t * H + j;
            if (linear < ssize(output_names)) return output_names[linear];
            return internal;
        }
        if (t == T - 1 && j < ssize(output_names)) return output_names[j];
        return internal;
    };

    ostringstream buf;
    buf.precision(10);

    auto gate_expr = [&](const string& name,
                         const string& activation,
                         const VectorMap& b,
                         const MatrixMap& W,
                         const MatrixMap& U,
                         Index t, Index j)
    {
        buf << name << " = " << activation << "( " << b(j);
        for (Index k = 0; k < F; ++k)
        {
            const Index feature_index = t * F + k;
            if (feature_index < ssize(feature_names))
                buf << " + (" << feature_names[feature_index] << "*" << W(k, j) << ")";
        }
        if (t > 0)
            for (Index p = 0; p < H; ++p)
                buf << " + (" << h_name(t - 1, p) << "*" << U(p, j) << ")";
        buf << " );\n";
    };

    for (Index t = 0; t < T; ++t)
    {
        for (Index j = 0; j < H; ++j)
        {
            const string f_var = format("lstm_f_{}_{}", t, j);
            const string i_var = format("lstm_i_{}_{}", t, j);
            const string g_var = format("lstm_g_{}_{}", t, j);
            const string o_var = format("lstm_o_{}_{}", t, j);
            const string c_var = format("lstm_c_{}_{}", t, j);

            gate_expr(f_var, ract, bf, Wf, Uf, t, j);
            gate_expr(i_var, ract, bi, Wi, Ui, t, j);
            gate_expr(g_var, act,  bg, Wg, Ug, t, j);
            gate_expr(o_var, ract, bo, Wo, Uo, t, j);

            if (t > 0)
                buf << c_var << " = (" << f_var << " * lstm_c_" << (t - 1) << "_" << j
                    << ") + (" << i_var << " * " << g_var << ");\n";
            else
                buf << c_var << " = " << i_var << " * " << g_var << ";\n";

            buf << h_name(t, j) << " = " << o_var << " * " << act << "( " << c_var << " );\n";
        }
    }

    return buf.str();
}

REGISTER(Layer, LongShortTermMemory, "LongShortTermMemory")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
