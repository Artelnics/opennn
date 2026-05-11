//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dense_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "random_utilities.h"

namespace opennn
{

Dense::Dense(const Shape& new_input_shape,
             const Shape& new_output_shape,
             const string& new_activation_function,
             bool new_batch_normalization,
             const string& new_label)
    : Layer("Dense", LayerType::Dense)
{
    operators = {&combination, &batch_norm, &activation, &dropout};

    set(new_input_shape,
        new_output_shape,
        new_activation_function,
        new_batch_normalization,
        new_label);
}

Shape Dense::get_output_shape() const
{
    if (input_shape.empty()) return {output_features};
    Shape output_shape = input_shape;
    output_shape.back() = output_features;
    return output_shape;
}

vector<pair<Shape, Type>> Dense::get_forward_specs(Index batch_size) const
{
    const Shape full   = Shape{batch_size}.append(get_output_shape());
    const Shape stats  = Shape{output_features};
    const Shape unused = {};

    return {
        {batch_norm.active() ? full  : unused, compute_dtype}, // CombinationView (pre-BN)
        {batch_norm.active() ? stats : unused, Type::FP32   }, // BatchNormMean
        {batch_norm.active() ? stats : unused, Type::FP32   }, // BatchNormInverseVariance
        {dropout.active()    ? full  : unused, compute_dtype}, // ActivationView (pre-dropout)
        {full,                                 compute_dtype}, // Output
    };
}

void Dense::configure_operators()
{
    combination.set(get_input_features(), output_features, compute_dtype);

    if (batch_norm.active())
        batch_norm.set(output_features, batch_norm.momentum);

    combination.input_slots  = {Input};
    combination.output_slots = batch_norm.active() ? vector<size_t>{CombinationView}
                                                   : vector<size_t>{Output};

    if (batch_norm.active())
    {
        batch_norm.input_slots  = {CombinationView};
        batch_norm.output_slots = {Output, BatchNormMean, BatchNormInverseVariance};
    }

    activation.input_slots  = {Output};
    activation.output_slots = {Output};

    dropout.input_slots  = {Output};
    dropout.output_slots = {Output};
    dropout.save_slots   = {ActivationView};

    combination.output_delta_slots = {OutputDelta};
    combination.input_delta_slots  = {InputDelta};

    batch_norm.output_delta_slots = {OutputDelta};

    activation.output_delta_slots    = {OutputDelta};
    activation.output_slots_backward = dropout.active()
        ? vector<size_t>{ActivationView}
        : vector<size_t>{};

    dropout.output_delta_slots = {OutputDelta};
}

void Dense::set_batch_normalization(bool enable)
{
    if (enable)
        batch_norm.set(output_features, batch_norm.momentum);
    else
        batch_norm.features = 0;
}

void Dense::set(const Shape& new_input_shape,
                const Shape& new_output_shape,
                const string& new_activation_function,
                bool new_batch_normalization,
                const string& new_label)
{
    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        return;
    }

    check_rank(new_input_shape, {1, 2}, "Dense", "input");
    check_rank(new_output_shape, {1}, "Dense", "output");

    input_shape = new_input_shape;
    output_features = new_output_shape.back();

    set_activation_function(new_activation_function);
    set_batch_normalization(new_batch_normalization);
    set_label(new_label);

    configure_operators();
}

void Dense::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {1, 2}, "Dense", "input");
    input_shape = new_input_shape;
    configure_operators();
}

void Dense::set_output_shape(const Shape& new_output_shape)
{
    output_features = new_output_shape.back();
    configure_operators();
}

void Dense::set_activation_function(const string& name)
{
    Activation::Function function = Activation::from_string(name);

    if (function == Activation::Function::Softmax && get_outputs_number() == 1)
        function = Activation::Function::Sigmoid;

    activation.set_function(function);
}

void Dense::set_momentum(float new_momentum)
{
    if (new_momentum < 0.0f || new_momentum >= 1.0f)
        throw runtime_error("Batch normalization momentum must be in [0,1).");

    batch_norm.momentum = new_momentum;
    if (batch_norm.active())
        batch_norm.set(output_features, batch_norm.momentum);
}

string Dense::write_expression(const vector<string>& input_names,
                               const vector<string>& output_names) const
{
    const vector<TensorView>& parameters = get_parameter_views();
    if (parameters.size() < 2 || !parameters[0].data || !parameters[1].data) return "";

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const float* bias_data = parameters[0].as<float>();
    const float* weight_data = parameters[1].as<float>();

    const string& activation_function_local = Activation::to_string(get_activation_function());

    ostringstream buffer;

    for (Index j = 0; j < outputs_number; ++j)
    {
        buffer << output_names[j] << " = " << activation_function_local << "( " << bias_data[j] << " + ";

        for (Index i = 0; i < inputs_number; ++i)
        {
            const Index weight_index = i * outputs_number + j;
            buffer << "(" << weight_data[weight_index] << "*" << input_names[i] << ")";
            if (i < inputs_number - 1) buffer << " + ";
        }

        buffer << " );\n";
    }

    return buffer.str();
}

void Dense::read_JSON_body(const Json*)
{
}

REGISTER(Layer, Dense, "Dense")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
