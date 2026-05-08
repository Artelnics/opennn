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
{
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

vector<Operator*> Dense::get_operators()
{
    vector<Operator*> operators = {&combination};
    if (batch_norm.active()) operators.push_back(&batch_norm);
    operators.push_back(&activation);
    if (dropout.active()) operators.push_back(&dropout);
    return operators;
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
    combination.input_delta_slots  = is_first_layer ? vector<size_t>{} : vector<size_t>{InputDelta};

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
    is_trainable = true;
    layer_type = LayerType::Dense;
    name = "Dense";

    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        return;
    }

    if (new_input_shape.rank != 1 && new_input_shape.rank != 2)
        throw runtime_error("Dense input shape rank must be 1 or 2 (got "
                            + to_string(new_input_shape.rank) + ").");

    if (new_output_shape.rank != 1)
        throw runtime_error("Dense output shape rank must be 1.");

    input_shape = new_input_shape;
    output_features = new_output_shape.back();

    set_activation_function(new_activation_function);
    set_batch_normalization(new_batch_normalization);
    set_label(new_label);

    configure_operators();
}

void Dense::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 1 && new_input_shape.rank != 2)
        throw runtime_error("Dense input shape rank must be 1 or 2.");

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

REGISTER(Layer, Dense, "Dense")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
