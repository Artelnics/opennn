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
    Shape result = input_shape;
    if (result.empty()) return {output_features};
    result.back() = output_features;
    return result;
}

vector<Operator*> Dense::get_operators()
{
    vector<Operator*> ops = {&combination};
    if (batch_normalization) ops.push_back(&batch_norm);
    return ops;
}

vector<pair<Shape, Type>> Dense::get_forward_specs(Index batch_size) const
{
    const Shape output_shape = Shape{batch_size}.append(get_output_shape());
    const Type act = activation_dtype;

    const Shape activation_shape   = dropout.active()        ? output_shape           : Shape{};
    const Shape combination_shape  = batch_normalization     ? output_shape           : Shape{};
    const Shape bn_stat_shape      = batch_normalization     ? Shape{output_features} : Shape{};

    return {
        {combination_shape, act},        // Combination
        {bn_stat_shape,     Type::FP32}, // BatchNormMean
        {bn_stat_shape,     Type::FP32}, // BatchNormInverseVariance
        {activation_shape,  act},        // Activation
        {output_shape,      act},        // Output
    };
}

vector<pair<Shape, Type>> Dense::get_backward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(get_input_shape()), activation_dtype}};
}

void Dense::configure_operators()
{
    combination.set(get_input_features(), output_features, activation_dtype);
    if (batch_normalization)
        batch_norm.set(output_features, momentum);
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

    momentum = new_momentum;
    if (batch_normalization)
        batch_norm.set(output_features, momentum);
}

void Dense::set_parameters_glorot()
{
    const float limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));
    set_random_uniform(parameters[Weight].as_vector(), -limit, limit);
    parameters[Bias].fill(0.0f);
    if (batch_normalization) batch_norm.init_defaults();
}

void Dense::set_parameters_random()
{
    set_random_uniform(parameters[Weight].as_vector());
    parameters[Bias].fill(0.0f);
    if (batch_normalization) batch_norm.init_defaults();
}

// link_parameters() and link_states() are inherited from Layer; the base
// auto-distributes slices to each operator returned by get_operators().

void Dense::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& input = forward_views[Input][0];
    TensorView& output      = forward_views[Output][0];

    if (batch_normalization)
    {
        TensorView& combination_out = forward_views[CombinationView][0];
        combination.apply(input, combination_out);

        if (is_training)
            batch_norm.apply_training(combination_out,
                                      forward_views[BatchNormMean][0],
                                      forward_views[BatchNormInverseVariance][0],
                                      output);
        else
            batch_norm.apply_inference(combination_out, output);

        activation.apply(output);
    }
    else
    {
        const bool fused_relu = (activation.function == Activation::Function::ReLU);
        const cublasLtEpilogue_t epilogue = fused_relu
                                          ? CUBLASLT_EPILOGUE_RELU_BIAS
                                          : CUBLASLT_EPILOGUE_BIAS;
        combination.apply(input, output, epilogue);
        if (!fused_relu) activation.apply(output);
    }

    if (is_training && dropout.active())
    {
        copy(output, forward_views[ActivationView][0]);
        dropout.apply(output);
    }
}

void Dense::back_propagate(ForwardPropagation& forward_propagation,
                           BackPropagation& back_propagation,
                           size_t layer) const noexcept
{
    auto& forward_views   = forward_propagation.views[layer];
    auto& delta_views     = back_propagation.delta_views[layer];
    auto& gradient_views  = back_propagation.gradient_views[layer];

    const TensorView& input  = forward_views[Input][0];
    const TensorView& output = forward_views[Output][0];

    TensorView& output_delta = delta_views[OutputDelta][0];

    if (dropout.active())
        dropout.apply_delta(output_delta);

    const TensorView& act_outputs = dropout.active()
                                  ? forward_views[ActivationView][0]
                                  : output;
    activation.apply_delta(act_outputs, output_delta);

    if (batch_normalization)
        batch_norm.apply_delta(forward_views[CombinationView][0],
                               forward_views[BatchNormMean][0],
                               forward_views[BatchNormInverseVariance][0],
                               gradient_views[Gamma],
                               gradient_views[Beta],
                               output_delta);

    const Index total_rows = input.size() / input.shape.back();

    TensorView output_delta_2d = output_delta.reshape({total_rows, output_delta.shape.back()});
    TensorView input_2d        = input.reshape({total_rows, input.shape.back()});

    TensorView input_delta_2d;
    if (!is_first_layer)
    {
        TensorView& input_delta = delta_views[InputDelta][0];
        input_delta_2d = input_delta.reshape({total_rows, input_delta.shape.back()});
    }

    combination.apply_delta(output_delta_2d,
                            input_2d,
                            input_delta_2d,
                            gradient_views[Weight],
                            gradient_views[Bias],
                            false);
}

void Dense::from_JSON(const JsonDocument& document)
{
    const Json* dense_layer_element = document.first_child("Dense");
    if (!dense_layer_element) throw runtime_error(name + " element is nullptr.");

    set_label(read_json_string(dense_layer_element, "Label"));

    set_input_shape(string_to_shape(read_json_string(dense_layer_element, "InputDimensions")));
    set_output_shape({ read_json_index(dense_layer_element, "NeuronsNumber") });

    set_batch_normalization(read_json_bool(dense_layer_element, "BatchNormalization"));

    activation.from_JSON(dense_layer_element);
    dropout.from_JSON(dense_layer_element);
    if (batch_normalization)
    {
        batch_norm.from_JSON(dense_layer_element);
        momentum = batch_norm.momentum;
    }
}

void Dense::load_state_from_JSON(const JsonDocument& document)
{
    if (!batch_normalization) return;

    const Json* dense_layer_element = document.first_child("Dense");
    if (!dense_layer_element) throw runtime_error(name + " element is nullptr.");

    VectorR tmp;
    string_to_vector(read_json_string(dense_layer_element, "RunningMeans"), tmp);
    if (tmp.size() == states[RunningMean].size() && states[RunningMean].data)
        states[RunningMean].as_vector() = tmp;

    string_to_vector(read_json_string(dense_layer_element, "RunningVariances"), tmp);
    if (tmp.size() == states[RunningVariance].size() && states[RunningVariance].data)
        states[RunningVariance].as_vector() = tmp;
}

void Dense::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dense");

    write_json(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(input_shape)},
        {"NeuronsNumber", to_string(output_features)},
        {"BatchNormalization", to_string(batch_normalization)}
    });

    activation.to_JSON(printer);
    dropout.to_JSON(printer);
    if (batch_normalization) batch_norm.to_JSON(printer);

    if (batch_normalization)
        write_json(printer, {
            {"RunningMeans", vector_to_string(states[RunningMean].as_vector())},
            {"RunningVariances", vector_to_string(states[RunningVariance].as_vector())}
        });

    printer.close_element();
}

REGISTER(Layer, Dense, "Dense")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
