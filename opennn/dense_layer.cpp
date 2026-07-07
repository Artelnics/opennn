//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dense_layer.h"

namespace opennn
{

Dense::Dense(const Shape& new_input_shape,
             const Shape& new_output_shape,
             const string& new_activation_function,
             bool new_batch_normalization,
             const string& new_label)
    : Layer(LayerType::Dense)
{
    operators = {&combination, &batch_norm, &activation_operator, &dropout};

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

vector<TensorSpec> Dense::get_forward_specs(Index batch_size) const
{
    const Shape full   = Shape{batch_size}.append(get_output_shape());
    const Shape stats  = Shape{output_features};

    return {
        {batch_norm.active() ? full  : Shape{}, compute_dtype},
        {batch_norm.active() ? stats : Shape{}, Type::FP32   },
        {batch_norm.active() ? stats : Shape{}, Type::FP32   },
        {dropout.active()    ? full  : Shape{}, compute_dtype},
        {full,                                  compute_dtype},
    };
}

void Dense::configure_operators()
{
    combination.set(get_input_features(), output_features, compute_dtype);

    if (batch_norm.active())
        batch_norm.set(output_features, batch_norm.momentum);

    combination.output_slots = batch_norm.active() ? vector<size_t>{CombinationView}
                                                   : vector<size_t>{Output};

    if (batch_norm.active())
    {
        batch_norm.input_slots  = {CombinationView};
        batch_norm.output_slots = {Output, BatchNormMean, BatchNormInverseVariance};
    }

    activation_operator.input_slots  = {Output};
    activation_operator.output_slots = {Output};

    const bool fuse_relu = (activation_operator.activation_function == ActivationFunction::ReLU)
                           && !batch_norm.active();
    combination.fuse_relu    = fuse_relu;
    activation_operator.forward_fused = fuse_relu;

    dropout.input_slots  = {Output};
    dropout.output_slots = {Output};

    activation_operator.save_slot = dropout.active() ? ActivationView : SIZE_MAX;
}

void Dense::set_batch_normalization(bool enable)
{
    batch_norm.features = enable ? output_features : 0;
    configure_operators();
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

    ActivationFunction function = ActivationOperator::from_string(new_activation_function);
    if (function == ActivationFunction::Softmax && get_outputs_number() == 1)
        function = ActivationFunction::Sigmoid;
    activation_operator.set_activation_function(function);

    batch_norm.features = new_batch_normalization ? output_features : 0;

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
    ActivationFunction function = ActivationOperator::from_string(name);

    if (function == ActivationFunction::Softmax && get_outputs_number() == 1)
        function = ActivationFunction::Sigmoid;

    activation_operator.set_activation_function(function);
    configure_operators();
}

void Dense::set_momentum(float new_momentum)
{
    throw_if(new_momentum < 0.0f || new_momentum >= 1.0f,
             "Batch normalization momentum must be in [0,1).");

    batch_norm.momentum = new_momentum;
    if (batch_norm.active())
        batch_norm.set(output_features, batch_norm.momentum);
}

string Dense::write_expression(const vector<string>& input_names,
                               const vector<string>& output_names) const
{
    const vector<TensorView>& parameter_views = get_parameter_views();

    throw_if(parameter_views.size() < 2 || !parameter_views[0].data || !parameter_views[1].data,
             "Dense::write_expression: layer not configured.");

    throw_if(batch_norm.active(),
             "Dense::write_expression: batch normalization is not supported in the exported expression.");

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const float* bias_data = parameter_views[0].as<float>();
    const float* weight_data = parameter_views[1].as<float>();

    const string& activation_function_local = ActivationOperator::to_string(get_activation_function());

    ostringstream buffer;
    buffer.precision(10);

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

void Dense::read_JSON_body(const Json* dense_layer_element)
{
    batch_norm.features = read_json_bool(dense_layer_element, "BatchNormalization") ? output_features : 0;
}

void Dense::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"BatchNormalization", to_string(batch_norm.active())}
    });
}

void Dense::from_JSON(const JsonDocument& document)
{
    Layer::from_JSON(document);

    configure_operators();
}

REGISTER(Layer, Dense, "Dense")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
