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

    if (gated)
        return {
            {full,    compute_dtype},   // CombinationView: gate projection (pre-SiLU)
            {Shape{}, Type::FP32   },
            {Shape{}, Type::FP32   },
            {full,    compute_dtype},   // ActivationView: up projection
            {full,    compute_dtype},
        };

    const bool keep_pre_activation = batch_norm.active() || activation_needs_input(activation_operator.activation_function);

    return {
        {keep_pre_activation ? full  : Shape{}, compute_dtype},
        {batch_norm.active() ? stats : Shape{}, Type::FP32   },
        {batch_norm.active() ? stats : Shape{}, Type::FP32   },
        {saves_pre_dropout_activation() ? full : Shape{}, compute_dtype},
        {full,                                  compute_dtype},
    };
}

bool Dense::saves_pre_dropout_activation() const
{
    // ReLU's backward can gate on the post-dropout output: kept units keep their
    // sign and dropped units already carry a zero delta. The fused GEMM epilogue
    // never writes ActivationView, so saving there would hand the backward zeros.
    return dropout.active()
        && !activation_needs_input(activation_operator.activation_function)
        && activation_operator.activation_function != ActivationFunction::ReLU;
}

vector<TensorSpec> Dense::get_backward_specs(Index batch_size) const
{
    if (!is_trainable) return {};

    vector<TensorSpec> specs = {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};

    if (gated)
    {
        // Gate and up projection deltas.
        const Shape full = Shape{batch_size}.append(get_output_shape());
        specs.push_back({full, compute_dtype});
        specs.push_back({full, compute_dtype});
        return specs;
    }

    if (activation_needs_input(activation_operator.activation_function))
        specs.push_back({Shape{batch_size}.append(get_output_shape()), compute_dtype});

    return specs;
}

void Dense::configure_operators()
{
    if (gated)
    {
        throw_if(batch_norm.active(),
                 "Dense: gated (SwiGLU) mode cannot be combined with batch normalization.");
        throw_if(activation_operator.activation_function != ActivationFunction::Identity,
                 "Dense: gated (SwiGLU) mode has a fixed SiLU gate; set the activation to Identity.");

        operators = {&combination, &up_combination, &swiglu, &dropout};

        combination.set(get_input_features(), output_features, compute_dtype);
        up_combination.set(get_input_features(), output_features, compute_dtype);
        up_combination.use_bias = combination.use_bias;

        combination.fused_activation = ActivationFunction::Identity;

        combination.input_slots     = {Input};
        combination.output_slots    = {CombinationView};
        up_combination.input_slots  = {Input};
        up_combination.output_slots = {ActivationView};

        swiglu.input_slots  = {CombinationView, ActivationView};
        swiglu.output_slots = {Output};

        // Backward (reverse order: swiglu, up, gate): both projections push
        // their input delta into the same buffer, the second accumulating.
        swiglu.output_delta_slots = {0};
        swiglu.input_delta_slots  = {2, 3};

        up_combination.output_delta_slots     = {3};
        up_combination.input_delta_slots      = {1};
        up_combination.accumulate_input_delta = false;

        combination.output_delta_slots     = {2};
        combination.input_delta_slots      = {1};
        combination.accumulate_input_delta = true;

        dropout.input_slots  = {Output};
        dropout.output_slots = {Output};

        activation_operator.forward_fused = false;
        activation_operator.save_slot = SIZE_MAX;
        return;
    }

    operators = {&combination, &batch_norm, &activation_operator, &dropout};
    combination.accumulate_input_delta = false;

    const bool input_deriv = activation_needs_input(activation_operator.activation_function);

    throw_if(input_deriv && batch_norm.active(),
             "Dense: input-derivative activations (e.g. GELU) cannot be fused with "
             "batch normalization. Use a standalone Activation layer after the Dense.");

    combination.set(get_input_features(), output_features, compute_dtype);

    if (batch_norm.active())
        batch_norm.set(output_features, batch_norm.momentum);

    const bool keep_pre_activation = batch_norm.active() || input_deriv;
    combination.output_slots = keep_pre_activation ? vector<size_t>{CombinationView}
                                                   : vector<size_t>{Output};

    if (batch_norm.active())
    {
        batch_norm.input_slots  = {CombinationView};
        batch_norm.output_slots = {Output, BatchNormMean, BatchNormInverseVariance};
    }

    combination.input_delta_slots  = {1};
    combination.output_delta_slots = {0};
    activation_operator.input_delta_slots   = {1};
    activation_operator.output_delta_slots  = {0};

    if (input_deriv)
    {
        activation_operator.input_slots        = {CombinationView};
        activation_operator.output_slots       = {Output};
        activation_operator.output_delta_slots = {0};
        activation_operator.input_delta_slots  = {2};
        combination.output_delta_slots         = {2};
        combination.input_delta_slots          = {1};
    }
    else
    {
        activation_operator.input_slots  = {Output};
        activation_operator.output_slots = {Output};
    }

    const bool fuse_relu = (activation_operator.activation_function == ActivationFunction::ReLU)
                           && !batch_norm.active();

    // The cuBLASLt GELU epilogue implements the tanh approximation, so only
    // GELUTanh can be folded without changing the math; its AUX buffer
    // requires a leading dimension divisible by 8.
    const bool fuse_gelu_tanh = (activation_operator.activation_function == ActivationFunction::GELUTanh)
                                && !batch_norm.active()
                                && output_features % 8 == 0;

    combination.fused_activation = fuse_relu      ? ActivationFunction::ReLU
                                 : fuse_gelu_tanh ? ActivationFunction::GELUTanh
                                                  : ActivationFunction::Identity;

    if (fuse_gelu_tanh)
        combination.output_slots = {CombinationView, Output};

    activation_operator.forward_fused = fuse_relu || fuse_gelu_tanh;

    dropout.input_slots  = {Output};
    dropout.output_slots = {Output};

    activation_operator.save_slot = saves_pre_dropout_activation() ? ActivationView : SIZE_MAX;
}

void Dense::set_batch_normalization(bool enable)
{
    batch_norm.features = enable ? output_features : 0;
    configure_operators();
}

void Dense::set_gated(bool new_gated)
{
    gated = new_gated;
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

    throw_if(gated,
             "Dense::write_expression: gated (SwiGLU) mode is not supported in the exported expression.");

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

    if (dense_layer_element->has("UseBias"))
        set_use_bias(read_json_bool(dense_layer_element, "UseBias"));

    if (dense_layer_element->has("Activation"))
        set_activation_function(read_json_string(dense_layer_element, "Activation"));

    if (dense_layer_element->has("Gated"))
        set_gated(read_json_bool(dense_layer_element, "Gated"));
}

void Dense::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"BatchNormalization", to_string(batch_norm.active())},
        {"UseBias", to_string(combination.use_bias)},
        {"Activation", ActivationOperator::to_string(get_activation_function())},
        {"Gated", to_string(gated)}
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
