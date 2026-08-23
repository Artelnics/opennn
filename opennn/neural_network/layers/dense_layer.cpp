//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/core/string_utilities.h"
#include "opennn/registry.h"

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
    output_shape.set_dimension(output_shape.get_rank() - 1, output_features);
    return output_shape;
}

vector<TensorSpec> Dense::get_forward_specs(Index batch_size) const
{
    const Shape full   = Shape{batch_size}.append(get_output_shape());
    const Shape stats  = Shape{output_features};
    const Shape dropout_mask = dropout.active() ? full : Shape{};
    const Index rows = output_features > 0 ? full.size() / output_features : 0;
    const Shape drelu_mask = combination.emit_relu_mask
        ? Shape{rows, output_features / 8}
        : Shape{};

    if (gated)
        return {
            {full,    compute_dtype},
            {Shape{}, Type::FP32   },
            {Shape{}, Type::FP32   },
            {full,    compute_dtype},
            {Shape{}, Type::INT8   },
            {dropout_mask, Type::INT8},
            {full,    compute_dtype},
        };

    const bool keep_pre_activation = batch_norm.active() || activation_needs_input(activation_operator.activation_function);

    return {
        {keep_pre_activation ? full  : Shape{}, compute_dtype},
        {batch_norm.active() ? stats : Shape{}, Type::FP32   },
        {batch_norm.active() ? stats : Shape{}, Type::FP32   },
        {saves_pre_dropout_activation() ? full : Shape{}, compute_dtype},
        {drelu_mask,                             Type::INT8   },
        {dropout_mask,                           Type::INT8   },
        {full,                                  compute_dtype},
    };
}

bool Dense::saves_pre_dropout_activation() const
{

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

    reset_drelu_fusion();
    combination.folds_input_delta_addend = false;

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
        dropout.mask_slot = DropoutMask;
        combination.relu_mask_slot = DreluMask;

        activation_operator.forward_fused = false;
        activation_operator.save_slot = SIZE_MAX;
        return;
    }

    operators = {&combination, &batch_norm, &activation_operator, &dropout};
    combination.accumulate_input_delta = false;
    combination.folds_input_delta_addend = true;

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
        activation_operator.input_slots       = {CombinationView};
        activation_operator.output_slots      = {Output};
        activation_operator.input_delta_slots = {2};
        combination.output_delta_slots        = {2};
    }
    else
    {
        activation_operator.input_slots  = {Output};
        activation_operator.output_slots = {Output};
    }

    const bool fuse_relu = (activation_operator.activation_function == ActivationFunction::ReLU)
                           && !batch_norm.active();

    // CUDA-only: the fusion is the cuBLASLt GELU_AUX_BIAS epilogue, which writes
    // the activated result to a second slot. The CPU combination has no such
    // epilogue and would leave that slot untouched while the activation operator
    // skipped its pass, so the layer would output zeros. configure_operators runs
    // again from on_compute_dtype_changed, which compile() calls after
    // set_compute_device, so the flag is correct by the time the arena is planned.
    const bool fuse_gelu_tanh = (activation_operator.activation_function == ActivationFunction::GELUTanh)
                                && !batch_norm.active()
                                && output_features % 8 == 0
                                && get_compute_device() == Device::CUDA;

    // A single-output layer - a classifier head, a regression output - runs its
    // combination as a row-wise reduction on CUDA, and that kernel can carry
    // any elementwise activation in the register it has just accumulated. Its
    // own activation pass would be a launch to read and write one number per
    // row: about a microsecond, which is five per cent of what a batch of 256
    // costs. The activations excluded are the ones whose backward reads the
    // pre-activation value, which this does not keep.
    // Dropout is excluded because the pre-dropout activation is saved by the
    // activation pass this fusion removes; ReLU and GELU-tanh reach the same
    // exclusion through saves_pre_dropout_activation's own two clauses.
    // OPENNN_SINGLE_OUTPUT_ACTIVATION=0 keeps the separate pass for the A/B.
    static const bool fuse_single_output_enabled =
        env_flag_enabled("OPENNN_SINGLE_OUTPUT_ACTIVATION", true);

    const bool fuse_single_output = fuse_single_output_enabled
                                    && output_features == 1
                                    && !batch_norm.active()
                                    && !input_deriv
                                    && !saves_pre_dropout_activation()
                                    && activation_operator.activation_function != ActivationFunction::Softmax
                                    && activation_operator.activation_function != ActivationFunction::Identity;

    combination.fused_activation = fuse_relu          ? ActivationFunction::ReLU
                                 : fuse_gelu_tanh     ? ActivationFunction::GELUTanh
                                 : fuse_single_output ? activation_operator.activation_function
                                                      : ActivationFunction::Identity;

    if (fuse_gelu_tanh)
        combination.output_slots = {CombinationView, Output};

    activation_operator.forward_fused = fuse_relu || fuse_gelu_tanh || fuse_single_output;

    dropout.input_slots  = {Output};
    dropout.output_slots = {Output};
    dropout.mask_slot = DropoutMask;
    combination.relu_mask_slot = DreluMask;

    activation_operator.save_slot = saves_pre_dropout_activation() ? ActivationView : SIZE_MAX;
}

void Dense::set_batch_normalization(bool enable)
{
    batch_norm.features = enable ? output_features : 0;
    configure_operators();
}

bool Dense::try_wire_drelu_fusion(Dense& producer, Index producer_layer)
{
    const bool producer_eligible = !producer.gated
        && !producer.batch_norm.active() && !producer.dropout.active()
        && producer.combination.use_bias && !producer.tied_source && producer.is_trainable
        && producer.activation_operator.activation_function == ActivationFunction::ReLU
        && producer.output_features > 0 && producer.output_features % 128 == 0;

    const bool consumer_eligible = !gated && !tied_source && is_trainable
        && !combination.accumulate_input_delta;

    if (!producer_eligible || !consumer_eligible) return false;

    producer.combination.emit_relu_mask = true;
    producer.activation_operator.backward_fused_by_consumer = true;
    combination.drelu_source = &producer.combination;
    combination.drelu_source_layer = producer_layer;
    drelu_producer = &producer;
    return true;
}

bool Dense::try_wire_single_output_relu_fusion(Dense& producer, Index producer_layer)
{
    const bool producer_eligible = !producer.gated
        && !producer.batch_norm.active() && !producer.dropout.active()
        && !producer.tied_source && producer.is_trainable
        && producer.activation_operator.activation_function == ActivationFunction::ReLU;

    const bool consumer_eligible = !gated && !tied_source && is_trainable
        && output_features == 1 && !combination.accumulate_input_delta
        && !combination.drelu_source;

    if (!producer_eligible || !consumer_eligible) return false;

    combination.fuse_input_relu = true;
    combination.input_relu_source_layer = producer_layer;
    producer.activation_operator.backward_fused_by_consumer = true;
    return true;
}

void Dense::reset_single_output_relu_fusion()
{
    combination.fuse_input_relu = false;
    combination.input_relu_source_layer = -1;
    activation_operator.backward_fused_by_consumer = false;
}

void Dense::reset_drelu_fusion()
{
    // The producer half is cleared too. Reconfiguring only the consumer used to
    // leave the producer still emitting its ReLU mask and still believing a
    // consumer would apply the ReLU backward for it - so that derivative was
    // simply dropped and the producer's gradient came out wrong.
    if (drelu_producer)
    {
        drelu_producer->combination.emit_relu_mask = false;
        drelu_producer->activation_operator.backward_fused_by_consumer = false;
        drelu_producer = nullptr;
    }

    combination.emit_relu_mask = false;
    combination.relu_mask_fusion_disabled = false;
    combination.drelu_source = nullptr;
    combination.drelu_source_layer = -1;
    activation_operator.backward_fused_by_consumer = false;
}

void Dense::set_gated(bool new_gated)
{
    gated = new_gated;
    configure_operators();
}

void Dense::set_tied_weight_source(const Layer* source)
{
    throw_if(source && (combination.use_bias || gated || batch_norm.active() || dropout.active()
                        || activation_operator.activation_function != ActivationFunction::Identity),
             "Dense::set_tied_weight_source: only a bias-free, non-gated, identity Dense can tie its weight.");
    tied_source = source;
    combination.tied_transposed = source != nullptr;
}

void Dense::set_tied_weight(const TiedWeight& tied_weight)
{
    throw_if(tied_weight.spec_index != 0 || tied_weight.source_spec_index != 0,
             "Dense::set_tied_weight: Dense only supports tying weight specification 0 to source specification 0.");
    set_tied_weight_source(tied_weight.source);
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

void Dense::apply_input_shape(const Shape& new_input_shape)
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

    throw_if(parameter_views.size() < 2 || !parameter_views[0].get_data() || !parameter_views[1].get_data(),
             "Dense::write_expression: layer not configured.");

    throw_if(batch_norm.active(),
             "Dense::write_expression: batch normalization is not supported in the exported expression.");

    throw_if(gated,
             "Dense::write_expression: gated (SwiGLU) mode is not supported in the exported expression.");

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const float* const bias_data = parameter_views[0].as<float>();
    const float* const weight_data = parameter_views[1].as<float>();

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

    set_use_bias(read_json_bool(dense_layer_element, "UseBias", get_use_bias()));

    if (dense_layer_element->has("Activation"))
        set_activation_function(read_json_string(dense_layer_element, "Activation"));

    set_gated(read_json_bool(dense_layer_element, "Gated", gated));

    set_transposed_inference(read_json_bool(dense_layer_element, "TransposedInference", get_transposed_inference()));
}

void Dense::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"BatchNormalization", batch_norm.active()},
        {"UseBias", combination.use_bias},
        {"Activation", ActivationOperator::to_string(get_activation_function())},
        {"Gated", gated},
        {"TransposedInference", combination.transposed_inference_preferred}
    });
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
