//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O M B I N A T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/combination_operator.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/core/profiler.h"

namespace opennn
{

void CombinationOperator::set(Index new_input_features, Index new_output_features, Type new_compute_dtype)
{
    input_features  = new_input_features;
    output_features = new_output_features;
    compute_dtype   = new_compute_dtype;
}

vector<TensorSpec> CombinationOperator::parameter_specs() const
{
    if (!use_bias)
        return {{{input_features, output_features}, weights_dtype}};

    return {
        {{output_features},                  compute_dtype},
        {{input_features, output_features},  weights_dtype},
    };
}

vector<Operator::SlotQuantization> CombinationOperator::parameter_quantization() const
{
    if (!use_bias)
        return {{output_features, 1}};

    return {{}, {output_features, 1}};
}

void CombinationOperator::link_parameters(span<const TensorView> views)
{
    if (use_bias) link_views(views, {&bias, &weights});
    else          link_views(views, {&weights});
}

void CombinationOperator::link_parameter_scales(span<const TensorView> views)
{
    if (views.empty()) return;
    weight_scale = use_bias && views.size() >= 2 ? views[1] : views[0];
}

void CombinationOperator::link_gradients(span<const TensorView> views)
{
    if (use_bias) link_views(views, {&bias_gradient, &weight_gradient});
    else          link_views(views, {&weight_gradient});
}

void CombinationOperator::set_parameters_random()
{
    if (weights.empty() || tied_transposed) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void CombinationOperator::set_parameters_glorot()
{
    if (weights.empty() || tied_transposed) return;
    const float limit = glorot_limit(input_features, output_features);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}

void CombinationOperator::set_parameters_pytorch()
{

    if (weights.empty() || tied_transposed) return;
    const float limit = 1.0f / sqrt(float(input_features > 0 ? input_features : 1));
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) set_random_uniform(bias.as_vector(), -limit, limit);
}

void CombinationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    PROFILE_SCOPE("op:combination_fwd");
    TensorView& output = get_output(forward_propagation, layer);

    if (tied_transposed)
        return linear_forward_transposed(get_input(forward_propagation, layer), weights, output, weight_scale);

    if (transposed_inference_active)
    {
        const TensorView transposed(weights.get_data(), {output_features, input_features},
                                    weights.get_type(), weights.get_device());
        return linear_forward_transposed(get_input(forward_propagation, layer), transposed, output, weight_scale);
    }

    if (fused_activation == ActivationFunction::GELUTanh
        && output_slots.size() > 1
        && output.is_cuda())
    {
        TensorView& activated = forward_propagation.slots[layer][output_slots[1]];
        return linear_forward(get_input(forward_propagation, layer), weights, bias,
                              activated, CUBLASLT_EPILOGUE_GELU_AUX_BIAS, &output, weight_scale);
    }

    const bool relu = (fused_activation == ActivationFunction::ReLU);

    if (relu && emit_relu_mask && is_training && output.is_cuda()
        && (output.is_fp32() || output.is_bf16()))
    {
        const Index rows = output.size() / output_features;
        try
        {
            relu_mask.ensure<uint8_t>(rows * (output_features / 8));
            if (relu_mask_view.get_data() != relu_mask.data() || relu_mask_view.get_shape().empty()
                || relu_mask_view.get_shape()[0] != rows)
                relu_mask_view = TensorView(relu_mask.data(), Shape{rows, output_features / 8},
                                            Type::INT8, Device::CUDA);
            linear_forward(get_input(forward_propagation, layer), weights, bias, output,
                           CUBLASLT_EPILOGUE_RELU_AUX_BIAS, &relu_mask_view, weight_scale);
            relu_mask_fused_active = true;
            return;
        }
        catch (const runtime_error& error)
        {
            static once_flag reported;
            call_once(reported, [&]{
                cerr << "linear_forward: ReLU-mask epilogue unavailable ("
                     << error.what() << "); ReLU backward runs unfused.\n"; });
            emit_relu_mask = false;
            relu_mask_fused_active = false;
        }
    }

    const cublasLtEpilogue_t epilogue = use_bias
        ? (relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS)
        : (relu ? CUBLASLT_EPILOGUE_RELU      : CUBLASLT_EPILOGUE_DEFAULT);
    linear_forward(get_input(forward_propagation, layer), weights, bias, output, epilogue, nullptr, weight_scale);
}

void CombinationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    PROFILE_SCOPE("op:combination_bwd");
    throw_if(tied_transposed, "CombinationOperator: a tied projection is inference-only.");
    auto& backward_slots = back_propagation.slots[layer];

    const TensorView& input        = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    TensorView empty_input_delta;
    TensorView& input_delta = slot_or(backward_slots, input_delta_slots, 0,
                                      empty_input_delta);

    bool recover_unfused = false;
    if (drelu_source && drelu_source->relu_mask_fused_active
        && input_delta.get_data() && !input_delta.empty())
    {
        try
        {
            return linear_backward(output_delta, input, weights, weight_gradient, bias_gradient,
                                   input_delta, accumulate_input_delta, &drelu_source->relu_mask_view);
        }
        catch (const runtime_error& error)
        {
            static once_flag reported;
            call_once(reported, [&]{
                cerr << "linear_backward: DReLU epilogue unavailable ("
                     << error.what() << "); ReLU backward runs unfused.\n"; });
            drelu_source->relu_mask_fused_active = false;
            drelu_source->emit_relu_mask = false;
            recover_unfused = true;
        }
    }

    // A residual read of this layer's input by another consumer: its delta is
    // summed by the same GEMM (BackPropagation::plan_delta_addends) instead of
    // an accumulate pass afterwards.
    static const TensorView no_addend;
    const TensorView& addend = folds_input_delta_addend && !accumulate_input_delta && !recover_unfused
        ? back_propagation.input_delta_addend(layer, 0)
        : no_addend;

    linear_backward(output_delta, input, weights, weight_gradient, bias_gradient, input_delta,
                    accumulate_input_delta, nullptr, addend.empty() ? nullptr : &addend);

    if (recover_unfused)
        activation_backward(input, input_delta, ActivationFunction::ReLU);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
