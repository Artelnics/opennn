//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O M B I N A T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "combination_operator.h"
#include "device_backend.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

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
    return {
        {{output_features},                  compute_dtype},
        {{input_features, output_features},  compute_dtype},
    };
}

void CombinationOperator::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void CombinationOperator::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias_gradient   = views[0];
    weight_gradient = views[1];
}

void CombinationOperator::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void CombinationOperator::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = glorot_limit(input_features, output_features);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}

void CombinationOperator::set_parameters_pytorch()
{
    // nn.Linear default: weight and bias ~ U(+-1/sqrt(fan_in)).
    if (weights.empty()) return;
    const float limit = 1.0f / sqrt(float(input_features > 0 ? input_features : 1));
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) set_random_uniform(bias.as_vector(), -limit, limit);
}

void CombinationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    PROFILE_SCOPE("op:combination_fwd");

    TensorView& output = get_output(forward_propagation, layer);

    if (fused_activation == ActivationFunction::GELUTanh
        && output_slots.size() > 1
        && output.is_cuda())
    {
        TensorView& activated = forward_propagation.forward_slots[layer][output_slots[1]];
        linear_forward(get_input(forward_propagation, layer), weights, bias,
                       activated, CUBLASLT_EPILOGUE_GELU_AUX_BIAS, &output);
        return;
    }

    linear_forward(get_input(forward_propagation, layer), weights, bias, output,
                   fused_activation == ActivationFunction::ReLU ? CUBLASLT_EPILOGUE_RELU_BIAS
                                                                : CUBLASLT_EPILOGUE_BIAS);
}


void CombinationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    PROFILE_SCOPE("op:combination_bwd");
    auto& backward_slots = back_propagation.backward_slots[layer];

    const TensorView& input        = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    TensorView& input_delta = slot_or(backward_slots, input_delta_slots, 0);

    linear_backward(output_delta, input, weights, weight_gradient, bias_gradient, input_delta, false);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
