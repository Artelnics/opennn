//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O M B I N A T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "combination_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

void CombinationOp::set(Index new_input_features, Index new_output_features, Type new_weight_type)
{
    input_features  = new_input_features;
    output_features = new_output_features;
    weight_type     = new_weight_type;
}

vector<TensorSpec> CombinationOp::parameter_specs() const
{
    return {
        {{output_features},                  weight_type},
        {{input_features, output_features},  weight_type},
    };
}

void CombinationOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void CombinationOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias_gradient   = views[0];
    weight_gradient = views[1];
}

void CombinationOp::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void CombinationOp::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = glorot_limit(input_features, output_features);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}

void CombinationOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    PROFILE_SCOPE("op:combination_fwd");
    apply(get_input(fp, layer), get_output(fp, layer),
          fuse_relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS);
}

void CombinationOp::apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    linear_forward(input, weights, bias, output, epilogue);
}

void CombinationOp::apply_delta(const TensorView& output_delta,
                              const TensorView& input,
                              TensorView& input_delta,
                              bool accumulate_input_delta) const
{
    linear_backward(output_delta, input, weights, weight_gradient, bias_gradient,
                    input_delta, accumulate_input_delta);
}

void CombinationOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    PROFILE_SCOPE("op:combination_bwd");
    auto& backward_slots = bp.backward_slots[layer];

    const TensorView& input        = get_input(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(backward_slots, input_delta_slots, 0, empty);

    apply_delta(output_delta, input, input_delta, false);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
