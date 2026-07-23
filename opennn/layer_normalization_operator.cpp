//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer_normalization_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void LayerNormalizationOperator::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> LayerNormalizationOperator::parameter_specs() const
{
    return vector<TensorSpec>(2, {Shape{embedding_dimension}, Type::FP32});
}

void LayerNormalizationOperator::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
}

void LayerNormalizationOperator::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
}

void LayerNormalizationOperator::init_defaults()
{
    if (gamma.data) gamma.as_vector().setOnes();
    if (beta.data)  beta.as_vector().setZero();
}

void LayerNormalizationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& means       = get_output(forward_propagation, layer);
    TensorView& stds        = get_output(forward_propagation, layer, 1);
    TensorView& normalized  = get_output(forward_propagation, layer, 2);
    TensorView& output      = get_output(forward_propagation, layer, 3);

    if (fuse_add)
    {
        const TensorView& residual = forward_propagation.input_views[layer][1];
        layer_normalization_add_forward(input, residual, gamma, beta, means, stds, normalized, normalized, output);
        return;
    }

    layer_normalization_forward(input, gamma, beta, means, stds, normalized, output);
}

void LayerNormalizationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& stds         = get_output(forward_propagation, layer, 1);
    const TensorView& normalized   = get_output(forward_propagation, layer, 2);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);

    const TensorView& norm_input = fuse_add ? normalized : get_input(forward_propagation, layer);

    layer_normalization_backward(norm_input, output_delta, get_output(forward_propagation, layer),
                        stds, normalized, gamma, gamma_gradient, beta_gradient,
                        input_delta);

    if (fuse_add && residual_delta_slot)
    {
        TensorView& residual_delta = back_propagation.backward_slots[layer][residual_delta_slot];
        if (residual_delta.data) copy(input_delta, residual_delta);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
