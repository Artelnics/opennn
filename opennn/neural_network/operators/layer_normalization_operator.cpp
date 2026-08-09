//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/layer_normalization_operator.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

namespace opennn
{

void LayerNormalizationOperator::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> LayerNormalizationOperator::parameter_specs() const
{

    const size_t count = (method == NormalizationMethod::RMS) ? 1 : 2;
    return vector<TensorSpec>(count, {Shape{embedding_dimension}, Type::FP32});
}

void LayerNormalizationOperator::link_parameters(span<const TensorView> views)
{
    if (method == NormalizationMethod::RMS)
    {
        beta = {};
        link_views(views, {&gamma});
        return;
    }
    link_views(views, {&gamma, &beta});
}

void LayerNormalizationOperator::link_gradients(span<const TensorView> views)
{
    if (method == NormalizationMethod::RMS)
    {
        beta_gradient = {};
        link_views(views, {&gamma_gradient});
        return;
    }
    link_views(views, {&gamma_gradient, &beta_gradient});
}

void LayerNormalizationOperator::init_defaults()
{
    if (gamma.data) gamma.as_vector().setOnes();
    if (beta.data)  beta.as_vector().setZero();
}

void LayerNormalizationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool  )
{
    const TensorView& input = get_input(forward_propagation, layer);

    if (method == NormalizationMethod::RMS)
    {

        TensorView& inverse_rms = get_output(forward_propagation, layer);
        TensorView& normalized  = get_output(forward_propagation, layer, 2);
        TensorView& output      = get_output(forward_propagation, layer, 3);

        rms_normalization_forward(input, gamma, inverse_rms, normalized, output, epsilon);
        return;
    }

    TensorView& means       = get_output(forward_propagation, layer);
    TensorView& stds        = get_output(forward_propagation, layer, 1);
    TensorView& normalized  = get_output(forward_propagation, layer, 2);
    TensorView& output      = get_output(forward_propagation, layer, 3);

    if (fuse_add)
    {

        const TensorView& residual = forward_propagation.inputs[layer][1];
        layer_normalization_add_forward(input, residual, gamma, beta, means, stds, normalized, normalized, output);
        return;
    }

    layer_normalization_forward(input, gamma, beta, means, stds, normalized, output);
}

void LayerNormalizationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);

    if (method == NormalizationMethod::RMS)
    {
        const TensorView& inverse_rms = get_output(forward_propagation, layer);
        const TensorView& normalized  = get_output(forward_propagation, layer, 2);

        rms_normalization_backward(get_input(forward_propagation, layer), output_delta,
                                   inverse_rms, normalized, gamma, gamma_gradient, input_delta);
        return;
    }

    const TensorView& stds       = get_output(forward_propagation, layer, 1);
    const TensorView& normalized = get_output(forward_propagation, layer, 2);

    const TensorView& norm_input = fuse_add ? normalized : get_input(forward_propagation, layer);

    layer_normalization_backward(norm_input, output_delta, get_output(forward_propagation, layer),
                        stds, normalized, gamma, gamma_gradient, beta_gradient,
                        input_delta);

    if (fuse_add)
    {
        TensorView& residual_delta = back_propagation.slots[layer][2];
        if (residual_delta.data && residual_delta.data != input_delta.data)
            copy(input_delta, residual_delta);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
