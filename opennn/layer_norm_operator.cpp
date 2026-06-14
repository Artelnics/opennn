//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer_norm_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

void LayerNormOp::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> LayerNormOp::parameter_specs() const
{
    return vector<TensorSpec>(2, {Shape{embedding_dimension}, Type::FP32});
}

void LayerNormOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
}

void LayerNormOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
}

void LayerNormOp::init_defaults()
{
    if (gamma.data) gamma.as_vector().setOnes();
    if (beta.data)  beta.as_vector().setZero();
}

void LayerNormOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/)
{
    const TensorView& input = get_input(fp, layer);
    TensorView& means       = get_output(fp, layer);
    TensorView& stds        = get_output(fp, layer, 1);
    TensorView& normalized  = get_output(fp, layer, 2);
    TensorView& output      = get_output(fp, layer, 3);

    layer_norm_forward(input, gamma, beta, means, stds, normalized, output);
}

void LayerNormOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    const TensorView& stds         = get_output(fp, layer, 1);
    const TensorView& normalized   = get_output(fp, layer, 2);
    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta        = get_input_delta(bp, layer);

    layer_norm_backward(get_input(fp, layer), output_delta, get_output(fp, layer),
                        stds, normalized, gamma, gamma_gradient, beta_gradient,
                        input_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
