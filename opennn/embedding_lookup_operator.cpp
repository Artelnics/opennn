//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L O O K U P   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "embedding_lookup_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void EmbeddingLookupOperator::set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension)
{
    vocabulary_size     = new_vocabulary_size;
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> EmbeddingLookupOperator::parameter_specs() const
{
    vector<TensorSpec> specs = {{{vocabulary_size, embedding_dimension},
                                 weights_follow_compute_dtype ? compute_dtype : Type::FP32}};
    if (positional_trainable)
        specs.push_back({{sequence_length, embedding_dimension}, Type::FP32});
    return specs;
}

vector<TensorSpec> EmbeddingLookupOperator::state_specs() const
{
    if (!add_positional_encoding || positional_trainable)
        return {};

    return {{{sequence_length, embedding_dimension}, Type::FP32}};
}

void EmbeddingLookupOperator::link_parameters(span<const TensorView> views)
{
    if (views.empty()) return;
    weights = views[0];
    if (positional_trainable && views.size() > 1)
        positional_encoding = views[1];
}

void EmbeddingLookupOperator::link_gradients(span<const TensorView> views)
{
    if (views.empty()) return;
    weight_gradient = views[0];
    if (positional_trainable && views.size() > 1)
        positional_gradient = views[1];
}

void EmbeddingLookupOperator::link_states(span<const TensorView> views)
{
    if (positional_trainable || views.empty()) return;
    const bool needs_init = positional_encoding.data == nullptr;
    positional_encoding = views[0];
    if (needs_init) init_positional_encoding();
}

void EmbeddingLookupOperator::set_parameters_random()
{
    if (weights.empty()) return;
    MatrixMap weights_matrix = weights.as_matrix();
    set_random_normal(weights_matrix, 0.0f, 1.0f);
    weights_matrix.row(0).setZero();
    init_trainable_positional();
}

void EmbeddingLookupOperator::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = glorot_limit(vocabulary_size, embedding_dimension);
    set_random_uniform(weights.as_vector(), -limit, limit);
    weights.as_matrix().row(0).setZero();
    init_trainable_positional();
}

void EmbeddingLookupOperator::init_trainable_positional()
{
    if (!positional_trainable || positional_encoding.empty() || !positional_encoding.data) return;
    MatrixMap positional_matrix = positional_encoding.as_matrix();
    set_random_normal(positional_matrix, 0.0f, 0.02f);
}

void EmbeddingLookupOperator::init_positional_encoding()
{
    if (!add_positional_encoding) return;
    if (positional_encoding.empty() || !positional_encoding.data) return;

    float* table = positional_encoding.as<float>();
    const Index half   = embedding_dimension / 2;
    const float half_f = float(embedding_dimension) / 2.0f;

    VectorR divisors(embedding_dimension);
    for (Index j = 0; j < embedding_dimension; ++j)
        divisors(j) = pow(10000.0f, (j < half ? j : j - half) / half_f);

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < sequence_length; ++i)
        for (Index j = 0; j < embedding_dimension; ++j)
            table[i * embedding_dimension + j] = (j < half)
                ? sin(i / divisors(j))
                : cos(i / divisors(j));
}

void EmbeddingLookupOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool /*is_training*/)
{
    const TensorView& indices = get_input(forward_propagation, layer);
    TensorView& output        = get_output(forward_propagation, layer);

    if (export_valid_lengths)
        compute_valid_lengths(indices, forward_propagation.attention_valid_lengths);

    embedding_lookup_forward(indices, weights, positional_encoding, output,
                             sequence_length, embedding_dimension, vocabulary_size,
                             scale_embedding, add_positional_encoding);
}

void EmbeddingLookupOperator::compute_valid_lengths(const TensorView& indices, vector<Index>& valid_lengths) const
{
    compute_token_valid_lengths(indices, sequence_length, valid_lengths);
}

void EmbeddingLookupOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& indices      = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    embedding_lookup_backward(indices, output_delta, weight_gradient, positional_gradient,
                              sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

void EmbeddingLookupOperator::load_state_from_JSON(const Json* /*parent*/)
{
    // The positional encoding is deterministic and is never written to JSON, so
    // there is nothing to deserialize: recompute it instead. This guarantees the
    // state is valid after a load even when the operator was linked before this
    // point with a buffer that init_positional_encoding() did not refill (the
    // link_states() init only runs on the first link). The device-resident copy
    // is populated by migrating the host state, not by this CPU-side fill, so we
    // only recompute while the state lives on the host.
    if (positional_encoding.is_cuda()) return;
    init_positional_encoding();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
