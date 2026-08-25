//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L O O K U P   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/embedding_lookup_operator.h"
#include "opennn/core/json.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/core/device_backend.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/operators/kernel_embedding.cuh"
#include "opennn/core/cuda/kernel_quantization.cuh"
#endif

namespace opennn
{

static void embedding_lookup_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, bool, const TensorView&);
static void embedding_lookup_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, Index, Index, Index, bool);

static void embedding_lookup_forward_cpu(const TensorView& indices, const TensorView& weights,
                                  const TensorView& positional_encoding, TensorView& output,
                                  Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                  bool scale_embedding, bool add_positional_encoding)
{
    const Index total_tokens = indices.size();

    MatrixMap output_mat        = output.as_flat_matrix();
    const MatrixMap weights_mat = weights.as_matrix();
    const float* input_indices  = indices.as<float>();

    static atomic<bool> out_of_range_warned{false};

    #pragma omp parallel for schedule(static)
    for (Index i = 0; i < total_tokens; ++i)
    {
        const Index token_id = static_cast<Index>(input_indices[i]);

        if (token_id == 0)
        {
            output_mat.row(i).setZero();
            continue;
        }

        if (token_id < 0 || token_id >= vocabulary_size)
        {
            if (!out_of_range_warned.exchange(true))
                cerr << format("EmbeddingLookup warning: token id {} out of range [0, {}); zeroing row. Further warnings suppressed.\n", token_id, vocabulary_size);
            output_mat.row(i).setZero();
            continue;
        }

        output_mat.row(i).noalias() = weights_mat.row(token_id);

        if (scale_embedding)
            output_mat.row(i) *= sqrt(to_type(embedding_dimension));

        if (add_positional_encoding)
            output_mat.row(i) += positional_encoding.as_matrix().row(i % sequence_length);
    }
}

static void embedding_lookup_backward_cpu(const TensorView& indices, const TensorView& output_delta,
                                   const TensorView& weight_gradient, const TensorView& positional_gradient,
                                   Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                   bool scale_embedding)
{
    const Index total_elements = indices.size();

    MatrixMap output_delta_map = output_delta.as_flat_matrix();
    MatrixMap weight_gradients = weight_gradient.as_matrix().setZero();
    const float scale = scale_embedding ? sqrt(to_type(embedding_dimension)) : 1.0f;

    const bool accumulate_positional = !positional_gradient.empty() && positional_gradient.get_data() != nullptr;

    for (Index token_index = 0; token_index < total_elements; ++token_index)
    {
        const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);

        if (vocabulary_index <= 0 || vocabulary_index >= vocabulary_size)
            continue;

        weight_gradients.row(vocabulary_index).noalias() += scale * output_delta_map.row(token_index);
    }

    if (accumulate_positional)
    {
        MatrixMap positional_gradients = positional_gradient.as_matrix();
        positional_gradients.setZero();
        for (Index token_index = 0; token_index < total_elements; ++token_index)
        {
            const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);
            if (vocabulary_index <= 0 || vocabulary_index >= vocabulary_size)
                continue;
            positional_gradients.row(token_index % sequence_length).noalias() += output_delta_map.row(token_index);
        }
    }
}

void embedding_lookup_forward(const TensorView& indices, const TensorView& weights,
                              const TensorView& positional_encoding, TensorView& output,
                              Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                              bool scale_embedding, bool add_positional_encoding,
                              const TensorView& weight_scale)
{
    if (output.is_cuda())
        return embedding_lookup_forward_gpu(indices, weights, positional_encoding, output,
                                            sequence_length, embedding_dimension, vocabulary_size,
                                            scale_embedding, add_positional_encoding, weight_scale);
    throw_if(weights.is_int8(), "embedding_lookup_forward: INT8 weights are CUDA-only.");
    embedding_lookup_forward_cpu(indices, weights, positional_encoding, output,
                                 sequence_length, embedding_dimension, vocabulary_size,
                                 scale_embedding, add_positional_encoding);
}

void embedding_lookup_backward(const TensorView& indices, const TensorView& output_delta,
                               const TensorView& weight_gradient, const TensorView& positional_gradient,
                               Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                               bool scale_embedding)
{
    if (output_delta.is_cuda())
        return embedding_lookup_backward_gpu(indices, output_delta, weight_gradient, positional_gradient,
                                             sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
    embedding_lookup_backward_cpu(indices, output_delta, weight_gradient, positional_gradient,
                                  sequence_length,
                                  embedding_dimension, vocabulary_size, scale_embedding);
}

void compute_token_valid_lengths(const TensorView& indices, Index sequence_length, vector<Index>& valid_lengths)
{
    const Index total = indices.size();
    const Index batch_size = sequence_length > 0 ? total / sequence_length : 0;

    valid_lengths.assign(batch_size, sequence_length);
    if (batch_size == 0) return;

    throw_if(indices.is_cuda(),
             "compute_token_valid_lengths: CUDA token ids are counted on the device (token_valid_lengths_cuda).");

    const float* ids = indices.as<float>();

    for (Index b = 0; b < batch_size; ++b)
    {
        Index count = 0;
        const float* row = ids + b * sequence_length;
        for (Index s = 0; s < sequence_length; ++s)
            if (static_cast<Index>(row[s]) != 0) ++count;
        valid_lengths[b] = count;
    }
}

#ifdef OPENNN_HAS_CUDA

static void embedding_lookup_forward_gpu(const TensorView& indices, const TensorView& weights,
                                  const TensorView& positional_encoding, TensorView& output,
                                  Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                  bool scale_embedding, bool add_positional_encoding,
                                  const TensorView& weight_scale)
{
    if (weights.is_int8())
    {
        throw_if(weight_scale.empty(),
                 "embedding_lookup_forward: INT8 weights require a per-row scale vector.");
        output.dispatch([&]<typename T>() {
            embedding_forward_w8_cuda<T>(
                output.size(),
                indices.as<float>(),
                weights.as<int8_t>(),
                weight_scale.as<float>(),
                add_positional_encoding ? positional_encoding.as<float>() : nullptr,
                output.as<T>(),
                to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size),
                scale_embedding);
        });
        return;
    }

    output.dispatch([&]<typename T>() {
        weights.dispatch([&]<typename TW>() {
            embedding_forward_cuda<TW, T>(
                output.size(),
                indices.as<float>(),
                weights.as<TW>(),
                add_positional_encoding ? positional_encoding.as<float>() : nullptr,
                output.as<T>(),
                to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size),
                scale_embedding);
        });
    });
}

static void embedding_lookup_backward_gpu(const TensorView& indices, const TensorView& output_delta,
                                   const TensorView& weight_gradient, const TensorView& positional_gradient,
                                   Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                   bool scale_embedding)
{
    weight_gradient.set_zero_async();

    const bool accumulate_positional = !positional_gradient.empty() && positional_gradient.get_data() != nullptr;
    if (accumulate_positional) positional_gradient.set_zero_async();

    output_delta.dispatch([&]<typename T>() {
        embedding_backward_cuda<T>(
            output_delta.size(),
            indices.as<float>(),
            output_delta.as<T>(),
            weight_gradient.as<float>(),
            accumulate_positional ? positional_gradient.as<float>() : nullptr,
            to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size), scale_embedding);
    });
}

#else

OPENNN_CUDA_STUB(void, embedding_lookup_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, bool, const TensorView&))
OPENNN_CUDA_STUB(void, embedding_lookup_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, Index, Index, Index, bool))

#endif

void EmbeddingLookupOperator::set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension)
{
    vocabulary_size     = new_vocabulary_size;
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> EmbeddingLookupOperator::parameter_specs() const
{
    vector<TensorSpec> specs = {{{vocabulary_size, embedding_dimension},
                                 weights_follow_compute_dtype ? weights_dtype : Type::FP32}};
    if (positional_trainable)
        specs.push_back({{sequence_length, embedding_dimension}, Type::FP32});
    return specs;
}

vector<Operator::SlotQuantization> EmbeddingLookupOperator::parameter_quantization() const
{
    return {{vocabulary_size, 0}};
}

vector<TensorSpec> EmbeddingLookupOperator::state_specs() const
{
    if (!add_positional_encoding || positional_trainable)
        return {};

    return {{{sequence_length, embedding_dimension}, Type::FP32}};
}

void EmbeddingLookupOperator::link_parameters(span<const TensorView> views)
{
    if (positional_trainable && views.size() > 1)
        link_views(views, {&weights, &positional_encoding});
    else
        link_views(views, {&weights});
}

void EmbeddingLookupOperator::link_parameter_scales(span<const TensorView> views)
{
    if (views.empty()) return;
    weight_scale = views[0];
}

void EmbeddingLookupOperator::link_gradients(span<const TensorView> views)
{
    if (positional_trainable && views.size() > 1)
        link_views(views, {&weight_gradient, &positional_gradient});
    else
        link_views(views, {&weight_gradient});
}

void EmbeddingLookupOperator::link_states(span<const TensorView> views)
{
    if (positional_trainable || views.empty()) return;
    positional_encoding = views[0];
}

void EmbeddingLookupOperator::initialize_states()
{
    if (positional_trainable || !positional_encoding.get_data()) return;
    init_positional_encoding();
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
    if (!positional_trainable || positional_encoding.empty() || !positional_encoding.get_data()) return;
    MatrixMap positional_matrix = positional_encoding.as_matrix();
    set_random_normal(positional_matrix, 0.0f, 0.02f);
}

void EmbeddingLookupOperator::init_positional_encoding()
{
    if (!add_positional_encoding) return;
    if (positional_encoding.empty() || !positional_encoding.get_data()) return;

    float* table = positional_encoding.as<float>();
    const Index half   = embedding_dimension / 2;
    const float half_f = float(embedding_dimension) / 2.0f;

    VectorR divisors(embedding_dimension);
    for (Index j = 0; j < embedding_dimension; ++j)
        divisors(j) = pow(10000.0f, (j < half ? j : j - half) / half_f);

    const Index values_count = sequence_length * embedding_dimension;

    #pragma omp parallel for
    for (Index value = 0; value < values_count; ++value)
    {
        const Index i = value / embedding_dimension;
        const Index j = value % embedding_dimension;
        table[value] = (j < half) ? sin(i / divisors(j)) : cos(i / divisors(j));
    }
}

void EmbeddingLookupOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode  )
{
    const TensorView& indices = get_input(forward_propagation, layer);
    TensorView& output        = get_output(forward_propagation, layer);

    if (export_valid_lengths)
    {
#ifdef OPENNN_HAS_CUDA
        if (indices.is_cuda())
        {
            const Index batch_size = sequence_length > 0 ? indices.size() / sequence_length : 0;
            token_valid_lengths_cuda(batch_size, sequence_length, indices.as<float>(),
                                     forward_propagation.device_valid_lengths_slot(layer, batch_size),
                                     device::get_compute_stream());
        }
        else
#endif
        compute_token_valid_lengths(indices, sequence_length,
                                    forward_propagation.valid_lengths[layer]);
    }

    embedding_lookup_forward(indices, weights, positional_encoding, output,
                             sequence_length, embedding_dimension, vocabulary_size,
                             scale_embedding, add_positional_encoding, weight_scale);
}

void EmbeddingLookupOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& indices      = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    embedding_lookup_backward(indices, output_delta, weight_gradient, positional_gradient,
                              sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

void EmbeddingLookupOperator::load_state_from_JSON(const Json*  )
{

    if (positional_encoding.is_cuda()) return;
    init_positional_encoding();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
