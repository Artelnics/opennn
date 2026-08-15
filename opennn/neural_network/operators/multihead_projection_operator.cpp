//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   P R O J E C T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/multihead_projection_operator.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_attention.cuh"
#endif

namespace opennn
{

// Defined below: against the CUDA kernels, or as throwing stubs.
static void split_heads_gpu(const TensorView&, TensorView&);
static void merge_heads_gpu(const TensorView&, TensorView&);

static void transpose_middle_axes(const float* src, float* dst,
                                  Index batch_size, Index src_m1, Index src_m2, Index D)
{
    const Index blocks_count = batch_size * src_m2 * src_m1;

    #pragma omp parallel for schedule(static)
    for (Index block = 0; block < blocks_count; ++block)
    {
        const Index j = block % src_m1;
        const Index batch_i = block / src_m1;
        const Index i = batch_i % src_m2;
        const Index batch_index = batch_i / src_m2;

        memcpy(dst + block * D,
               src + ((batch_index * src_m1 + j) * src_m2 + i) * D,
               D * sizeof(float));
    }
}

void split_heads(const TensorView& source, TensorView& destination)
{
    if (source.is_cuda()) { split_heads_gpu(source, destination); return; }

    transpose_middle_axes(source.as<float>(), destination.as<float>(),
                          source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
}

void merge_heads(const TensorView& source, TensorView& destination)
{
    if (source.is_cuda()) { merge_heads_gpu(source, destination); return; }

    transpose_middle_axes(source.as<float>(), destination.as<float>(),
                          source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
}

#ifdef OPENNN_HAS_CUDA

static void split_heads_gpu(const TensorView& source, TensorView& destination)
{
    const Index sequence_length = source.shape[1];
    const Index heads_number = source.shape[2];
    const Index head_dimension = source.shape[3];

    destination.dispatch([&]<typename T>() {
        split_heads_cuda<T>(source.size(), source.as<T>(), destination.as<T>(),
                            to_int(sequence_length),
                            to_int(heads_number),
                            to_int(head_dimension));
    });
}

static void merge_heads_gpu(const TensorView& source, TensorView& destination)
{
    const Index heads_number = source.shape[1];
    const Index sequence_length = source.shape[2];
    const Index head_dimension = source.shape[3];

    destination.dispatch([&]<typename T>() {
        merge_heads_cuda<T>(source.size(), source.as<T>(), destination.as<T>(),
                            to_int(sequence_length),
                            to_int(heads_number),
                            to_int(head_dimension));
    });
}

#else

static void split_heads_gpu(const TensorView&, TensorView&) { throw runtime_error("split_heads_gpu: CUDA support not compiled in."); }
static void merge_heads_gpu(const TensorView&, TensorView&) { throw runtime_error("merge_heads_gpu: CUDA support not compiled in."); }

#endif

void MultiHeadProjectionOperator::set(Index new_input_features, Index new_heads_number,
                              Index new_head_dimension, Type new_compute_dtype)
{
    CombinationOperator::set(new_input_features,
                             new_heads_number * new_head_dimension,
                             new_compute_dtype);
}

void MultiHeadProjectionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{

    throw_if(tied_transposed || transposed_inference_active
             || fused_activation != ActivationFunction::Identity,
             "MultiHeadProjectionOperator: tied, transposed and fused-activation "
             "projections are not supported.");

    auto& forward_slots = forward_propagation.slots[layer];
    const auto& input_views = get_inputs(forward_propagation, layer);
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    TensorView& head_output = get_output(forward_propagation, layer);

    const Index batch_size     = input.shape[0];
    const Index seq_len        = input.shape[1];
    const Index rows           = batch_size * seq_len;
    const Index heads_number   = head_output.shape[1];
    const Index head_dimension = head_output.shape[3];

    TensorView&       scratch     = forward_slots[scratch_slot];
    TensorView        scratch_2d  = scratch.reshape({rows, input_features});
    const TensorView  scratch_4d  = scratch.reshape({batch_size, seq_len, heads_number, head_dimension});
    const TensorView  input_2d    = input.reshape({rows, input_features});

    linear_forward(input_2d, weights, bias, scratch_2d,
                   CUBLASLT_EPILOGUE_BIAS, nullptr, weight_scale);
    split_heads(scratch_4d, head_output);
}

void MultiHeadProjectionOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& forward_slots = forward_propagation.slots[layer];
    auto& backward_slots = back_propagation.slots[layer];

    const auto& input_views = get_inputs(forward_propagation, layer);
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    const bool self_attention = (input_views.size() == 1);

    const TensorView& head_delta = get_output_delta(back_propagation, layer);

    const Index batch_size     = input.shape[0];
    const Index seq_len        = input.shape[1];
    const Index rows           = batch_size * seq_len;
    const Index heads_number   = head_delta.shape[1];
    const Index head_dimension = head_delta.shape[3];

    TensorView&       scratch     = forward_slots[scratch_slot];
    TensorView        scratch_4d  = scratch.reshape({batch_size, seq_len, heads_number, head_dimension});
    const TensorView  scratch_2d  = scratch.reshape({rows, input_features});
    const TensorView  input_2d    = input.reshape({rows, input_features});

    merge_heads(head_delta, scratch_4d);

    TensorView& input_delta    = backward_slots[self_attention ? input_delta_slot_self : input_delta_slot_cross];

    TensorView  input_delta_2d = input_delta.empty()
        ? TensorView{}
        : input_delta.reshape({rows, input_features});

    const bool accumulate = self_attention
        ? accumulate_input_delta_self
        : accumulate_input_delta_cross;

    linear_backward(scratch_2d, input_2d, weights, weight_gradient, bias_gradient, input_delta_2d, accumulate);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
