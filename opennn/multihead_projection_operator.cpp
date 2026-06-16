//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   P R O J E C T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "multihead_projection_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

void MultiHeadProjectionOp::set(Index new_input_features, Index new_heads_number,
                              Index new_head_dimension, Type new_compute_dtype)
{
    input_features = new_input_features;
    compute_dtype  = new_compute_dtype;

    combination.set(input_features, new_heads_number * new_head_dimension, compute_dtype);
}

void MultiHeadProjectionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/)
{
    auto& forward_slots = fp.forward_slots[layer];
    const auto& input_views = get_inputs(fp, layer);
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    TensorView& head_output = get_output(fp, layer);

    const Index batch_size     = input.shape[0];
    const Index seq_len        = input.shape[1];
    const Index rows           = batch_size * seq_len;
    const Index heads_number   = head_output.shape[1];
    const Index head_dimension = head_output.shape[3];

    TensorView& scratch     = forward_slots[scratch_slots[0]];
    TensorView  scratch_2d  = scratch.reshape({rows, input_features});
    TensorView  scratch_4d  = scratch.reshape({batch_size, seq_len, heads_number, head_dimension});
    TensorView  input_2d    = input.reshape({rows, input_features});

    combination.apply(input_2d, scratch_2d);
    split_heads(scratch_4d, head_output);
}

void MultiHeadProjectionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    auto& forward_slots = fp.forward_slots[layer];
    auto& backward_slots = bp.backward_slots[layer];

    const auto& input_views = get_inputs(fp, layer);
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    const bool self_attention = (input_views.size() == 1);

    const TensorView& head_delta = get_output_delta(bp, layer);

    const Index batch_size     = input.shape[0];
    const Index seq_len        = input.shape[1];
    const Index rows           = batch_size * seq_len;
    const Index heads_number   = head_delta.shape[1];
    const Index head_dimension = head_delta.shape[3];

    TensorView& scratch     = forward_slots[scratch_slots[0]];
    TensorView  scratch_4d  = scratch.reshape({batch_size, seq_len, heads_number, head_dimension});
    TensorView  scratch_2d  = scratch.reshape({rows, input_features});
    TensorView  input_2d    = input.reshape({rows, input_features});

    merge_heads(head_delta, scratch_4d);

    TensorView& input_delta    = backward_slots[(self_attention ? input_delta_slots_self : input_delta_slots_cross)[0]];
    TensorView  input_delta_2d = input_delta.empty()
        ? TensorView{}
        : input_delta.reshape({rows, input_features});
    const bool  accumulate     = self_attention ? accumulate_input_delta_self : accumulate_input_delta_cross;

    combination.apply_delta(scratch_2d, input_2d, input_delta_2d, accumulate);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
