//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E R G E   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "merge_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void MergeOperator::set(Index new_heads_number, Index new_query_sequence_length, Index new_head_dimension, Type new_compute_dtype)
{
    heads_number          = new_heads_number;
    query_sequence_length = new_query_sequence_length;
    head_dimension        = new_head_dimension;
    compute_dtype         = new_compute_dtype;
}

void MergeOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const Index batch_size = forward_propagation.batch_size;

    const TensorView source_4d = get_input(forward_propagation, layer)
        .reshape({batch_size, heads_number, query_sequence_length, head_dimension});
    TensorView dest_4d = get_output(forward_propagation, layer).reshape({batch_size, query_sequence_length, heads_number, head_dimension});

    merge_heads(source_4d, dest_4d);
}

void MergeOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const Index batch_size = forward_propagation.batch_size;

    const TensorView concat_gradient_4d = get_output_delta(back_propagation, layer)
        .reshape({batch_size, query_sequence_length, heads_number, head_dimension});
    TensorView heads_gradient_4d = get_input(forward_propagation, layer)
        .reshape({batch_size, heads_number, query_sequence_length, head_dimension});

    split_heads(concat_gradient_4d, heads_gradient_4d);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
