//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pool_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void PoolOperator::set(Index input_h, Index input_w, Index input_c,
               Index pool_h, Index pool_w,
               Index new_row_stride, Index new_column_stride,
               Index padding_h, Index padding_w,
               Method new_method)
{
    input_height    = input_h;
    input_width     = input_w;
    input_channels  = input_c;
    pool_height     = pool_h;
    pool_width      = pool_w;
    row_stride      = new_row_stride;
    column_stride   = new_column_stride;
    padding_height  = padding_h;
    padding_width   = padding_w;
    method          = new_method;
}

void PoolOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    auto& forward_slots = forward_propagation.forward_slots[layer];
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    TensorView& indices = slot_or(forward_slots, output_slots, 1);

    pooling_2d_forward(input, output, indices,
                       input_height, input_width, input_channels,
                       pool_height, pool_width,
                       row_stride, column_stride,
                       padding_height, padding_width,
                       method == Max);
}

void PoolOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& forward_slots = forward_propagation.forward_slots[layer];

    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);
    if (input_delta.empty()) return;

    const TensorView& indices = slot_or(forward_slots, output_slots, 1);

    pooling_2d_backward(get_input(forward_propagation, layer), get_output(forward_propagation, layer),
                        output_delta, indices, input_delta,
                        input_height, input_width, input_channels,
                        pool_height, pool_width,
                        row_stride, column_stride,
                        padding_height, padding_width,
                        method == Max);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
