//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pool_operator.h"
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

#ifdef OPENNN_HAS_CUDA

namespace
{

void configure_pooling_descriptor(PoolOperator& op)
{
    if (op.pool_height <= 0 || op.pool_width <= 0) return;

    if (!op.pooling_descriptor)
    {
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&op.pooling_descriptor.handle));
        op.pooling_descriptor.deleter = &cudnnDestroyPoolingDescriptor;
    }

    const cudnnPoolingMode_t mode = (op.method == PoolOperator::Max)
        ? CUDNN_POOLING_MAX
        : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    CHECK_CUDNN(cudnnSetPooling2dDescriptor(op.pooling_descriptor,
                                            mode,
                                            CUDNN_PROPAGATE_NAN,
                                            to_int(op.pool_height), to_int(op.pool_width),
                                            to_int(op.padding_height), to_int(op.padding_width),
                                            to_int(op.row_stride), to_int(op.column_stride)));
}

}

#endif


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

#ifdef OPENNN_HAS_CUDA
    configure_pooling_descriptor(*this);
#endif
}

void PoolOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    auto& forward_slots = forward_propagation.forward_slots[layer];
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    TensorView empty;
    TensorView& indices = view_at_slot_or(forward_slots, output_slots, 1, empty);

    pooling_2d_forward(input, output, indices,
                       input_height, input_width, input_channels,
                       pool_height, pool_width,
                       row_stride, column_stride,
                       padding_height, padding_width,
                       method == Max,
                       pooling_descriptor);
}

void PoolOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& forward_slots = forward_propagation.forward_slots[layer];

    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);
    if (input_delta.empty()) return;

    TensorView empty;
    const TensorView& indices = view_at_slot_or(forward_slots, output_slots, 1, empty);

    pooling_2d_backward(get_input(forward_propagation, layer), get_output(forward_propagation, layer),
                        output_delta, indices, input_delta,
                        input_height, input_width, input_channels,
                        pool_height, pool_width,
                        row_stride, column_stride,
                        padding_height, padding_width,
                        method == Max,
                        pooling_descriptor);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
