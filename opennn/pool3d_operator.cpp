//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L 3 D   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pool3d_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

void Pool3dOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);
    TensorView& indices     = get_output(forward_propagation, layer, 1);

    if (method == Max)
        max_pooling_3d_forward(input, output, indices, is_training);
    else
        average_pooling_3d_forward(input, output);
}

void Pool3dOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);
    if (input_delta.empty()) return;

    if (method == Max)
        max_pooling_3d_backward(get_output(forward_propagation, layer, 1), output_delta, input_delta);
    else
        average_pooling_3d_backward(get_input(forward_propagation, layer), output_delta, input_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
