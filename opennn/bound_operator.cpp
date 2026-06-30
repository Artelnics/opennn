//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bound_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void BoundOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool /*is_training*/)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (method == Method::NoBounding || !lower.data)
    {
        copy(input, output);
        return;
    }

    bound(input, lower, upper, output);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
