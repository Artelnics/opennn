//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L E   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scale_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void ScaleOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (!minimums.data)
    {
        copy(input, output);
        return;
    }

    if (invert)
        unscale(input, minimums, maximums, means, standard_deviations, scalers,
                min_range, max_range, output);
    else
        scale(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
