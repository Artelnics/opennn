//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "add_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void AdditionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const vector<TensorView>& inputs = get_inputs(forward_propagation, layer);
    TensorView& output               = get_output(forward_propagation, layer);

    copy(inputs[0], output);
    for (size_t i = 1; i < inputs.size(); ++i)
        add(output, inputs[i], output);
}

void AdditionOperator::back_propagate(ForwardPropagation&, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    for (size_t i = 0; i < input_delta_slots.size(); ++i)
    {
        TensorView& input_delta = get_input_delta(back_propagation, layer, i);
        if (!input_delta.empty())
            copy(output_delta, input_delta);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
