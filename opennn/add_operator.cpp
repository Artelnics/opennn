//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "add_operator.h"
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

void AddOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    const vector<TensorView>& inputs = get_inputs(fp, layer);
    TensorView& output               = get_output(fp, layer);

    check(inputs, output);

    add(inputs[0], inputs[1], output);
    for (size_t i = 2; i < inputs.size(); ++i)
        add(output, inputs[i], output);
}

void AddOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(bp, layer);

    for (size_t i = 0; i < input_delta_slots.size(); ++i)
    {
        TensorView& input_delta = get_input_delta(bp, layer, i);
        if (!input_delta.empty())
            copy(output_delta, input_delta);
    }
}

void AddOp::check(const vector<TensorView>& inputs, const TensorView& output) const
{
    throw_if(inputs.size() < 2,
             "Add: needs at least 2 inputs.");

    for (const TensorView& input : inputs)
        throw_if(input.size() != output.size(),
                 "Add: tensor dimensions do not match.");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
