//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S W I G L U   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "swiglu_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void SwiGLUOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool /*is_training*/)
{
    const TensorView& gate = get_input(forward_propagation, layer, 0);
    const TensorView& up   = get_input(forward_propagation, layer, 1);
    TensorView& output     = get_output(forward_propagation, layer);

    swiglu_forward(gate, up, output);
}

void SwiGLUOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& gate         = get_input(forward_propagation, layer, 0);
    const TensorView& up           = get_input(forward_propagation, layer, 1);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    TensorView& gate_delta = get_input_delta(back_propagation, layer, 0);
    TensorView& up_delta   = get_input_delta(back_propagation, layer, 1);

    swiglu_backward(output_delta, gate, up, gate_delta, up_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
