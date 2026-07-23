//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S W I G L U   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

// silu(gate) * up element-wise over two same-shaped tensors read through
// input_slots (the gate and up projection slots of a gated Dense).
struct SwiGLUOperator : Operator
{
    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
