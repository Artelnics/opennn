//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S W I G L U   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

void swiglu_forward(const TensorView&, const TensorView&, TensorView&);
void swiglu_backward(const TensorView&, const TensorView&, const TensorView&,
                     TensorView&, TensorView&);

struct SwiGLUOperator : Operator
{
    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
