// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"

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
