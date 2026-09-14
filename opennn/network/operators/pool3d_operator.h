// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"

namespace opennn
{

void max_pooling_3d_forward(const TensorView&, TensorView&, TensorView&, bool is_training, SequenceLengths);
void average_pooling_3d_forward(const TensorView&, TensorView&, SequenceLengths);
void max_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void average_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&, SequenceLengths);
void first_token_3d_forward(const TensorView&, TensorView&);
void first_token_3d_backward(const TensorView&, TensorView&);

struct Pool3dOperator : Operator
{
    enum Method { Max, Average, First };
    Method method = Average;

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}
