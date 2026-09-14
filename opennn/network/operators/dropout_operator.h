// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"

namespace opennn
{

void dropout_forward(TensorView&, TensorView&, float);
void dropout_backward(TensorView&, const TensorView&, float);

struct DropoutOperator : Operator
{
    float rate = 0.0f;
    optional<size_t> mask_slot;

    bool active() const { return rate > 0.0f; }

    void set_rate(float);

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;
};

}
