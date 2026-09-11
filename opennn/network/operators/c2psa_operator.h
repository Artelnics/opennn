// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"

namespace opennn
{

struct C2PSAOperator : Operator
{
    // The forward slot layout. The layer plans these and the operator reads
    // them, so it lives where both can see it; the operator used to reach into
    // slots[layer][1] .. [6] by ordinal and agree with the layer by accident.
    enum Slot
    {
        Input,
        Split,
        Query,
        Key,
        AttentionWeights,
        Value,
        Concatenated,
        ForwardScratch,
        Output
    };

    Index h = 0, w = 0, channels = 0;

    TensorView Wq, Wk, Wv, Wout;
    TensorView dWq, dWk, dWv, dWout;

    void set(Index new_h, Index new_w, Index new_channels);

    vector<TensorSpec> parameter_specs() const override;
    vector<ParameterSlot> parameter_slots() override;
    void set_parameters_random() override { set_parameters_glorot(); }
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    optional<size_t> forward_scratch_slot;
    optional<size_t> backward_scratch_slot;
};

}
