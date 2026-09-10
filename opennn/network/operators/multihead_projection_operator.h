// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"
#include "opennn/network/operators/combination_operator.h"

namespace opennn
{

void split_heads(const TensorView&, TensorView&);
void concatenate_heads(const TensorView&, TensorView&);

struct MultiHeadProjectionOperator : CombinationOperator
{
    size_t input_view_index = 0;

    size_t scratch_slot = 0;

    size_t input_delta_slot_self  = 0;
    size_t input_delta_slot_cross = 0;
    bool accumulate_input_delta_self  = false;
    bool accumulate_input_delta_cross = false;

    bool interleaved_heads = false;

    void set(Index, Index, Index, Type);

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}
