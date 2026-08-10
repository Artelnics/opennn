//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   P R O J E C T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"
#include "opennn/neural_network/operators/combination_operator.h"

namespace opennn
{

// Transpose the two middle axes: (batch, sequence, heads, dim) <-> (batch, heads, sequence, dim).
void split_heads(const TensorView&, TensorView&);
void merge_heads(const TensorView&, TensorView&);

struct MultiHeadProjectionOperator : CombinationOperator
{
    size_t input_view_index = 0;

    size_t scratch_slot = 0;

    size_t input_delta_slot_self  = 0;
    size_t input_delta_slot_cross = 0;
    bool accumulate_input_delta_self  = false;
    bool accumulate_input_delta_cross = false;

    void set(Index, Index, Index, Type);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
