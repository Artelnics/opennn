//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

struct C2PSAOperator : Operator
{
    Index h = 0, w = 0, channels = 0;

    TensorView Wq, Wk, Wv, Wout;
    TensorView dWq, dWk, dWv, dWout;

    void set(Index new_h, Index new_w, Index new_channels);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;
    void set_parameters_random() override { set_parameters_glorot(); }
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    size_t forward_scratch_slot = SIZE_MAX;
    size_t backward_scratch_slot = SIZE_MAX;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
