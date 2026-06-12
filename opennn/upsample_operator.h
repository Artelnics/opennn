//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L E   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct UpsampleOp : Operator
{
    Index input_height = 0;
    Index input_width = 0;
    Index channels = 0;
    Index scale_factor = 2;

    void set(Index in_h, Index in_w, Index ch, Index scale);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void apply(const TensorView& input, TensorView& output) const;
    void apply_delta(const TensorView& output_delta, TensorView& input_delta) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
