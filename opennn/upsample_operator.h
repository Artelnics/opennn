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

struct UpsampleOperator : Operator
{
    Index input_height = 0;
    Index input_width = 0;
    Index channels = 0;
    Index scale_factor = 2;

    void set(Index, Index, Index, Index);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

private:
    void apply(const TensorView&, TensorView&) const;
    void apply_delta(const TensorView&, TensorView&) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
