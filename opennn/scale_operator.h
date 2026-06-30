//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L E   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct ScaleOperator : Operator
{
    bool invert = false;

    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
