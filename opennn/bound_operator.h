//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct BoundOperator : Operator
{
    enum class Method { NoBounding, Bounding };

    Method method = Method::Bounding;

    TensorView lower;
    TensorView upper;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
