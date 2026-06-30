//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D R O P O U T   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct DropoutOperator : Operator
{
    float rate = 0.0f;

    Buffer mask;

    bool active() const { return rate > 0.0f; }

    void set_rate(float);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

    DropoutOperator() = default;
    DropoutOperator(DropoutOperator&&) noexcept = default;
    DropoutOperator& operator=(DropoutOperator&&) noexcept = default;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
