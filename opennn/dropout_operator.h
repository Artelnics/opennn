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

struct DropoutOp : Operator
{
    float rate = 0.0f;

    Buffer mask;

    vector<size_t> save_slots;

    bool active() const { return rate > 0.0f; }

    void set_rate(float new_rate);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    DropoutOp() = default;
    DropoutOp(DropoutOp&&) noexcept = default;
    DropoutOp& operator=(DropoutOp&&) noexcept = default;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
