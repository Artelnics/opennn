// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/layers/layer.h"
#include "opennn/network/operators/activation_operator.h"

namespace opennn
{

class Activation final : public Layer
{
public:

    Activation(const Shape& = {},
               const string& = "ReLU",
               const string& = "activation_layer");

    Shape get_output_shape() const override { return input_shape; }
    ActivationFunction get_output_activation() const override { return activation_operator.activation_function; }

    void set(const Shape&, const string&, const string&);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 1, 2, 3); }

private:

    ActivationOperator activation_operator;
};

}
