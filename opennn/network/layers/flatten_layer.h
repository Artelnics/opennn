// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/layers/layer.h"

namespace opennn
{

class Flatten final : public Layer
{
public:

    Flatten(const Shape& = {}, const string& = "flatten_layer");

    Shape get_output_shape() const override { return { input_shape.size() }; }

    vector<TensorSpec> get_forward_specs(Index) const override { return {}; }

    vector<TensorSpec> get_backward_specs(Index) const override { return {}; }

    void set(const Shape&, const string& = "flatten_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 1, 2, 3); }

    void apply_input_shape(const Shape& new_input_shape) override { set(new_input_shape, label); }
};

}
