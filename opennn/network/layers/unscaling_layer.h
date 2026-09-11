// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/layers/scaling_layer.h"

namespace opennn
{

class Unscaling final : public Scaling
{
public:

    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    void set(Index = 0, const string& = "unscaling_layer");

    void apply_input_shape(const Shape&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;
};

}
