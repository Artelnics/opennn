//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "scaling_layer.h"

namespace opennn
{

class Unscaling final : public Scaling
{
public:

    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    Shape get_input_shape()  const noexcept override { return { Index(scalers.size()) }; }
    Shape get_output_shape() const noexcept override { return { Index(scalers.size()) }; }

    void set(Index = 0, const string& = "unscaling_layer");

    void set_input_shape(const Shape&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
