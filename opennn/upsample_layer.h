//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "upsample_operator.h"

namespace opennn
{

class Upsample final : public Layer
{
public:

    Upsample(const Shape& = {},
             Index scale_factor = 2,
             const string& = "upsample_layer");

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override;

    void set(const Shape&, Index, const string&);
    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override {}
    void set_scale_factor(Index);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    UpsampleOperator upsample;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
