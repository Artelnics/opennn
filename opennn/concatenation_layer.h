//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Concatenation final : public Layer
{
public:

    Concatenation(const Shape& = {},
                  const vector<Index>& per_input_channels = {},
                  const string& = "concatenation_layer");

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override;

    Index get_inputs_number() const { return ssize(concatenation.input_channels); }
    const vector<Index>& get_input_channels() const { return concatenation.input_channels; }

    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    void set(const Shape&, const vector<Index>& per_input_channels, const string&);
    void set_input_shape(const Shape&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    ConcatenationOp concatenation;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
