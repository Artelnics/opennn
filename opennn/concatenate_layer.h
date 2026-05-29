//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

// n-ary concatenation along the channel axis. All inputs must agree on H and W;
// channel counts add. Mirrors how YOLO v3 FPN heads merge an upsampled top-down
// feature map with an earlier backbone stage of the same spatial size.
class Concatenate final : public Layer
{
public:

    Concatenate(const Shape& = {},
                const vector<Index>& per_input_channels = {},
                const string& = "concatenate_layer");

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override;

    Index get_inputs_number() const { return ssize(concatenate.input_channels); }
    const vector<Index>& get_input_channels() const { return concatenate.input_channels; }

    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    void set(const Shape&, const vector<Index>& per_input_channels, const string&);
    void set_input_shape(const Shape&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    ConcatenateOp concatenate;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
