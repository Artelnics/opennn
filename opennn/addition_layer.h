//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Addition final : public Layer
{
public:

    Addition(const Shape& = {}, const string& = "", Index num_inputs = 2);

    Shape get_output_shape() const override { return input_shape; }

    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    void set(const Shape&, const string&, Index num_inputs);
    void set_input_shape(const Shape& shape) override { set(shape, label, inputs_number); }

    Index get_inputs_number() const { return inputs_number; }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    AddOp add;

    Index inputs_number = 2;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
