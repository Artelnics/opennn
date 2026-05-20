//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Detection final : public Layer
{
public:

    Detection(const Shape& = {},
              const vector<array<float, 2>>& = {},
              const string& = "detection_layer");

    Shape get_output_shape() const override { return input_shape; }
    const vector<array<float, 2>>& get_anchors() const { return detection.anchors; }

    void set(const Shape&, const vector<array<float, 2>>&, const string&);
    void set_input_shape(const Shape&) override;
    void set_anchors(const vector<array<float, 2>>&);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    DetectionOp detection;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
