//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   V 8   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "detection_v8_operator.h"

namespace opennn
{

// Thin layer wrapper around DetectionV8Operator.
// Placed after the concatenated box+class logit branches in a YOLOv8-style head.
class DetectionV8 final : public Layer
{
public:

    DetectionV8(const Shape& = {}, const string& = "detection_v8");

    Shape get_output_shape() const override { return input_shape; }
    Index get_classes_number() const { return detection.classes_number; }

    void set(const Shape&, const string&);
    void set_input_shape(const Shape&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    DetectionV8Operator detection;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
