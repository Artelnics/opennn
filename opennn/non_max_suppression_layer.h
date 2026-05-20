//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P P R E S S I O N   L A Y E R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class NonMaxSuppression final : public Layer
{
public:

    NonMaxSuppression(const Shape& = {},
                      Index boxes_per_cell = 1,
                      float confidence_threshold = 0.5f,
                      float iou_threshold = 0.4f,
                      const string& = "non_max_suppression_layer");

    Shape get_output_shape() const override;

    vector<TensorSpec> get_backward_specs(Index) const override { return {}; }

    void set(const Shape&,
             Index boxes_per_cell,
             float confidence_threshold,
             float iou_threshold,
             const string&);

    void set_input_shape(const Shape&) override;

private:

    NonMaxSuppressionOp nms;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
