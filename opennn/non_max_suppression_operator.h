//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P P R E S S I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct NonMaxSuppressionOp : Operator
{
    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.4f;

    void set(const Shape& input_shape,
             Index new_boxes_per_cell,
             float new_confidence_threshold,
             float new_iou_threshold);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;

private:
    void apply(const TensorView& input, TensorView& output) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
