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

struct NonMaxSuppressionOperator : Operator
{
    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.4f;

    void set(const Shape&,
             Index,
             float,
             float);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

private:
    void apply(const TensorView&, TensorView&) const;

    mutable vector<float> cpu_input_staging;
    mutable vector<float> cpu_output_staging;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
