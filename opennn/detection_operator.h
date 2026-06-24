//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct DetectionOp : Operator
{
    enum class ClassActivation { Softmax, Sigmoid };

    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    ClassActivation class_activation = ClassActivation::Softmax;

    vector<array<float, 2>> anchors;
    mutable Buffer device_anchors;  // GPU mirror of anchors; lazily populated

    void set(const Shape& input_shape, const vector<array<float, 2>>& new_anchors);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void apply(const TensorView& input, TensorView& output) const;
    void apply_delta(const TensorView& output, const TensorView& output_delta, TensorView& input_delta) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
