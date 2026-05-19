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

/// @brief Element-wise addition layer that sums two input tensors of identical shape (residual connections).
class Addition final : public Layer
{
public:

    /// @brief Constructs an addition layer for the given input shape and label.
    Addition(const Shape& = {}, const string& = "");

    Shape get_output_shape() const override { return input_shape; }

    /// @brief Returns the tensor specs of intermediate backward buffers for a given batch size.
    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with a new input shape and label.
    void set(const Shape&, const string&);
    /// @copydoc Layer::set_input_shape
    void set_input_shape(const Shape& shape) override { set(shape, label); }

private:

    AddOp add;

    enum Backward {OutputDelta, InputDelta0, InputDelta1};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
