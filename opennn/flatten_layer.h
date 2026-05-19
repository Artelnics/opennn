//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/// @brief Flatten layer that reshapes a multi-dimensional input into a single 1D feature vector.
class Flatten final : public Layer
{
public:

    /// @brief Constructs a flatten layer for the given input shape.
    Flatten(const Shape& = {});

    /// @brief Returns the flattened 1D output shape (product of input dimensions).
    Shape get_output_shape() const override { return { input_shape.size() }; }

    /// @brief Reconfigures the layer with a new input shape.
    void set(const Shape&);

    /// @copydoc Layer::set_input_shape
    void set_input_shape(const Shape& new_input_shape) override { set(new_input_shape); }

private:

    FlatOp flat;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
