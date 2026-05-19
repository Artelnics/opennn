//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   R E L U   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/// @brief Fused dense + ReLU layer; combines linear projection and ReLU activation in a single op for performance.
class DenseRelu final : public Layer
{
public:

    /// @brief Constructs a fused dense+ReLU layer with the given input and output shapes.
    DenseRelu(const Shape& = {},
              const Shape& = {},
              const string& = "dense_relu_layer");

    /// @brief Returns the layer output shape, derived from input shape and configured output features.
    Shape get_output_shape() const override;

    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }
    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    ActivationOp::Function get_output_activation() const override { return ActivationOp::Function::ReLU; }

    /// @brief Reconfigures the layer with new input/output shapes and label.
    void set(const Shape& = {},
             const Shape& = {},
             const string& = "dense_relu_layer");

    /// @copydoc Layer::set_input_shape
    void set_input_shape(const Shape&) override;
    /// @copydoc Layer::set_output_shape
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

private:

    Index output_features = 0;

    CombinationReluOp combination_relu;

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
