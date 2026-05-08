//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file flatten_layer.h
 * @brief Declares the Flatten layer that reshapes a multi-dimensional
 *        input into a 1D vector.
 */

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/**
 * @class Flatten
 * @brief Reshape layer that collapses every input axis into a single feature axis.
 *
 * Given an input of any rank, the output is a 1D tensor of length equal to
 * the product of the input dimensions. Forward and backward passes are
 * pure reshapes (no element movement); the layer has no trainable
 * parameters.
 *
 * Typically inserted between convolutional / pooling layers and dense
 * layers in CNN classifiers.
 */
class Flatten final : public Layer
{
public:

    /**
     * @brief Constructs a Flatten layer.
     * @param input_shape Per-sample input shape; the output length is its
     *                    flat size.
     */
    Flatten(const Shape& input_shape = {});

    /** @brief Returns the per-sample input shape. */
    Shape get_input_shape() const override { return input_shape; }
    /**
     * @brief Returns the per-sample output shape.
     * @return 1D shape with length equal to the flat size of the input.
     */
    Shape get_output_shape() const override { return { input_shape.size() }; }

    /** @brief Returns the single Flat operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&flat}; }

    /**
     * @brief Re-initializes the layer.
     *
     * Receives the new per-sample input shape.
     */
    void set(const Shape&);

    /**
     * @brief Updates the input shape; equivalent to calling set().
     * @param new_input_shape New per-sample input shape.
     */
    void set_input_shape(const Shape& new_input_shape) override { set(new_input_shape); }

    /**
     * @brief Backward pass: reshapes the output gradient back to the input shape.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;


private:

    /** @brief Per-sample input shape. */
    Shape input_shape;

    /** @brief Underlying flatten operator. */
    Flat flat;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
