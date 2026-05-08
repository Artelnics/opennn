//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file addition_layer.h
 * @brief Declares the Addition layer used to implement residual / skip
 *        connections by summing two upstream tensors elementwise.
 */

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/**
 * @class Addition
 * @brief Elementwise tensor addition layer (residual / skip connections).
 *
 * Receives two upstream inputs of identical shape and writes their sum into
 * its output buffer. Used as the merge point in ResNet-style and
 * Transformer-style architectures, where the layer's two inputs are wired
 * via NeuralNetwork::layer_input_indices.
 *
 * The layer has no trainable parameters; its backward pass simply forwards
 * the output gradient to both inputs.
 */
class Addition final : public Layer
{
public:

    /**
     * @brief Constructs an Addition layer.
     * @param input_shape Per-sample input (and output) shape.
     * @param label Human-readable label assigned to this layer.
     */
    Addition(const Shape& input_shape = {}, const string& label = "");

    /** @brief Returns the per-sample input shape. */
    Shape get_input_shape() const override { return input_shape; }
    /** @brief Returns the per-sample output shape (same as input). */
    Shape get_output_shape() const override { return input_shape; }

    /**
     * @brief Specifications of the backward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return One delta buffer per upstream input.
     */
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    /** @brief Returns the single Add operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&add}; }

    /**
     * @brief Re-initializes the layer.
     * @param input_shape Per-sample input shape.
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape, const string& label);
    /** @brief Updates the input shape and re-runs set() to keep label consistent. */
    void set_input_shape(const Shape& shape) override { set(shape, label); }

    /**
     * @brief Forwards the output gradient to both upstream input deltas.
     * @param fp Forward intermediates from the matching forward pass.
     * @param bp BackPropagation buffer in which to accumulate gradients.
     * @param layer Index of this layer inside the network.
     */
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;


private:

    /** @brief Per-sample input (and output) shape. */
    Shape input_shape;

    /** @brief Underlying elementwise-addition operator. */
    Add add;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta0, InputDelta1};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
