//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   R E L U   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file dense_relu_layer.h
 * @brief Declares DenseRelu, a fused Dense + ReLU layer optimized for
 *        CUDA Graph capture and the cuBLASLt RELU_BIAS epilogue.
 */

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/**
 * @class DenseRelu
 * @brief Dense + ReLU fused into a single forward op.
 *
 * Uses the cuBLASLt RELU_BIAS epilogue on GPU and a ReLU baked into
 * Combination::apply_cpu when the epilogue is RELU_BIAS. No batch
 * normalization, no dropout, activation hard-wired to ReLU — keeps
 * forward_propagate() branch-free for CUDA Graph capture.
 *
 * Use this layer instead of Dense when ReLU is the desired activation
 * and the runtime cost of branching matters (e.g. latency-bound inference).
 */
class DenseRelu final : public Layer
{
public:

    /**
     * @brief Constructs a DenseRelu layer.
     * @param input_shape Per-sample input shape; empty means "set later".
     * @param output_shape Per-sample output shape; the trailing dimension
     *                     is the number of output features.
     * @param label Human-readable label assigned to this layer.
     */
    DenseRelu(const Shape& input_shape = {},
              const Shape& output_shape = {},
              const string& label = "dense_relu_layer");

    /** @brief Returns the per-sample input shape. */
    Shape get_input_shape() const override { return input_shape; }

    /**
     * @brief Returns the per-sample output shape.
     * @return Same leading dimensions as the input with the configured number
     *         of output features as the trailing dimension.
     */
    Shape get_output_shape() const override;

    /**
     * @brief Number of input features (last dimension of the input shape).
     * @return 0 if the input shape is empty; the trailing dimension otherwise.
     */
    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }

    /**
     * @brief Sequence length when the input is 2D, 1 otherwise.
     * @return Leading dimension of the input shape if rank-2, else 1.
     */
    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    /** @brief Activation function fused at the end of this layer (always ReLU). */
    Activation::Function get_output_activation() const override { return Activation::Function::ReLU; }

    /** @brief Returns the single operator (Combination with fused ReLU). */
    vector<Operator*> get_operators() override { return {&combination}; }

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Specs for the Input and Output slots in the Forward enum.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer; same arguments as the constructor.
     * @param input_shape Per-sample input shape.
     * @param output_shape Per-sample output shape.
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape = {},
             const Shape& output_shape = {},
             const string& label = "dense_relu_layer");

    /** @brief Updates the input shape and re-shapes weight tensors accordingly. */
    void set_input_shape(const Shape&) override;

    /** @brief Updates the output features and re-shapes weight tensors accordingly. */
    void set_output_shape(const Shape&) override;

    /** @brief Reconfigures inner operators when the compute dtype changes. */
    void on_compute_dtype_changed() override { configure_operators(); }

    /**
     * @brief Backward pass through the fused ReLU and Combination operators.
     * @param fp Forward intermediates from the matching forward pass.
     * @param bp BackPropagation buffer in which to accumulate gradients.
     * @param layer Index of this layer inside the network.
     */
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;


private:

    /** @brief Per-sample input shape. */
    Shape input_shape;

    /** @brief Number of output features (last dimension of the output shape). */
    Index output_features = 0;

    /** @brief Linear projection with fused ReLU epilogue. */
    Combination combination;
    /** @brief Activation handle kept for backward-pass derivative dispatch. */
    Activation  activation;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};

    /** @brief Reconfigures inner operators after a shape or dtype change. */
    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
