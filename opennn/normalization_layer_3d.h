//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file normalization_layer_3d.h
 * @brief Declares the Normalization3d layer: layer normalization applied
 *        across the embedding dimension of rank-2 inputs.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @class Normalization3d
 * @brief Layer normalization across the embedding dimension of rank-2 inputs.
 *
 * Inputs are rank-2 tensors (sequence_length, embedding_dimension). The
 * layer computes the mean and standard deviation along the embedding axis
 * for each (sample, position) pair and rescales the input accordingly.
 * Trainable affine parameters (gain, bias) are exposed via the underlying
 * LayerNorm operator.
 */
class Normalization3d final : public Layer
{
public:

    /**
     * @brief Constructs a Normalization3d layer.
     * @param input_shape Per-sample input shape (sequence_length, embedding_dimension).
     * @param label Human-readable label assigned to this layer.
     */
    Normalization3d(const Shape& input_shape = Shape({0,0}),
                    const string& label = "normalization_layer_3d");

    /** @brief Returns the per-sample input shape (sequence_length, embedding_dimension). */
    Shape get_input_shape() const override;
    /** @brief Returns the per-sample output shape (same as input). */
    Shape get_output_shape() const override;

    /** @brief Sequence length of the input. */
    Index get_sequence_length() const { return sequence_length; }
    /** @brief Embedding dimension along which normalization is applied. */
    Index get_embedding_dimension() const { return embedding_dimension; }

    /** @brief Returns the single LayerNorm operator. */
    vector<Operator*> get_operators() override;

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return One spec per slot in the Forward enum.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer.
     * @param sequence_length Sequence length of the input.
     * @param embedding_dimension Embedding dimension along which to normalize.
     * @param label Human-readable label.
     */
    void set(Index sequence_length = 0,
             Index embedding_dimension = 0,
             const string& label = "normalization_layer_3d");

    /**
     * @brief Updates the input shape (sequence_length, embedding_dimension).
     * @param new_input_shape New per-sample input shape; ignored if rank < 2.
     */
    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank >= 2)
        {
            sequence_length     = new_input_shape[0];
            embedding_dimension = new_input_shape[1];
        }
    }

    /**
     * @brief Backward pass through layer normalization.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (sequence length and
     *        embedding dimension).
     */
    void read_JSON_body(const Json*) override;

private:

    /** @brief Sequence length of the input. */
    Index sequence_length = 0;
    /** @brief Embedding dimension along which normalization is applied. */
    Index embedding_dimension = 0;

    /** @brief Underlying layer-norm operator (with affine gain/bias). */
    LayerNorm layer_norm;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Means, StandardDeviations, NormalizedInput, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
