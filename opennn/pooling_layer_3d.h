//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file pooling_layer_3d.h
 * @brief Declares the Pooling3d layer: sequence pooling that reduces over
 *        the time / sequence axis of rank-2 inputs.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "pooling_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @class Pooling3d
 * @brief Sequence-pooling layer for rank-2 inputs (sequence_length, features).
 *
 * Reduces along the sequence axis using either max or average pooling and
 * outputs a single feature vector per sample. Reuses PoolingMethod from
 * pooling_layer.h.
 */
class Pooling3d final : public Layer
{
public:

    /**
     * @brief Constructs a Pooling3d layer.
     * @param input_shape Per-sample input shape (sequence_length, features).
     * @param pooling_method Reduction method applied along the sequence axis.
     * @param label Human-readable label assigned to this layer.
     */
    Pooling3d(const Shape& input_shape = {0, 0},
              const PoolingMethod& pooling_method = PoolingMethod::MaxPooling,
              const string& label = "sequence_pooling_layer");

    /** @brief Returns the per-sample input shape (sequence_length, features). */
    Shape get_input_shape() const override { return {sequence_length, input_features}; }
    /**
     * @brief Returns the per-sample output shape.
     * @return Shape with a single feature axis (sequence reduced away).
     */
    Shape get_output_shape() const override;

    /** @brief Sequence (time) length of the input. */
    Index get_sequence_length() const { return sequence_length; }
    /** @brief Number of features per time step. */
    Index get_input_features() const { return input_features; }

    /** @brief Configured pooling reduction method. */
    PoolingMethod get_pooling_method() const { return pooling_method; }

    /**
     * @brief Returns the canonical string name of the pooling method.
     * @return "MaxPooling" or "AveragePooling".
     */
    string write_pooling_method() const;

    /** @brief Returns the single Pool3d operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&pool3d}; }

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Specs for Input, MaximalIndices and Output slots.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer.
     * @param input_shape Per-sample input shape (sequence_length, features).
     * @param pooling_method Reduction method.
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape,
             const PoolingMethod& pooling_method,
             const string& label);

    /**
     * @brief Updates the input shape (sequence_length, features).
     * @param new_input_shape New per-sample input shape.
     */
    void set_input_shape(const Shape& new_input_shape) override
    {
        sequence_length = new_input_shape[0];
        input_features  = new_input_shape[1];
    }

    /**
     * @brief Sets the pooling method directly.
     * @param new_pooling_method New reduction method enum.
     */
    void set_pooling_method(const PoolingMethod& new_pooling_method) { pooling_method = new_pooling_method; }
    /**
     * @brief Sets the pooling method by name.
     *
     * Receives "MaxPooling" or "AveragePooling".
     */
    void set_pooling_method(const string&);

    /**
     * @brief Routes output gradients back to the corresponding sequence
     *        positions in InputDelta.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (input shape and method).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (input shape and method).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Sequence (time) length of the input. */
    Index sequence_length = 0;
    /** @brief Number of features per time step. */
    Index input_features = 0;

    /** @brief Selected reduction method (max or average). */
    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    /** @brief Underlying sequence-pooling operator. */
    Pool3d pool3d;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward { Input, MaximalIndices, Output };
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward { OutputDelta, InputDelta };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
