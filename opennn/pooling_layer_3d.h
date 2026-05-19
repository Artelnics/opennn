//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "pooling_layer.h"

namespace opennn
{

/// @brief Sequence pooling layer reducing the time axis of a (sequence, features) input.
class Pooling3d final : public Layer
{
public:

    /// @brief Constructs a sequence pooling layer.
    /// @param input_shape Shape of the input as (sequence_length, input_features).
    /// @param pooling_method Reduction method applied along the sequence axis.
    /// @param name Layer name used for serialization.
    Pooling3d(const Shape& = {0, 0},
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    /// @brief Returns the input tensor shape (sequence_length, input_features).
    Shape get_input_shape() const override { return {sequence_length, input_features}; }

    /// @brief Returns the output tensor shape after sequence-axis pooling.
    Shape get_output_shape() const override;

    Index get_sequence_length() const { return sequence_length; }
    Index get_input_features() const { return input_features; }

    PoolingMethod get_pooling_method() const { return pooling_method; }

    /// @brief Returns the tensor specifications used during forward propagation.
    vector<TensorSpec> get_forward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with a new input shape, pooling method and name.
    void set(const Shape&, const PoolingMethod&, const string&);

    /// @brief Updates the layer for a new input shape, preserving pooling method and label.
    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape, pooling_method, get_label());
    }

    void set_output_shape(const Shape&) override {}

    /// @brief Sets the pooling method via enum.
    void set_pooling_method(PoolingMethod);

    /// @brief Sets the pooling method by name ("MaxPooling" or "AveragePooling").
    void set_pooling_method(const string&);

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

    /// @brief Writes the layer configuration to a JSON writer.
    void write_JSON_body(JsonWriter&) const override;

private:

    Index sequence_length = 0;
    Index input_features = 0;

    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    Pool3dOp pool3d;

    enum Forward { Input, MaximalIndices, Output };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
