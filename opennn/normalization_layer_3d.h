//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/// @brief Layer normalization over the embedding axis of a 3D (batch, sequence, embedding) tensor.
class Normalization3d final : public Layer
{
public:

    /// @brief Constructs a 3D layer normalization layer.
    /// @param input_shape Input shape as (sequence_length, embedding_dimension).
    /// @param name Layer name used for serialization.
    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    /// @brief Returns the input tensor shape (sequence_length, embedding_dimension).
    Shape get_input_shape() const override;

    /// @brief Returns the output tensor shape (same as input).
    Shape get_output_shape() const override;

    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    /// @brief Returns the tensor specifications used during forward propagation.
    vector<TensorSpec> get_forward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with new sequence length, embedding dimension and name.
    void set(Index = 0, Index = 0, const string& = "normalization_layer_3d");

    /// @brief Updates the layer for a new input shape.
    void set_input_shape(const Shape&) override;

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

private:

    Index sequence_length = 0;
    Index embedding_dimension = 0;

    LayerNormOp layer_norm;

    enum Forward {Input, Means, StandardDeviations, NormalizedInput, Output};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
