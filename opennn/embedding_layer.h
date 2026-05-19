//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/// @brief Token-id to dense vector embedding layer with optional scaling and positional encoding.
class Embedding final : public Layer
{
public:

    /// @brief Constructs an embedding layer.
    /// @param input_shape Input shape as (vocabulary_size, sequence_length).
    /// @param embedding_dimension Size of each embedding vector.
    /// @param name Layer name used for serialization.
    Embedding(const Shape& = {0, 0},
              Index = 0,
              const string& = "embedding_layer");

    /// @brief Returns the input tensor shape (sequence_length of token ids).
    Shape get_input_shape() const override { return {sequence_length}; }

    /// @brief Returns the output tensor shape (sequence_length, embedding_dimension).
    Shape get_output_shape() const override;

    Index get_vocabulary_size() const { return vocabulary_size; }
    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    vector<TensorSpec> get_backward_specs(Index) const override { return {}; }

    /// @brief Reconfigures the layer with vocabulary size, sequence length, embedding dimension and name.
    void set(Index = 0,
             Index = 0,
             Index = 0,
             const string& = "embedding_layer");

    /// @brief Enables or disables embedding scaling by sqrt(embedding_dimension).
    void set_scale_embedding(bool enabled) { embedding_lookup.scale_embedding = enabled; }

    /// @brief Enables or disables sinusoidal positional encoding added to the embeddings.
    void set_add_positional_encoding(bool enabled) { embedding_lookup.add_positional_encoding = enabled; }

    /// @brief Sets the dropout rate applied to the embedding output.
    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

    /// @brief Writes the layer configuration to a JSON writer.
    void write_JSON_body(JsonWriter&) const override;

private:

    Index vocabulary_size = 0;
    Index sequence_length = 0;
    Index embedding_dimension = 0;

    EmbeddingLookupOp embedding_lookup;
    DropoutOp         dropout;

    enum Backward {OutputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
