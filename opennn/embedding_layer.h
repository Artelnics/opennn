//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file embedding_layer.h
 * @brief Declares the Embedding layer: a vocabulary lookup followed by an
 *        optional scale, positional encoding and dropout.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @class Embedding
 * @brief Token-id-to-vector lookup layer used in language models.
 *
 * Inputs are token id sequences (rank-1, length sequence_length). Each id
 * is replaced by its row in an embedding table of shape
 * (vocabulary_size, embedding_dimension), producing a rank-2 output of
 * shape (sequence_length, embedding_dimension).
 *
 * Optional flags toggle Transformer-style sqrt(d_model) scaling and
 * positional encoding addition; an optional dropout is applied at the end.
 */
class Embedding final : public Layer
{
public:

    /**
     * @brief Constructs an Embedding layer.
     * @param input_shape Per-sample input shape; carries (sequence_length,
     *                    vocabulary_size) when both are known up front.
     * @param embedding_dimension Width of each embedding vector.
     * @param label Human-readable label assigned to this layer.
     */
    Embedding(const Shape& input_shape = {0, 0},
              Index embedding_dimension = 0,
              const string& label = "embedding_layer");

    /** @brief Returns the per-sample input shape (sequence_length,). */
    Shape get_input_shape() const override { return {sequence_length}; }
    /**
     * @brief Returns the per-sample output shape.
     * @return (sequence_length, embedding_dimension).
     */
    Shape get_output_shape() const override;

    /** @brief Number of distinct tokens in the vocabulary. */
    Index get_vocabulary_size() const { return vocabulary_size; }
    /** @brief Sequence length expected at input. */
    Index get_sequence_length() const { return sequence_length; }
    /** @brief Width of each embedding vector. */
    Index get_embedding_dimension() const { return embedding_dimension; }

    /**
     * @brief Returns the active operators in pipeline order.
     * @return EmbeddingLookup followed by Dropout if its rate is non-zero.
     */
    vector<Operator*> get_operators() override;

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Specs for the Input and Output slots.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer.
     * @param vocabulary_size Number of distinct tokens.
     * @param sequence_length Sequence length expected at input.
     * @param embedding_dimension Width of each embedding vector.
     * @param label Human-readable label.
     */
    void set(Index vocabulary_size = 0,
             Index sequence_length = 0,
             Index embedding_dimension = 0,
             const string& label = "embedding_layer");

    /**
     * @brief Enables Transformer-style sqrt(d_model) scaling on the
     *        embedding table output.
     * @param enabled True to scale; false to leave embeddings as-is.
     */
    void set_scale_embedding(bool enabled) { embedding_lookup.scale_embedding = enabled; }
    /**
     * @brief Enables addition of a sinusoidal positional encoding after lookup.
     * @param enabled True to add positional encodings; false to skip.
     */
    void set_add_positional_encoding(bool enabled) { embedding_lookup.add_positional_encoding = enabled; }
    /**
     * @brief Sets the dropout rate applied at the layer output.
     * @param rate Probability of dropping each unit (0 disables dropout).
     */
    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    /**
     * @brief Backward pass: scatters output gradients into the embedding
     *        table rows referenced by the input ids.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (vocabulary size, sequence
     *        length, embedding dimension, scale/positional flags, dropout).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (vocabulary size, sequence
     *        length, embedding dimension, scale/positional flags, dropout).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Number of distinct tokens in the vocabulary. */
    Index vocabulary_size = 0;
    /** @brief Sequence length expected at input. */
    Index sequence_length = 0;
    /** @brief Width of each embedding vector. */
    Index embedding_dimension = 0;

    /** @brief Underlying lookup operator (with scale and positional flags). */
    EmbeddingLookup embedding_lookup;
    /** @brief Optional dropout applied at the layer output during training. */
    Dropout         dropout;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
