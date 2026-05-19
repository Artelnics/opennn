//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "math_utilities.h"

namespace opennn
{

/// @brief Multi-head scaled dot-product attention layer used in transformer architectures.
class MultiHeadAttention final : public Layer
{
public:

    /// @brief Constructs a self-attention layer where queries and keys share the same sequence.
    /// @param query_dimensions Query input shape as (sequence_length, embedding_dimension).
    /// @param heads_number Number of attention heads.
    /// @param name Layer name used for serialization.
    MultiHeadAttention(const Shape& = Shape({0, 0}),
                       Index = 0,
                       const string& = {});

    /// @brief Constructs a cross-attention layer with separate query and source (key/value) sequences.
    /// @param new_query_dimensions Query shape as (query_sequence_length, embedding_dimension).
    /// @param new_source_dimensions Source shape as (source_sequence_length, embedding_dimension).
    /// @param heads_number Number of attention heads.
    /// @param name Layer name used for serialization.
    MultiHeadAttention(const Shape& new_query_dimensions,
                       const Shape& new_source_dimensions,
                       Index = 0,
                       const string& = {});

    /// @brief Returns the input tensor shape.
    Shape get_input_shape() const override;

    /// @brief Returns the output tensor shape.
    Shape get_output_shape() const override;

    Index get_query_sequence_length() const { return query_sequence_length; }
    Index get_source_sequence_length() const { return source_sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }
    Index get_heads_number() const { return heads_number; }

    /// @brief Returns the per-head dimension (embedding_dimension / heads_number).
    Index get_head_dimension() const
    {
        return (heads_number == 0) ? 0 : Index(embedding_dimension / heads_number);
    }

    /// @brief Returns the per-head tensor shape used internally during attention.
    Shape get_heads_shape(Index batch_size) const
    {
        return {batch_size, heads_number, query_sequence_length, get_head_dimension()};
    }

    /// @brief Returns the shape used when concatenating heads back to the embedding dimension.
    Shape get_concat_shape(Index batch_size) const
    {
        return {batch_size, query_sequence_length, heads_number, get_head_dimension()};
    }

    /// @brief Returns the tensor specifications used during forward propagation.
    vector<TensorSpec> get_forward_specs(Index batch_size) const override;

    /// @brief Returns the tensor specifications used during back propagation.
    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with new sequence, embedding, head sizes and causal flag.
    void set(Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    /// @brief Updates the layer for a new input shape.
    void set_input_shape(const Shape&) override;

    /// @brief Rebuilds projection operators when the compute dtype changes.
    void on_compute_dtype_changed() override;

    /// @brief Sets the dropout rate applied to the attention weights.
    void set_dropout_rate(float new_dropout_rate) { attention.set_dropout_rate(new_dropout_rate); }

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

    /// @brief Writes the layer configuration to a JSON writer.
    void write_JSON_body(JsonWriter&) const override;

private:

    Index embedding_dimension = 0;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    MultiHeadProjectionOp query_projection;
    MultiHeadProjectionOp key_projection;
    MultiHeadProjectionOp value_projection;
    CombinationOp         output_projection;
    AttentionOp           attention;
    MergeOp               merge;

    enum Forward {Input, Query, Key, AttentionWeights, AttentionWeightsDropped,
                  ConcatenatedAttentionOutputs, Value, TransposeScratch, Output};
    enum Backward {
        OutputDelta,
        InputQueryDelta,         // final dInput query, embed shape
        InputSourceDelta,        // final dInput source, embed shape
        AttentionWeightDelta,    // unfused attention scratch
        ValueHeadDelta,          // dV, head shape
        ConcatenatedOutputDelta, // dConcat, embed shape
        QueryHeadDelta,          // dQ, head shape
        KeyHeadDelta             // dK, head shape
    };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
