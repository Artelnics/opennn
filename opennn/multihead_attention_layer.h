//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

/**
 * @file multihead_attention_layer.h
 * @brief Declares the MultiHeadAttention layer used in Transformer-style
 *        encoders and decoders.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @class MultiHeadAttention
 * @brief Scaled dot-product attention with multiple heads and learned
 *        linear projections.
 *
 * Wraps four Combination/MultiHeadProjection operators (query, key, value,
 * output projection) and one Attention operator that performs the
 * scaled dot-product attention with optional dropout.
 *
 * Two input modes are supported:
 * - Self-attention: a single rank-2 input is used as query, key and value.
 * - Cross-attention: two rank-2 inputs (query side and source side); used
 *   in the decoder's encoder-decoder attention.
 */
class MultiHeadAttention final : public Layer
{
public:

    /**
     * @brief Constructs a self-attention layer.
     * @param input_shape Per-sample input shape (sequence_length, embedding_dimension).
     * @param heads_number Number of attention heads (must divide embedding_dimension).
     * @param label Human-readable label assigned to this layer.
     */
    MultiHeadAttention(const Shape& input_shape = Shape({0, 0}),
                       Index heads_number = 0,
                       const string& label = string());

    /**
     * @brief Constructs a cross-attention layer.
     * @param new_query_dimensions Query side input shape
     *                             (query_sequence_length, embedding_dimension).
     * @param new_source_dimensions Source side input shape
     *                              (source_sequence_length, embedding_dimension).
     * @param heads_number Number of attention heads.
     * @param label Human-readable label assigned to this layer.
     */
    MultiHeadAttention(const Shape& new_query_dimensions,
                       const Shape& new_source_dimensions,
                       Index heads_number = 0,
                       const string& label = string());

    /**
     * @brief Returns the per-sample input shape.
     * @return (query_sequence_length, embedding_dimension); subclasses of
     *         the network may provide an additional source input separately.
     */
    Shape get_input_shape() const override;
    /**
     * @brief Returns the per-sample output shape.
     * @return (query_sequence_length, embedding_dimension).
     */
    Shape get_output_shape() const override;

    /** @brief Length of the query side sequence. */
    Index get_query_sequence_length() const { return query_sequence_length; }
    /** @brief Length of the source side sequence (equal to query length for self-attention). */
    Index get_source_sequence_length() const { return source_sequence_length; }
    /** @brief Width of the embedding (model) dimension. */
    Index get_embedding_dimension() const { return embedding_dimension; }
    /** @brief Number of attention heads. */
    Index get_heads_number() const { return heads_number; }
    /**
     * @brief Per-head feature dimension.
     * @return embedding_dimension / heads_number, or 0 if heads_number is 0.
     */
    Index get_head_dimension() const
    {
        return (heads_number == 0) ? 0 : Index(embedding_dimension / heads_number);
    }
    /**
     * @brief Shape of the per-head attention scratch buffer.
     * @param batch_size Batch size used for sizing.
     * @return (batch_size, heads_number, query_sequence_length, head_dimension).
     */
    Shape get_heads_shape(Index batch_size) const
    {
        return {batch_size, heads_number, query_sequence_length, get_head_dimension()};
    }

    /**
     * @brief Shape of the concatenated attention output before projection.
     * @param batch_size Batch size used for sizing.
     * @return (batch_size, query_sequence_length, heads_number, head_dimension).
     */
    Shape get_concat_shape(Index batch_size) const
    {
        return {batch_size, query_sequence_length, heads_number, get_head_dimension()};
    }

    /**
     * @brief Convenience predicate: true when the layer was wired in
     *        self-attention mode (single input view).
     * @param forward_views ForwardPropagation views[layer] for this layer.
     * @return True if self-attention, false if cross-attention.
     */
    static bool is_self_attention(const vector<vector<TensorView>>& forward_views)
    {
        return forward_views[Input].size() == 1;
    }

    /**
     * @brief Returns the query side input view.
     * @param forward_views ForwardPropagation views[layer] for this layer.
     * @return Reference to the query input TensorView.
     */
    static const TensorView& get_query_input(const vector<vector<TensorView>>& forward_views)
    {
        return forward_views[Input][0];
    }

    /**
     * @brief Returns the source side input view (key/value source).
     * @param forward_views ForwardPropagation views[layer] for this layer.
     * @return The query input itself for self-attention; otherwise the
     *         second wired input.
     */
    static const TensorView& get_source_input(const vector<vector<TensorView>>& forward_views)
    {
        return is_self_attention(forward_views) ? forward_views[Input][0] : forward_views[Input][1];
    }

    /**
     * @brief Returns the active operators in pipeline order.
     * @return Q/K/V projections, Attention, then output projection.
     */
    vector<Operator*> get_operators() override;
    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return One spec per slot in the Forward enum.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    /**
     * @brief Specifications of the backward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return One spec per slot in the Backward enum.
     */
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer.
     * @param query_sequence_length Length of the query sequence.
     * @param source_sequence_length Length of the source sequence.
     * @param embedding_dimension Embedding (model) dimension.
     * @param heads_number Number of attention heads.
     * @param use_causal_mask True to apply a causal mask in self-attention.
     * @param label Human-readable label.
     */
    void set(Index query_sequence_length = 0,
             Index source_sequence_length = 0,
             Index embedding_dimension = 0,
             Index heads_number = 0,
             bool use_causal_mask = false,
             const string& label = "multihead_attention_layer");

    /**
     * @brief Updates the input shape; rejects shapes whose rank is not 2.
     * @param new_input_shape New per-sample input shape (sequence_length,
     *                        embedding_dimension).
     */
    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank != 2)
            throw runtime_error("MultiHeadAttention input shape must have rank 2.");

        query_sequence_length  = new_input_shape[0];
        source_sequence_length = new_input_shape[0];
        embedding_dimension    = new_input_shape[1];
    }

    /**
     * @brief Propagates a compute dtype change to all sub-operators.
     */
    void on_compute_dtype_changed() override
    {
        query_projection .compute_dtype           = compute_dtype;
        query_projection .combination.weight_type = compute_dtype;
        key_projection   .compute_dtype           = compute_dtype;
        key_projection   .combination.weight_type = compute_dtype;
        value_projection .compute_dtype           = compute_dtype;
        value_projection .combination.weight_type = compute_dtype;
        output_projection.weight_type             = compute_dtype;
        attention.compute_dtype                   = compute_dtype;
    }

    /**
     * @brief Sets the dropout rate applied to attention weights.
     * @param new_dropout_rate Probability of dropping each attention weight
     *                         (0 disables dropout).
     */
    void set_dropout_rate(float new_dropout_rate) { attention.set_dropout_rate(new_dropout_rate); }

    /**
     * @brief Forward pass: Q/K/V projections, scaled dot-product attention,
     *        head concatenation, output projection.
     *
     * Receives the ForwardPropagation buffer slice, this layer's index and
     * the training flag.
     */
    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;
    /**
     * @brief Backward pass through every operator in reverse order.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (heads, sequences, dimension,
     *        causal flag, dropout).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (heads, sequences, dimension,
     *        causal flag, dropout).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Embedding (model) dimension. */
    Index embedding_dimension = 0;
    /** @brief Number of attention heads. */
    Index heads_number = 0;
    /** @brief Length of the query side sequence. */
    Index query_sequence_length = 0;
    /** @brief Length of the source side sequence. */
    Index source_sequence_length = 0;

    /** @brief Linear projection of inputs into queries. */
    MultiHeadProjection query_projection;
    /** @brief Linear projection of inputs into keys. */
    MultiHeadProjection key_projection;
    /** @brief Linear projection of inputs into values. */
    MultiHeadProjection value_projection;
    /** @brief Final linear projection applied after head concatenation. */
    Combination         output_projection;
    /** @brief Scaled dot-product attention with optional dropout/mask. */
    Attention           attention;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Query, Key, AttentionWeights, AttentionWeightsDropped,
                  ConcatenatedAttentionOutputs, Value, TransposeScratch};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputQueryDelta, InputSourceDelta,
                   AttentionWeightDelta, ValueDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
