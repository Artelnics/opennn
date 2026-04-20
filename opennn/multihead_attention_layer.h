//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class MultiHeadAttention final : public Layer
{

public:

    MultiHeadAttention(const Shape& = Shape({0,0}),
                       Index = 0,
                       const string& = string());

    MultiHeadAttention(const Shape&,
                       const Shape&,
                       Index = 0,
                       const string& = string());

    Index get_query_sequence_length() const { return query_sequence_length; }
    Index get_source_sequence_length() const { return source_sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }
    Index get_heads_number() const { return heads_number; }
    Index get_head_dimension() const;

    // Per-head-split layout used by attention GEMMs: [B, H, Sq, D].
    Shape heads_shape(Index batch_size) const
    {
        return {batch_size, heads_number, query_sequence_length, get_head_dimension()};
    }

    // Concatenated-heads layout after merge_heads: [B, Sq, H, D].
    Shape concat_shape(Index batch_size) const
    {
        return {batch_size, query_sequence_length, heads_number, get_head_dimension()};
    }

    type get_scaling_factor() const;

    Shape get_input_shape() const override;

    Shape get_output_shape() const override;

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index head_dimension = get_head_dimension();

        const Index max_seq = max(query_sequence_length, source_sequence_length);

        return {{batch_size, heads_number, query_sequence_length, head_dimension},         // Query
                {batch_size, heads_number, source_sequence_length, head_dimension},        // Key
                {batch_size, heads_number, query_sequence_length, source_sequence_length}, // AttentionWeights
                {batch_size, query_sequence_length, embedding_dimension},                  // ConcatenatedAttentionOutputs
                {batch_size, heads_number, source_sequence_length, head_dimension},        // Value
                {batch_size, source_sequence_length},                                      // PaddingMask
                {batch_size, max_seq, embedding_dimension},                                // TransposeScratch
                {batch_size, heads_number, query_sequence_length, head_dimension},         // AttentionOutputTransposed
                {batch_size, query_sequence_length, embedding_dimension}};                 // Output (must be last)
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        const Index head_dimension = get_head_dimension();

        return {{batch_size, query_sequence_length, embedding_dimension},                          // InputQueryGradient
                {batch_size, source_sequence_length, embedding_dimension},                         // InputSourceGradient
                {batch_size, heads_number, query_sequence_length, source_sequence_length},         // AttentionWeightGradient
                {batch_size, query_sequence_length, embedding_dimension},                          // ConcatenatedOutputGradient
                {batch_size, heads_number, query_sequence_length, head_dimension},                 // QueryGradient (transposed)
                {batch_size, heads_number, source_sequence_length, head_dimension},                // KeyGradient (transposed)
                {batch_size, heads_number, source_sequence_length, head_dimension}};               // ValueGradient (transposed)
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        query_sequence_length = new_input_shape[0];
        embedding_dimension = new_input_shape[1];
    }

    void set(Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type r) { dropout_rate = r; }

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void to_XML(XmlPrinter&) const override;
    void from_XML(const XmlDocument&) override;

private:

    Index embedding_dimension;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    enum Parameters {QueryWeight, QueryBias, KeyWeight, KeyBias, ValueWeight, ValueBias,
                     ProjectionWeight, ProjectionBias};
    enum Forward {Input, Query, Key, AttentionWeights, ConcatenatedAttentionOutputs, Value,
                  PaddingMask, TransposeScratch, AttentionOutputTransposed};
    // Output is always the last forward slot (wiring convention) — access via .back()
    enum Backward {OutputGradient, InputQueryGradient, InputSourceGradient,
                   AttentionWeightGradient, ConcatenatedOutputGradient,
                   QueryGradient, KeyGradient, ValueGradient};

    // True when the layer is invoked with a single input (query == source).
    static bool is_self_attention(const vector<vector<TensorView>>& forward_views)
    {
        return forward_views[Input].size() == 1;
    }

    // Query input for the attention (first input).
    static const TensorView& get_query_input(const vector<vector<TensorView>>& forward_views)
    {
        return forward_views[Input][0];
    }

    // Source input for the attention: the second input when cross-attention, otherwise the (only) query input.
    static const TensorView& get_source_input(const vector<vector<TensorView>>& forward_views)
    {
        return is_self_attention(forward_views) ? forward_views[Input][0] : forward_views[Input][1];
    }

    bool use_causal_mask = false;

    MatrixR causal_mask;
    MatrixB key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);
};

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
