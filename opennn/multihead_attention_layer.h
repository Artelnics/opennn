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
    Index get_embedding_dimension() const { return input_shape.back(); }
    Index get_heads_number() const { return heads_number; }
    Index get_head_dimension() const;

    type get_scaling_factor() const;

    Shape get_input_shape() const override;

    Shape get_output_shape() const override;

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index embedding_dimension = get_embedding_dimension();
        const Index head_dimension = get_head_dimension();

        return {{batch_size, heads_number, query_sequence_length, head_dimension},         // Query
                {batch_size, heads_number, source_sequence_length, head_dimension},        // Key
                {batch_size, heads_number, query_sequence_length, source_sequence_length}, // AttentionWeights
                {batch_size, query_sequence_length, embedding_dimension},                  // ConcatenatedAttentionOutputs
                {batch_size, heads_number, source_sequence_length, head_dimension},        // Value
                {batch_size, query_sequence_length, embedding_dimension}};                 // Outputs (must be last)
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        const Index embedding_dimension = get_embedding_dimension();
        const Index head_dimension = get_head_dimension();

        const Index q_len = query_sequence_length;
        const Index s_len = source_sequence_length;
        const Index h_num = heads_number;

        return {{batch_size, q_len, embedding_dimension}, // Input Query Gradients
                {batch_size, s_len, embedding_dimension}, // Input Source Gradients (dX_s)
                {batch_size, h_num, q_len, s_len}, // Intermediate: Attention Weight Gradients: {batch, h_num, q_len, s_len}
                {batch_size, q_len, embedding_dimension}, // Intermediate: Concatenated Output Gradients: {batch, q_len, emb_dim}
                {batch_size, h_num, q_len, head_dimension}, // Query Gradients
                {batch_size, h_num, s_len, head_dimension}, // Key Gradients
                {batch_size, h_num, s_len, head_dimension}};  // Value Gradients
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        input_shape = new_input_shape;
        query_sequence_length = new_input_shape[0];
        embedding_dimension = new_input_shape[1];
    }

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type r) { dropout_rate = r; }

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void to_XML(XmlPrinter&) const override;
    void from_XML(const XmlDocument&) override;

private:

    void apply_causal_mask(Tensor4&) const;
    void apply_key_padding_mask(const TensorMap3&, Tensor4&) const;

    Index embedding_dimension;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    enum Parameters {QueryWeights, QueryBiases, KeyWeights, KeyBiases, ValueWeights, ValueBiases,
                     ProjectionWeights, ProjectionBiases};
    enum Forward {Inputs, Query, Key, AttentionWeights, ConcatenatedAttentionOutputs, Value};
    // Outputs is always the last forward slot (wiring convention) — access via .back()
    enum Backward {OutputGradient, InputQueryGradient, InputSourceGradient,
                   AttentionWeightGradient, ConcatenatedOutputGradient,
                   QueryGradient, KeyGradient, ValueGradient};

    bool use_causal_mask = false;

    MatrixR causal_mask;
    MatrixB key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    static constexpr type padding_threshold = type(1e-7f);
    static constexpr type mask_value = type(-1e9f);
    static constexpr type minus_inf = -numeric_limits<float>::infinity();
};

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
