//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "pch.h"

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

    Index get_query_sequence_length() const;
    Index get_source_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;
    Index get_head_dimension() const;

    type get_scaling_factor() const;

    Shape get_input_shape() const override;

    Shape get_output_shape() const override;

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index embedding_dimension = get_embedding_dimension();
        const Index head_dimension = get_head_dimension();

        return {{batch_size, query_sequence_length, embedding_dimension},                  // Outputs
                {batch_size, heads_number, query_sequence_length, head_dimension},         // Query (split heads)
                {batch_size, heads_number, source_sequence_length, head_dimension},        // Key (split heads)
                {batch_size, heads_number, query_sequence_length, source_sequence_length}, // AttentionWeights
                {batch_size, query_sequence_length, embedding_dimension},                  // ConcatenatedAttentionOutputs
                {batch_size, heads_number, source_sequence_length, head_dimension}};       // Value (split heads)
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

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type);

    void apply_causal_mask(Tensor4&) const;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void print() const override;

    void to_XML(XMLPrinter&) const override;
    void from_XML(const XMLDocument&) override;

    void apply_key_padding_mask(const TensorMap3&, Tensor4&) const;

private:

    Index embedding_dimension;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    enum Parameters {QueryWeights, QueryBiases, KeyWeights, KeyBiases, ValueWeights, ValueBiases};
    enum Forward {Inputs, Query, Key, AttentionWeights, ConcatenatedAttentionOutputs, Value, Outputs};

    bool use_causal_mask = false;

    MatrixR causal_mask;
    MatrixB key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    const type minus_inf = -numeric_limits<float>::infinity();
};

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
