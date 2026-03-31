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

    vector<Shape> get_forward_shapes() const override
    {
        /*
    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index head_dimension = multihead_attention_layer->get_head_dimension();

    query.resize(batch_size, heads_number, query_sequence_length, head_dimension);
    key.resize(batch_size, heads_number, source_sequence_length, head_dimension);
    value.resize(batch_size, heads_number, source_sequence_length, head_dimension);

    attention_weights.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

    // @todo can we remove concatenated_attention_outputs and assign to outputs?

    concatenated_attention_outputs.resize(batch_size, query_sequence_length, embedding_dimension);

    outputs.shape = {batch_size, query_sequence_length, embedding_dimension};
*/
        return {};
    }

    vector<Shape> get_backward_shapes() const override
    {
    const Index query_sequence_length = get_query_sequence_length();
    const Index source_sequence_length = get_source_sequence_length();
    const Index embedding_dimension = get_embedding_dimension();
    const Index heads_number = get_heads_number();
    const Index head_dimension = get_head_dimension();
    /*
    query_weight_gradients.shape = {embedding_dimension, embedding_dimension};
    key_weight_gradients.shape = {embedding_dimension, embedding_dimension};
    value_weight_gradients.shape = {embedding_dimension, embedding_dimension};
    projection_weight_gradients.shape = {embedding_dimension, embedding_dimension};

    query_bias_gradients.shape = {embedding_dimension};
    key_bias_gradients.shape = {embedding_dimension};
    value_bias_gradients.shape = {embedding_dimension};
    projection_bias_gradients.shape = {embedding_dimension};

    input_gradients = {{nullptr, {batch_size, query_sequence_length, embedding_dimension}},
                       {nullptr, {batch_size, source_sequence_length, embedding_dimension}}};

    // Auxiliar

    attention_weight_gradients.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

    concatenated_attention_output_gradients.resize(batch_size, query_sequence_length, embedding_dimension);

    query_gradients.resize(batch_size, heads_number, query_sequence_length, head_dimension);
    key_gradients.resize(batch_size, heads_number, source_sequence_length, head_dimension);
    value_gradients.resize(batch_size, heads_number, source_sequence_length, head_dimension);
*/
        return {};
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

    enum Parameters {QueryWeights, QueryBiases, KeyWeights, KeyBiases, ValueWeights, ValueBiases};
    enum Forward {Inputs, Query, Key, AttentionWeights, ConcatenatedAttentionOutputs, Value, Outputs};

    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    TensorView query_weights;
    TensorView query_biases;

    TensorView key_weights;
    TensorView key_biases;

    TensorView value_weights;
    TensorView value_biases;

    TensorView projection_weights;
    TensorView projection_biases;

    bool use_causal_mask = false;

    MatrixR causal_mask;
    MatrixB key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    const type minus_inf = -numeric_limits<float>::infinity();
};



struct MultiHeadAttentionBackPropagation final : LayerBackPropagation
{
    Tensor4 attention_weight_gradients;
    Tensor3 concatenated_attention_output_gradients;

    Tensor4 query_gradients;
    Tensor4 key_gradients;
    Tensor4 value_gradients;

    TensorView query_weight_gradients;
    TensorView key_weight_gradients;
    TensorView value_weight_gradients;
    TensorView projection_weight_gradients;

    TensorView query_bias_gradients;
    TensorView key_bias_gradients;
    TensorView value_bias_gradients;
    TensorView projection_bias_gradients;
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
