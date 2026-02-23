//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "layer.h"

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

    vector<TensorView*> get_parameter_views() override;

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type);

    void apply_causal_mask(Tensor4&) const;

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void calculate_projection(const TensorMap3 inputs,
                              const TensorView& weights,
                              const TensorView& biases,
                              Index sequence_length,
                              Index batch_size,
                              Tensor4& output) const
    {
        const Index embed_dim = get_embedding_dimension();
        const Index h_dim = get_head_dimension();

        const MatrixMap W = matrix_map(weights);
        const VectorMap B = vector_map(biases);

        output.device(get_device()) =
            (inputs.reshape(array_2(batch_size * sequence_length, embed_dim))
                 .contract(W, axes(1, 0))
             + B.reshape(array_2(1, embed_dim))
                   .broadcast(array_2(batch_size * sequence_length, 1)))
                .reshape(array_4(batch_size, sequence_length, heads_number, h_dim))
                .shuffle(array_4(0, 2, 1, 3));
    }

    void calculate_projection_gradient(const Tensor4& d_head,
                                       const TensorMap3 input,
                                       const TensorView& weights,
                                       VectorMap d_bias,
                                       MatrixMap d_weights,
                                       TensorMap3 d_input,
                                       Index batch_size,
                                       bool accumulate) const
    {
        const Index embedding_dimension = get_embedding_dimension();

        // Map parameters view to Eigen TensorMap for calculation
        const MatrixMap W = matrix_map(weights);

        // Reinterpret 4D head gradients as 3D [Batch, Sequence, Embedding]
        // const_cast is required because TensorMap constructor expects non-const pointer,
        // even though we treat it as read-only here.
        TensorMap3 d_reshaped(const_cast<type*>(d_head.data()), d_input.dimensions());

        const array<Index, 2> flat_dims = {batch_size * d_input.dimension(1), embedding_dimension};

        // Calculate Gradients
        // dW = Input^T * Delta
        d_weights.device(get_device()) = input.reshape(flat_dims).contract(d_reshaped.reshape(flat_dims), axes(0, 0));

        // db = Sum(Delta)
        d_bias.device(get_device()) = d_reshaped.sum(array_2(0, 1));

        // Calculate Input Delta (Error signal to previous layer)
        // dX = Delta * W^T
        if (accumulate)
            d_input.device(get_device()) += d_reshaped.contract(W, axes(2, 1));
        else
            d_input.device(get_device()) = d_reshaped.contract(W, axes(2, 1));
    }

    void print() const override;

    void to_XML(XMLPrinter&) const override;
    void from_XML(const XMLDocument&) override;

    void apply_key_padding_mask(const MatrixB&,Tensor4&) const;

#ifdef OPENNN_CUDA
        // @todo
#endif

private:

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

    Tensor2 causal_mask;
    Tensor<bool,2> key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    const type minus_inf = -numeric_limits<float>::infinity();
};


struct MultiHeadAttentionForwardPropagation final : LayerForwardPropagation
{
    MultiHeadAttentionForwardPropagation(const Index new_batch_size = 0,
                                         Layer* new_layer = nullptr);

    void initialize() override;

    void print() const override;

    Tensor4 query;
    Tensor4 key;
    Tensor4 value;

    Tensor4 attention_weights;
    Tensor4 attention_outputs;

    Tensor3 concatenated_attention_outputs;
};


struct MultiHeadAttentionBackPropagation final : LayerBackPropagation
{
    MultiHeadAttentionBackPropagation(const Index = 0, Layer* = nullptr);

    vector<TensorView*> get_gradient_views() override;

    void initialize() override;

    void print() const override;

    Tensor4 attention_weight_gradients;
    Tensor4 attention_output_gradients;
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

    VectorR aux_rows;

//    Tensor3 input_query_gradients;
//    Tensor3 input_source_gradients;

    Tensor4 softmax_gradients;
};

#ifdef OPENNN_CUDA
    // @todo
#endif

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
