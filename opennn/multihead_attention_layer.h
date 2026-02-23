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
        const Index embedding_dimension = get_embedding_dimension();
        const Index head_dimension = get_head_dimension();
        const Index heads = heads_number;
        const Index total_rows = batch_size * sequence_length;

        const MatrixMap inputs_map(inputs.data(), total_rows, embedding_dimension);
        const MatrixMap weights_map = matrix_map(weights);
        const VectorMap biases_map = vector_map(biases);

        MatrixR projected(total_rows, embedding_dimension);
        projected.noalias() = (inputs_map * weights_map).rowwise() + biases_map.transpose();

        type* output_data = output.data();

        #pragma omp parallel for collapse(2)
        for(Index b = 0; b < batch_size; ++b)
        {
            for(Index h = 0; h < heads; ++h)
            {
                type* destination_block = output_data + (b * heads + h) * (sequence_length * head_dimension);
                MatrixMap destination_map(destination_block, sequence_length, head_dimension);

                destination_map.noalias() = projected.block(b * sequence_length, h * head_dimension, sequence_length, head_dimension);
            }
        }
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
        const Index sequence_length = input.dimension(1);
        const Index embedding_dimension = get_embedding_dimension();
        const Index heads = heads_number;
        const Index head_dimension = get_head_dimension();
        const Index total_rows = batch_size * sequence_length;

        MatrixR Delta(total_rows, embedding_dimension);

        #pragma omp parallel for collapse(2)
        for(Index b = 0; b < batch_size; ++b)
        {
            for(Index h = 0; h < heads; ++h)
            {
                const type* source_data = d_head.data() + (b * heads + h) * (sequence_length * head_dimension);
                const MatrixMap source_map(const_cast<type*>(source_data), sequence_length, head_dimension);

                Delta.block(b * sequence_length, h * head_dimension, sequence_length, head_dimension).noalias() = source_map;
            }
        }

        const MatrixMap X(input.data(), total_rows, embedding_dimension);
        const MatrixMap W = matrix_map(weights);

        d_weights.noalias() = X.transpose() * Delta;

        d_bias.noalias() = Delta.colwise().sum();

        MatrixMap dX_mat(d_input.data(), total_rows, embedding_dimension);

        if(accumulate)
            dX_mat.noalias() += Delta * W.transpose();
        else
            dX_mat.noalias() = Delta * W.transpose();
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
