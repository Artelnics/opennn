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

    vector<TensorView*> get_parameter_views() override;

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type);

    void apply_causal_mask(Tensor4&) const;

    void forward_propagate(unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    void back_propagate(unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void calculate_projection(const TensorMap3& inputs,
                              const TensorView& weights,
                              const TensorView& biases,
                              Index sequence_length,
                              Index batch_size,
                              Tensor4& output) const;


    void calculate_projection_gradient(const Tensor4& d_head,
                                       const TensorMap3& input,
                                       const TensorView& weights,
                                       VectorMap& d_bias,
                                       MatrixMap& d_weights,
                                       TensorMap3& d_input,
                                       Index batch_size,
                                       bool accumulate) const;

    void print() const override;

    void to_XML(XMLPrinter&) const override;
    void from_XML(const XMLDocument&) override;

    void apply_key_padding_mask(const TensorMap3&, Tensor4&) const;

#ifdef OPENNN_CUDA

    public:

<<<<<<< HEAD
        void forward_propagate(unique_ptr<LayerForwardPropagationCuda>&, bool) override;
=======
        void forward_propagate(unique_ptr<LayerForwardPropagationCuda>&,
                               bool) override;
>>>>>>> 5d737e55da14848ee8ba7ff9b116f94be9e7a84f

        void back_propagate(unique_ptr<LayerForwardPropagationCuda>&,
                            unique_ptr<LayerBackPropagationCuda>&) const override;

        vector<TensorViewCuda*> get_parameter_views_device() override;

        void linear_projection_cuda(const float*, const float*, const float*,
                                    cudnnTensorDescriptor_t, float*, cudnnTensorDescriptor_t,
                                    int, int, int) const;

    protected:

        TensorViewCuda query_weights_device;
        TensorViewCuda query_biases_device;
        TensorViewCuda key_weights_device;
        TensorViewCuda key_biases_device;
        TensorViewCuda value_weights_device;
        TensorViewCuda value_biases_device;
        TensorViewCuda projection_weights_device;
        TensorViewCuda projection_biases_device;
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

    MatrixR causal_mask;
    MatrixB key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    const type minus_inf = -numeric_limits<float>::infinity();
};


struct MultiHeadAttentionForwardPropagation final : LayerForwardPropagation
{
    MultiHeadAttentionForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;

    Tensor4 query;
    Tensor4 key;
    Tensor4 value;

    Tensor4 attention_weights;

    Tensor3 concatenated_attention_outputs;
};


struct MultiHeadAttentionBackPropagation final : LayerBackPropagation
{
    MultiHeadAttentionBackPropagation(const Index = 0, Layer* = nullptr);

    vector<TensorView*> get_gradient_views() override;

    void initialize() override;

    void print() const override;

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


#ifdef OPENNN_CUDA

struct MultiHeadAttentionForwardPropagationCuda : public LayerForwardPropagationCuda
{
    MultiHeadAttentionForwardPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override {}

    void free() override {}

    TensorCuda query, key, value;                       // [B*S, E]
    TensorCuda attention_weights;                       // Scores [B*H*Sq, Sk]
    //TensorCuda attention_outputs;                       // [B*H*Sq, D]
    TensorCuda concatenated_attention_outputs;          // [B*Sq, E]

    TensorCuda query_transposed, key_transposed, value_transposed;
    TensorCuda attention_outputs_transposed;
    TensorCuda attention_probabilities;
};


struct MultiHeadAttentionBackPropagationCuda : public LayerBackPropagationCuda
{
    MultiHeadAttentionBackPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorViewCuda*> get_gradient_views() override;

    void print() const override {}

    void free() override {}

    TensorViewCuda query_weight_gradients, key_weight_gradients, value_weight_gradients, projection_weight_gradients;
    TensorViewCuda query_bias_gradients, key_bias_gradients, value_bias_gradients, projection_bias_gradients;

    TensorCuda query_gradients, key_gradients, value_gradients;
    TensorCuda attention_weight_gradients;              // dP
    //TensorCuda attention_output_gradients;              // dO
    TensorCuda concatenated_attention_output_gradients; // dY_proj
    TensorCuda softmax_gradients;                       // dS

    TensorCuda query_gradients_transposed, key_gradients_transposed, value_gradients_transposed;
    TensorCuda attention_output_gradients_transposed;
    TensorCuda query_input_gradients, source_input_gradients; // dX_q, dX_s
    TensorCuda ones;
};

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
