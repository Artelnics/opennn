//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MULTIHEADATTENTIONLAYER_H
#define MULTIHEADATTENTIONLAYER_H

#include "layer.h"

namespace opennn
{

class MultiHeadAttention : public Layer
{

public:

    MultiHeadAttention(const dimensions& = dimensions({0,0}),
                       const Index& = 0,
                       const string& = string());

    MultiHeadAttention(const dimensions&,
                       const dimensions&,
                       const Index& = 0,
                       const string& = string());

    Index get_query_sequence_length() const;
    Index get_source_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;
    Index get_hidden_depth() const;

    type get_scaling_factor() const;

    dimensions get_input_dimensions() const override;

    dimensions get_output_dimensions() const override;

    vector<pair<type*, Index>> get_parameter_pairs() const override
    {
        return {
            {(type*)query_weights.data(), query_weights.size()},
            {(type*)query_biases.data(), query_biases.size()},
            {(type*)key_weights.data(), key_weights.size()},
            {(type*)key_biases.data(), key_biases.size()},
            {(type*)value_weights.data(), value_weights.size()},
            {(type*)value_biases.data(), value_biases.size()},
            {(type*)projection_weights.data(), projection_weights.size()},
            {(type*)projection_biases.data(), projection_biases.size()}
        };
    }

    void set(const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const bool& = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type&);

    void apply_causal_mask(Tensor<type, 4>&) const;

    void calculate_attention_weights(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void calculate_attention_outputs(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void concatenate_heads(const Tensor<type, 4>&, Tensor<type, 3>&) const;

    void calculate_output_projection(const Tensor<type, 3>&, Tensor<type, 3>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         Index&,
                         Tensor<type, 1>&) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
       // @todo
    #endif

    void apply_key_padding_mask(const Tensor<bool, 2>&,Tensor<type, 4>&) const;

private:

    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    Index heads_number = 0;
    Index embedding_dimension = 0;

    Tensor<type, 3> query_weights;
    Tensor<type, 2> query_biases;

    Tensor<type, 3> key_weights;
    Tensor<type, 2> key_biases;

    Tensor<type, 3> value_weights;
    Tensor<type, 2> value_biases;

    Tensor<type, 2> projection_weights;
    Tensor<type, 1> projection_biases;

    bool use_causal_mask = false;

    Tensor<type, 2> causal_mask;
    Tensor<bool,2> key_mask;            //Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    const type minus_inf = -numeric_limits<float>::infinity();
};


struct MultiheadAttentionForwardPropagation : LayerForwardPropagation
{
    MultiheadAttentionForwardPropagation(const Index& new_batch_size = 0,
                                         Layer* new_layer = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 4> query;
    Tensor<type, 4> key;
    Tensor<type, 4> value;

    Tensor<type, 4> attention_weights;
    Tensor<type, 4> attention_outputs;

    Tensor<type, 3> concatenated_attention_outputs;

    Tensor<type, 4> projection_outputs;

    Tensor<type, 3> outputs;

    Tensor<type, 2> sample_matrix;
};


struct MultiheadAttentionBackPropagation : LayerBackPropagation
{
    MultiheadAttentionBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    vector<pair<type*, Index>> get_parameter_delta_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 4> attention_weight_deltas_xxx;
    Tensor<type, 4> attention_output_deltas;
    Tensor<type, 3> concatenated_attention_output_deltas;

    Tensor<type, 4> query_deltas;
    Tensor<type, 4> key_deltas;
    Tensor<type, 4> value_deltas;

    Tensor<type, 3> query_weight_deltas;
    Tensor<type, 3> key_weight_deltas;
    Tensor<type, 3> value_weight_deltas;

    Tensor<type, 2> projection_weight_deltas;

    Tensor<type, 2> query_bias_deltas;
    Tensor<type, 2> key_bias_deltas;
    Tensor<type, 2> value_bias_deltas;
    Tensor<type, 1> projection_bias_deltas;

    Tensor<type, 1> aux_rows;

    Tensor<type, 3> input_query_deltas;
    Tensor<type, 3> input_source_deltas;
};

#ifdef OPENNN_CUDA
    // @todo
#endif

}

#endif // MULTIHEAD_ATTENTION_LAYER_H


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
