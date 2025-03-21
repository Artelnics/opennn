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

#ifdef OPENNN_CUDA
    struct MultiheadAttentionForwardPropagationCuda;
    struct MultiheadAttentionBackPropagationCuda;
#endif

class MultiHeadAttention : public Layer
{

public:

    MultiHeadAttention(const Index& = 0,
                       const Index& = 0,
                       const Index& = 0,
                       const Index& = 0,
                       const bool& = false,
                       const string& = "multihead_attention_layer");

    Index get_input_size() const;
    Index get_context_size() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;
    Index get_weights_depth() const;

    type get_scaling_factor() const;

    Index get_hidden_depth() const;

    dimensions get_input_dimensions() const override;

    dimensions get_output_dimensions() const override;

    Index get_parameters_number() const override;
    Tensor<type, 1> get_parameters() const override;

    void set(const Index& = 0, 
             const Index& = 0, 
             const Index& = 0, 
             const Index& = 0, 
             const string& = "multihead_attention_layer");

    void set_parameters(const Tensor<type, 1>&, Index&) override;

    void set_parameters_random() override;
    void set_parameters_glorot();
    void set_parameters_constant(const type&) override;

    void set_dropout_rate(const type&);
    void set_use_causal_mask(const bool&);

    void apply_causal_mask(Tensor<type, 4>&) const;

    void calculate_transformation(const Tensor<type, 3>&, Tensor<type, 4>&, const Tensor<type, 3>&, const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void calculate_output_projection(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 3>&) const;

    void calculate_attention_scores(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    // void calculate_attention_scores(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void calculate_attention_outputs(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void dropout(Tensor<type, 4>&) const;

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

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/multihead_attention_layer_cuda.h"
    #endif

    Tensor<float, 4> compute_query(
        const Eigen::Tensor<float, 3>& input,             // (batch, seq_len, embed_dim)
        const Eigen::Tensor<float, 3>& query_weights)     // (embed_dim, head_dim, num_heads)
    {
        const int batch_size = input.dimension(0);
        const int seq_len = input.dimension(1);
        const int embed_dim = input.dimension(2);

        const int head_dim = query_weights.dimension(1);
        const int num_heads = query_weights.dimension(2);

        // Verify embedding dimension matches
        assert(embed_dim == query_weights.dimension(0));

        // Tensor contraction dimensions:
        // input: (batch_size, seq_len, embed_dim)
        // query_weights: (embed_dim, head_dim, num_heads)
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(2, 0) };

        // Perform contraction to get (batch_size, seq_len, head_dim, num_heads)
        Tensor<float, 4> query(batch_size, seq_len, head_dim, num_heads);
        query = input.contract(query_weights, contract_dims)
            .reshape(Eigen::array<int, 4>{batch_size, seq_len, head_dim, num_heads});

        // Optionally, reorder dimensions to (batch_size, seq_len, num_heads, head_dim) for consistency
        Eigen::array<int, 4> shuffle_dims = { 0, 1, 3, 2 };
        Eigen::Tensor<float, 4> output = query.shuffle(shuffle_dims);

        return output; // (batch_size, seq_len, num_heads, head_dim)
    }

private:

    Index query_sequence_length = 0;

    Index source_sequence_length = 0;

    Tensor<type, 3> query_weights;
    Tensor<type, 2> query_biases;

    Tensor<type, 3> key_weights;
    Tensor<type, 2> key_biases;

    Tensor<type, 3> value_weights;
    Tensor<type, 2> value_biases;

    Tensor<type, 3> projection_weights;
    Tensor<type, 1> projection_biases;

    bool use_causal_mask = false;

    Tensor<type, 2> causal_mask;

    type dropout_rate = type(0);

    const Eigen::array<Index, 1> projection_sum_index = { 3 };
    const Eigen::array<Index, 2> biases_derivatives_sum_indices = { 0, 2 };
    const Eigen::array<Index, 2> projection_biases_derivatives_sum_indices = { 0, 1 };

    const Eigen::array<IndexPair<Index>, 2> projection_weights_derivatives_contraction_indices = { IndexPair<Index>(2, 0), IndexPair<Index>(0, 1) };
    const Eigen::array<IndexPair<Index>, 2> transformation_weights_derivatives_contraction_indices = { IndexPair<Index>(1, 0), IndexPair<Index>(0, 2) };
};


struct MultiheadAttentionForwardPropagation : LayerForwardPropagation
{
    MultiheadAttentionForwardPropagation(const Index& new_batch_samples_number = 0,
                                              Layer* new_layer = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 4> query;
    Tensor<type, 4> key;
    Tensor<type, 4> value;

    Tensor<type, 2> sample_matrix;

    Tensor<type, 4> attention_scores;
    Tensor<type, 4> attention_weights;
    Tensor<type, 4> attention_outputs;

    Tensor<type, 4> projection_outputs;
    Tensor<type, 3> outputs;
};


struct MultiheadAttentionBackPropagation : LayerBackPropagation
{

    MultiheadAttentionBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 4> error_attention_scores_derivatives;
    Tensor<type, 4> error_attention_weights_derivatives;
    Tensor<type, 4> error_attention_output_derivatives;

    Tensor<type, 2> sample_deltas;

    Tensor<type, 4> error_query_derivatives;
    Tensor<type, 4> error_key_derivatives;
    Tensor<type, 4> error_value_derivatives;

    Tensor<type, 3> query_weights_derivatives;
    Tensor<type, 3> key_weights_derivatives;
    Tensor<type, 3> value_weights_derivatives;

    Tensor<type, 3> projection_weights_derivatives;

    Tensor<type, 2> query_biases_derivatives;
    Tensor<type, 2> key_biases_derivatives;
    Tensor<type, 2> value_biases_derivatives;
    Tensor<type, 1> projection_biases_derivatives;

    Tensor<type, 1> aux_rows;

    Tensor<type, 3> input_derivatives;
    Tensor<type, 3> context_derivatives;
};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/multihead_attention_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/multihead_attention_layer_back_propagation_cuda.h"
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
