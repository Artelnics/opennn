//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MULTIHEADATTENTIONLAYER_H
#define MULTIHEADATTENTIONLAYER_H

#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct MultiheadAttentionLayerForwardPropagation;
struct MultiheadAttentionLayerBackPropagation;
struct MultiheadAttentionLayerBackPropagationLM;

#ifdef OPENNN_CUDA
    struct MultiheadAttentionLayerForwardPropagationCuda;
    struct MultiheadAttentionLayerBackPropagationCuda;
#endif

class MultiheadAttentionLayer : public Layer
{

public:

    // Constructors

    explicit MultiheadAttentionLayer();

    explicit MultiheadAttentionLayer(const Index&,
                                     const Index&,
                                     const Index&,
                                     const Index&,
                                     const bool & = false);

    // Get

    bool is_empty() const;

    Index get_input_size() const;
    Index get_context_size() const;
    Index get_depth() const;
    Index get_heads_number() const;
    Index get_weights_depth() const;

    dimensions get_output_dimensions() const final;

    Tensor<type, 3> get_query_weights() const;
    Tensor<type, 2> get_query_biases() const;

    Tensor<type, 3> get_key_weights() const;
    Tensor<type, 2> get_key_biases() const;

    Tensor<type, 3> get_value_weights() const;
    Tensor<type, 2> get_value_biases() const;

    Tensor<type, 3> get_projection_weights() const;
    Tensor<type, 1> get_projection_biases() const;

    Index get_parameters_number() const final;
    Tensor<type, 1> get_parameters() const final;

    // Display messages

    const bool& get_display() const;

    // Set

    void set();
    void set(const Index&, const Index&, const Index&, const Index&);

    void set_default();

    // Architecture

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;

    void set_input_size(const Index&);
    void set_context_size(const Index&);
    void set_depth(const Index&);
    void set_heads_number(const Index&);

    void set_weights();
    void set_parameters_random() final;
    void set_parameters_glorot();
    void set_parameters_constant(const type&) final;

    void set_dropout_rate(const type&);
    void set_causal_mask(const bool&);

    // Display messages

    void set_display(const bool&);

    void build_causal_mask();
    void apply_causal_mask(Tensor<type, 4>&) const;

    // Linear transformation & projection

    void calculate_transformation(const Tensor<type, 3>&, Tensor<type, 4>&, const Tensor<type, 3>&, const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void calculate_output_projection(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 3>&) const;

    // Attention computation

    void compute_attention_scores(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;

    void compute_attention_outputs(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void dropout(Tensor<type, 4>&) const;

    // Multihead Attention layer outputs

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) final;

    // Gradient

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const final;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         const Index&,
                         Tensor<type, 1>&) const final;

    // Serialization

    // @todo

    void from_XML(const tinyxml2::XMLDocument&) final;
    void to_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/multihead_attention_layer_cuda.h"
    #endif

protected:

    // MEMBERS

    Index input_size;

    Index context_size;

    Index depth;

    Index heads_number;

    Index hidden_depth;

    type scaling_factor = 1;

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

    bool display = true;

    const Eigen::array<Index, 1> projection_sum_index = Eigen::array<Index, 1>({ 3 });
    const Eigen::array<Index, 2> biases_derivatives_sum_indices = Eigen::array<Index, 2>({ 0, 2 });
    const Eigen::array<Index, 2> projection_biases_derivatives_sum_indices = Eigen::array<Index, 2>({ 0, 1 });

    const Eigen::array<IndexPair<Index>, 2> projection_weights_derivatives_contraction_indices = { IndexPair<Index>(2, 0), IndexPair<Index>(0, 1) };
    const Eigen::array<IndexPair<Index>, 2> transformation_weights_derivatives_contraction_indices = { IndexPair<Index>(1, 0), IndexPair<Index>(0, 2) };
};

struct MultiheadAttentionLayerForwardPropagation : LayerForwardPropagation
{
        
    explicit MultiheadAttentionLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit MultiheadAttentionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }
                
    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void print() const
    {
//            cout << "Attention scores:" << endl;
//            cout << attention_scores.dimensions() << endl;


//            cout << "Outputs dimensions:" << endl;
//            cout << output_dimensions << endl;

//            cout << "Outputs:" << endl;
//            cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;

//            cout << "Attention scores:" << endl;
//            cout << attention_scores << endl;
    }

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


struct MultiheadAttentionLayerBackPropagation : LayerBackPropagation
{
        

    explicit MultiheadAttentionLayerBackPropagation() : LayerBackPropagation()
    {

    }


    explicit MultiheadAttentionLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void print() const
    {
    }

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
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
