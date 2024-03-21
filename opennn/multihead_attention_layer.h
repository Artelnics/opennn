//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MULTIHEADATTENTIONLAYER_H
#define MULTIHEADATTENTIONLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct MultiheadAttentionLayerForwardPropagation;
struct MultiheadAttentionLayerBackPropagation;
struct MultiheadAttentionLayerBackPropagationLM;

struct PerceptronLayer3DForwardPropagation;
struct PerceptronLayer3DBackPropagation;


/// This class represents a layer of Multihead Attention.

/// MultiheadAttentionLayer has 2 types of input: Context and Input. Output has the shape of Input
///
/// Layers of Multihead Attention will be used to construct Transformer models .

class MultiheadAttentionLayer : public Layer
{
/// @todo get_parameters() and set_parameters()
public:

    // Constructors

    explicit MultiheadAttentionLayer();

    explicit MultiheadAttentionLayer(const Index&, /// Input size
                                     const Index&, /// Context size
                                     const Index&, /// Embedding depth
                                     const Index&, /// Number of attention heads
                                     const bool & = false); /// Apply causal mask

    // Get methods

    bool is_empty() const;

    Index get_input_size() const;
    Index get_context_size() const;
    Index get_depth() const;
    Index get_heads_number() const;
    Index get_weights_depth() const;

    Tensor<type, 3> get_query_weights() const;
    Tensor<type, 3> get_key_weights() const;
    Tensor<type, 3> get_value_weights() const;

    Tensor<type, 3> get_projection_weights() const;
    Tensor<type, 1> get_projection_biases() const;

    Index get_parameters_number() const final;
    Tensor<type, 1> get_parameters() const final;

    // Display messages

    const bool& get_display() const;

    // Set methods

    void set();
    void set(const Index&, const Index&, const Index&, const Index&);

    void set_default();
    void set_name(const string&);

    // Architecture

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;

    void set_input_size(const Index&);
    void set_context_size(const Index&);
    void set_depth(const Index&);
    void set_heads_number(const Index&);

    void set_weights();
    void set_parameters_random() final;
    void set_parameters_constant(const type&) final;

    void set_dropout_rate(const type&);
    void set_causal_mask(const bool&);

    // Display messages

    void set_display(const bool&);

    void apply_causal_mask(Tensor<type, 4>&) const;

    // Linear transformation & projection

    void calculate_transformation(const Tensor<type, 3>&, Tensor<type, 4>&, const Tensor<type, 3>&) const;

    void calculate_output_projection(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 3>&) const;

    // Attention computation

    void compute_attention_scores(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;

    void compute_attention_outputs(const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void dropout(Tensor<type, 4>&) const;

    // Multihead Attention layer outputs

    void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                           LayerForwardPropagation*,
                           const bool&) final;

    // Delta methods

    void calculate_hidden_delta(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerForwardPropagation*,
                                LayerBackPropagation*) const final;

    void calculate_hidden_delta(PerceptronLayer3DForwardPropagation*,
                                PerceptronLayer3DBackPropagation*,
                                MultiheadAttentionLayerBackPropagation*) const;

    void calculate_hidden_delta(MultiheadAttentionLayerForwardPropagation*,
                                MultiheadAttentionLayerBackPropagation*,
                                MultiheadAttentionLayerBackPropagation*) const;

    // Gradient methods

    void calculate_error_gradient(const Tensor<pair<type*, dimensions>, 1>&,
                                  LayerForwardPropagation*,
                                  LayerBackPropagation*) const final;

    void insert_gradient(LayerBackPropagation*, const Index&, Tensor<type, 1>&) const final;

    // Serialization methods

    /// @todo

    //void from_XML(const tinyxml2::XMLDocument&) final;
    //void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    // MEMBERS

    /// Input size

    Index input_size;

    /// Context size

    Index context_size;

    /// Embedding depth

    Index depth;

    /// Number of attention heads

    Index heads_number;

    /// Depth used in attention computation

    Index weights_depth;

    // Scaling factor used for attention computation in each head

    type scaling_factor = 1;

    /// Linear transformation weights

    Tensor<type, 3> query_weights;
    Tensor<type, 3> key_weights;
    Tensor<type, 3> value_weights;

    /// Linear projection weights

    Tensor<type, 3> projection_weights;
    Tensor<type, 1> projection_biases;

    /// Dropout rate

    type dropout_rate = type(0);

    /// Apply causal mask or not

    bool causal_mask = false;

    /// Display messages to screen.

    bool display = true;

    // Operation indices

    const Eigen::array<Index, 1> projection_sum_index = Eigen::array<Index, 1>({ 3 });
    const Eigen::array<Index, 2> projection_biases_derivatives_sum_indices = Eigen::array<Index, 2>({ 0, 1 });

    const Eigen::array<IndexPair<Index>, 2> projection_weights_derivatives_contraction_indices = { IndexPair<Index>(2, 0), IndexPair<Index>(0, 1) };
    const Eigen::array<IndexPair<Index>, 1> attention_output_derivatives_contraction_indices = { IndexPair<Index>(2, 1) };
    const Eigen::array<IndexPair<Index>, 2> transformation_weights_derivatives_contraction_indices = { IndexPair<Index>(1, 0), IndexPair<Index>(0, 2) };
};

    struct MultiheadAttentionLayerForwardPropagation : LayerForwardPropagation
    {
        // Default constructor

        explicit MultiheadAttentionLayerForwardPropagation() : LayerForwardPropagation()
        {
        }

        virtual ~MultiheadAttentionLayerForwardPropagation()
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
//            cout << outputs_dimensions << endl;

//            cout << "Outputs:" << endl;
//            cout << TensorMap<Tensor<type,3>>(outputs_data, outputs_dimensions(0), outputs_dimensions(1), outputs_dimensions(2)) << endl;

//            cout << "Attention scores:" << endl;
//            cout << attention_scores << endl;
        }

        Tensor<type, 4> query;
        Tensor<type, 4> key;
        Tensor<type, 4> value;

        Tensor<type, 4> attention_scores;
        Tensor<type, 4> softmax_attention_scores;
        Tensor<type, 4> attention_outputs;

        Tensor<type, 4> projection_outputs;
        Tensor<type, 3> outputs;
    };


    struct MultiheadAttentionLayerBackPropagation : LayerBackPropagation
    {
        // Default constructor

        explicit MultiheadAttentionLayerBackPropagation() : LayerBackPropagation()
        {

        }


        explicit MultiheadAttentionLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerBackPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~MultiheadAttentionLayerBackPropagation()
        {
        }


        void set(const Index& new_batch_samples_number, Layer* new_layer) final;

        void print() const
        {
            cout << "Deltas:" << endl;
            //cout << deltas << endl;
        }

        Tensor<type, 3> deltas;

        Tensor<type, 4> error_attention_scores_derivatives;
        Tensor<type, 4> error_softmax_attention_scores_derivatives;
        Tensor<type, 4> error_attention_output_derivatives;

        Tensor<type, 4> error_query_derivatives;
        Tensor<type, 4> error_key_derivatives;
        Tensor<type, 4> error_value_derivatives;

        Tensor<type, 3> error_input_derivatives;
        Tensor<type, 3> error_context_derivatives;

        Tensor<type, 3> query_weights_derivatives;
        Tensor<type, 3> key_weights_derivatives;
        Tensor<type, 3> value_weights_derivatives;

        Tensor<type, 3> projection_weights_derivatives;
        Tensor<type, 1> projection_biases_derivatives;
    };

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