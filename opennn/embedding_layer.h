//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef EMBEDDINGLAYER_H
#define EMBEDDINGLAYER_H

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct EmbeddingLayerForwardPropagation;
struct EmbeddingLayerBackPropagation;
struct EmbeddingLayerBackPropagationLM;

#ifdef OPENNN_CUDA
struct EmbeddingLayerForwardPropagationCuda;
struct EmbeddingLayerBackPropagationCuda;
#endif


class EmbeddingLayer : public Layer
{

public:

    // Constructors

    explicit EmbeddingLayer();

    explicit EmbeddingLayer(const Index&,
                            const Index&,
                            const Index&,
                            const bool& = false);
    // Get

    Index get_input_dimension() const;
    Index get_inputs_number() const;
    Index get_depth() const;
    bool get_positional_encoding() const;

    dimensions get_input_dimensions() const;
    dimensions get_output_dimensions() const final;

    Tensor<type, 2> get_embedding_weights() const;

    Index get_parameters_number() const final;
    Tensor<type, 1> get_parameters() const final;
    Index get_neurons_number() const final;

    // Display messages

    const bool& get_display() const;

    // Set

    void set();
    void set(const Index&, const Index&, const Index&, const bool& = false);

    void set_default();

    // Architecture

    void set_input_dimensions(const Index&);
    void set_inputs_number(const Index&);
    void set_depth(const Index&);
    void set_dropout_rate(const type&);

    void set_embedding_weights();

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;
    void set_parameters_random() final;
    void set_parameters_constant(const type&) final;

    // Display messages

    void set_display(const bool&);

    // Dropout

    void dropout(Tensor<type, 3>&) const;

    // Embedding lookup

    void lookup_embedding(const Tensor<type, 2>&, Tensor<type, 3>&);

    // Embedding layer outputs

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>,
                           const bool&) final;

    // Gradient

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>,
                        unique_ptr<LayerBackPropagation>) const final;

    void add_deltas(const vector<pair<type*, dimensions>>&) const;

    void insert_gradient(unique_ptr<LayerBackPropagation> back_propagation, 
                         const Index& index, 
                         Tensor<type, 1>& gradient) const;

    // Serialization

    // @todo

    void from_XML(const tinyxml2::XMLDocument&) final;
    void to_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/embedding_layer_cuda.h"
    #endif

protected:

    // MEMBERS

    Index input_dimensions;

    Index inputs_number;

    Index depth;

    type dropout_rate;

    Tensor<type, 2> embedding_weights;

    bool positional_encoding;

    bool display = true;

    const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 1) };
    };


struct EmbeddingLayerForwardPropagation : LayerForwardPropagation
    {
        

        explicit EmbeddingLayerForwardPropagation() : LayerForwardPropagation()
        {
        }


        explicit EmbeddingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerForwardPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~EmbeddingLayerForwardPropagation()
        {
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

        void build_positional_encoding_matrix();

        // Struct members

        bool built_positional_encoding_matrix = false;

        Tensor<type, 2> positional_encoding;

        Tensor<type, 3> outputs;
    };


struct EmbeddingLayerBackPropagation : LayerBackPropagation
    {
        

        explicit EmbeddingLayerBackPropagation() : LayerBackPropagation()
        {

        }


        explicit EmbeddingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerBackPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~EmbeddingLayerBackPropagation()
        {
        }

        vector<pair<type*, dimensions>> get_input_derivative_pairs() const
        {
            return vector<pair<type*, dimensions>>();
        }

        void set(const Index& new_batch_samples_number, Layer* new_layer) final;

        void print() const
        {
        }

        Tensor<type, 2> sample_deltas;
        Tensor<type, 2> embedding_weights_derivatives;
    };


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/embedding_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/embedding_layer_back_propagation_cuda.h"
#endif


}

#endif // EMBEDDING_LAYER_H


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
