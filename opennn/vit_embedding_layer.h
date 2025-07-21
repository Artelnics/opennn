//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I T  E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef VIT_EMBEDDINGLAYER_H
#define VIT_EMBEDDINGLAYER_H

#include "layer.h"

namespace opennn
{

#ifdef OPENNN_CUDA
    struct EmbeddingLayerForwardPropagationCuda;
    struct EmbeddingLayerBackPropagationCuda;
#endif


    class VitEmbeddingLayer : public Layer
    {

    public:

        VitEmbeddingLayer(const Index & = 32,
            const Index & = 32,
            const Index & = 3,
            const Index & = 4,
            const Index & = 128,
            const bool& = true,
            const string & = "vit_embedding_layer");

        Index get_patch_number() const;
        Index get_input_dimension() const;
        Index get_image_height() const;
        Index get_image_width() const;
        Index get_image_channels() const;
        Index get_patch_size() const;
        Index get_embedding_dimension() const;
        bool get_positional_encoding() const;

        dimensions get_input_dimensions() const override;
        dimensions get_output_dimensions() const override;

        Index get_parameters_number() const override;
        Tensor<type, 1> get_parameters() const override;

        void set(const Index & = 0,
            const Index & = 0,
            const Index & = 0,
            const Index & = 0,
            const Index & = 0,
            const bool& = false,
            const string & = "embedding_layer");

        void set_image_height(const Index&);
        void set_image_width(const Index&);
        void set_image_channels(const Index&);
        void set_patch_size(const Index&);
        void set_embedding_size(const Index&);
        void set_positional_encoding(const bool&);

        void set_dropout_rate(const type&);

        void set_embedding_weights();

        void set_parameters(const Tensor<type, 1>&, const Index& index = 0) override;
        void set_parameters_random() override;
        void set_parameters_glorot();
        void set_parameters_constant(const type&) override;

        //void dropout(Tensor<type, 3>&) const;
        void dropout(Tensor<type, 3>&, Tensor<type, 3>&);

        void lookup_embedding(const Tensor<type, 3>&, Tensor<type, 3>&);

        void forward_propagate(const vector<pair<type*, dimensions>>&,
            unique_ptr<LayerForwardPropagation>&,
            const bool&) override;

        void back_propagate(const vector<pair<type*, dimensions>>&,
            const vector<pair<type*, dimensions>>&,
            unique_ptr<LayerForwardPropagation>&,
            unique_ptr<LayerBackPropagation>&) const override;
        
        void add_deltas(const vector<pair<type*, dimensions>>&) const;

        void insert_gradient(unique_ptr<LayerBackPropagation>&,
            const Index&,
            Tensor<type, 1>&) const override;

        void from_XML(const XMLDocument&) override;
        void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/embedding_layer_cuda.h"
#endif

    private:

        Index image_height = 32;

        Index image_width = 32;

        Index image_channels = 3;

        Index patch_size;

        Tensor<type, 2> embedding_weights;

        Tensor<type, 3> output_with_class_token;

        type dropout_rate = 0.1;

        bool positional_encoding;

        const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 1) };

        const Eigen::array<IndexPair<Index>, 1> product_dims = { IndexPair<Index>(2, 0) };

        const Eigen::array<IndexPair<Index>, 2> embedding_weights_derivatives_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
    };


    struct VitEmbeddingLayerForwardPropagation : LayerForwardPropagation
    {
        VitEmbeddingLayerForwardPropagation(const Index & = 0, Layer* = nullptr);

        pair<type*, dimensions> get_outputs_pair() const override;

        void set(const Index & = 0, Layer* = nullptr);

        void print() const override;

        void build_positional_encoding_matrix();

        bool built_positional_encoding_matrix = false;

        Tensor<type, 2> positional_encoding;

        Tensor<type, 3> outputs;

        Tensor<type, 3> dropout_mask;
    };


    struct VitEmbeddingLayerBackPropagation : LayerBackPropagation
    {
        VitEmbeddingLayerBackPropagation(const Index & = 0, Layer* = nullptr);

        vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

        void set(const Index & = 0, Layer* = nullptr);

        void print() const override;

        Tensor<type, 2> embedding_weights_derivatives;
    };

#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/embedding_layer_forward_propagation_cuda.h"
#include "../../opennn_cuda/opennn_cuda/embedding_layer_back_propagation_cuda.h"
#endif

}

#endif // VIT_EMBEDDING_LAYER_H


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
