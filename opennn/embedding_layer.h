//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef EMBEDDINGLAYER_H
#define EMBEDDINGLAYER_H

#include "layer.h"

namespace opennn
{

class Embedding : public Layer
{

public:

    Embedding(const dimensions& = dimensions({0, 0}),
              const Index& = 0,
              const string& = "embedding_layer");

    Index get_vocabulary_size() const;
    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    Index get_parameters_number() const override;
    void get_parameters(Tensor<type, 1>&) const override;

    void set(const Index& = 0, 
             const Index& = 0, 
             const Index& = 0, 
             const string & = "embedding_layer");

    void set_dropout_rate(const type&);

    void set_parameters(const Tensor<type, 1>&, Index&) override;
    void set_parameters_random() override;
    

    void embedding_lookup(const Tensor<type, 2>&, Tensor<type, 3>&);
    void add_positional_encodings(Tensor<type, 3>&) const;

    bool scale_embedding = false;
    bool positional_encoding_xxx = false;

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

public:

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<float*>&,
                             const vector<float*>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

    void insert_gradient_cuda(unique_ptr<LayerBackPropagationCuda>&,
                              Index&,
                              float*) const override;

    void set_parameters_cuda(const float*, Index&);

    void copy_parameters_host();

    void copy_parameters_device();

    void allocate_parameters_device();

    void free_parameters_device();

private:

    float* weights_device = nullptr;

    float* positional_encoding_device = nullptr;

#endif

private:

    Index sequence_length = 0;

    Tensor<type, 2> weights;

    Tensor<type, 2> positional_encoding;

    type dropout_rate = type(0);
};


struct EmbeddingForwardPropagation : LayerForwardPropagation
{
    EmbeddingForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 3> outputs;
};


struct EmbeddingBackPropagation : LayerBackPropagation
{
    EmbeddingBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> weight_deltas;
};

#ifdef OPENNN_CUDA

struct EmbeddingLayerForwardPropagationCuda : public LayerForwardPropagationCuda
{
    EmbeddingLayerForwardPropagationCuda(const Index& = 0, Layer* = nullptr);

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;
};


struct EmbeddingLayerBackPropagationCuda : public LayerBackPropagationCuda
{
    EmbeddingLayerBackPropagationCuda(const Index& = 0, Layer* = nullptr);

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    type* weight_deltas_device = nullptr;
};

#endif

}

#endif // EMBEDDING_LAYER_H


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
