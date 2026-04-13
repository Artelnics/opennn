//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Embedding final : public Layer
{

public:

    Embedding(const Shape& = {0, 0},
              Index = 0,
              const string& = "embedding_layer");

    Index get_vocabulary_size() const;
    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    vector<TensorView*> get_parameter_views() override;

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             const string & = "embedding_layer");

    void set_scale_embedding(bool);
    void set_add_positional_encoding(bool);

    void set_dropout_rate(const type);

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    void back_propagate(unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

public:

    void forward_propagate(unique_ptr<LayerForwardPropagationCuda>&, bool) override;

    void back_propagate(unique_ptr<LayerForwardPropagationCuda>&,
                        unique_ptr<LayerBackPropagationCuda>&) const override;

    vector<TensorViewCuda*> get_parameter_views_device() override;

    void copy_positional_encoding_device();

private:

    TensorViewCuda weights_device;

    TensorCuda positional_encoding_device;

#endif

private:

    Index sequence_length = 0;

    TensorView weights;

    bool scale_embedding = false;
    bool add_positional_encoding = false;

    MatrixR positional_encoding;
    bool pos_encoding_synced = false;

    type dropout_rate = type(0);
};


struct EmbeddingForwardPropagation final : LayerForwardPropagation
{
    EmbeddingForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
};


struct EmbeddingBackPropagation final : LayerBackPropagation
{
    EmbeddingBackPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorView*> get_gradient_views() override;

    void print() const override;

    TensorView weight_gradients;
};

#ifdef OPENNN_CUDA

struct EmbeddingForwardPropagationCuda : public LayerForwardPropagationCuda
{
    EmbeddingForwardPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
};


struct EmbeddingBackPropagationCuda : public LayerBackPropagationCuda
{
    EmbeddingBackPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorViewCuda*> get_gradient_views() override;

    void print() const override;

    TensorViewCuda weight_gradients;
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
