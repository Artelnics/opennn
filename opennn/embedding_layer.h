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

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes() const override
    {
/*
        outputs.shape = {batch_size, sequence_length, embedding_dimension};
*/
        return {};
    }

    vector<Shape> get_backward_shapes() const override
    {
        return {};
    }

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             const string & = "embedding_layer");

    void set_scale_embedding(bool);
    void set_add_positional_encoding(bool);

    void set_dropout_rate(const type);

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t index, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t index) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

public:

    void copy_positional_encoding_device();

private:

    TensorView weights_device;

    TensorCuda positional_encoding_device;

#endif

private:

    enum Parameters {Weights};
    enum Forward {Inputs, Outputs};

    Index sequence_length = 0;
    Index embedding_dimension = 0;
    Index vocabulary_size = 0;

    bool scale_embedding = false;
    bool add_positional_encoding = false;

    MatrixR positional_encoding;
    bool pos_encoding_synced = false;

    type dropout_rate = type(0);
};

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
