//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Embedding final : public Layer
{

public:

    Embedding(const Shape& = {0, 0},
              Index = 0,
              const string& = "embedding_layer");

    Index get_vocabulary_size() const { return vocabulary_size; }
    Index get_sequence_length() const { return input_shape.empty() ? 0 : input_shape[0]; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    Shape get_output_shape() const override;

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {{batch_size, get_sequence_length(), embedding_dimension}}; // Outputs
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{batch_size, get_sequence_length()}};
    }

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             const string & = "embedding_layer");

    void set_scale_embedding(bool v) { scale_embedding = v; }
    void set_add_positional_encoding(bool v) { add_positional_encoding = v; }

    void set_dropout_rate(const type r) { dropout_rate = r; }

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t index, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t index) const noexcept override;

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

#ifdef OPENNN_WITH_CUDA

public:

    void copy_positional_encoding_device();

private:

    Memory positional_encoding_device;

#endif

private:

    enum Parameters {Weights};
    enum Forward {Inputs, Outputs};

    Index vocabulary_size = 0;
    Index embedding_dimension = 0;

    bool scale_embedding = false;
    bool add_positional_encoding = false;

    MatrixR positional_encoding;
    bool pos_encoding_synced = false;

    type embedding_scale = type(1);

    type dropout_rate = type(0);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
