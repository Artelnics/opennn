//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
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
    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    Shape get_input_shape() const override { return {sequence_length}; }
    Shape get_output_shape() const override;

    // get_parameter_specs() and get_state_specs() are inherited from Layer and
    // auto-derived from get_operators(). EmbeddingLookup::parameter_specs()
    // pins the weight matrix to FP32 (atomicAdd<bf16> requires CC ≥ 9.0; on
    // pre-Hopper GPUs this avoids unsupported intrinsics).
    vector<Operator*> get_operators() override;

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        return {{{batch_size, sequence_length, embedding_dimension},
                 activation_dtype}}; // Output
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        return {{{batch_size, sequence_length}, activation_dtype}};
    }

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             const string & = "embedding_layer");

    void set_scale_embedding(bool enabled) { embedding_lookup.scale_embedding = enabled; }
    void set_add_positional_encoding(bool enabled) { embedding_lookup.add_positional_encoding = enabled; }

    void set_dropout_rate(const float rate)
    {
        dropout.set_rate(rate);
    }

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t index, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t index) const noexcept override;

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    enum Parameters {Weight};
    enum States {PositionalEncoding};
    enum Forward {Input, Output};
    enum Backward {OutputDelta};

    Index vocabulary_size = 0;
    Index sequence_length = 0;
    Index embedding_dimension = 0;

    EmbeddingLookup embedding_lookup;
    Dropout         dropout;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
