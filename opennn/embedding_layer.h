//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "dropout_operator.h"
#include "embedding_lookup_operator.h"

namespace opennn
{

class Embedding final : public Layer
{
public:

    Embedding(const Shape& = {0, 0},
              Index = 0,
              const string& = "embedding_layer");

    Shape get_input_shape() const noexcept override { return {sequence_length}; }
    Shape get_output_shape() const override;

    Index get_vocabulary_size() const { return vocabulary_size; }
    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    vector<TensorSpec> get_backward_specs(Index) const override { return {}; }

    void set(Index = 0,
             Index = 0,
             Index = 0,
             const string& = "embedding_layer");

    void set_scale_embedding(bool enabled) { embedding_lookup.scale_embedding = enabled; }
    void set_add_positional_encoding(bool enabled) { embedding_lookup.add_positional_encoding = enabled; }

    void set_learned_positional(bool enabled)
    {
        embedding_lookup.add_positional_encoding = enabled;
        embedding_lookup.positional_trainable    = enabled;
    }
    bool get_learned_positional() const { return embedding_lookup.positional_trainable; }

    void set_export_valid_lengths(bool enabled) { embedding_lookup.export_valid_lengths = enabled; }
    void set_weights_follow_compute_dtype(bool enabled) { embedding_lookup.weights_follow_compute_dtype = enabled; }
    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    void on_compute_dtype_changed() override { embedding_lookup.compute_dtype = get_compute_dtype(); }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index vocabulary_size = 0;
    Index sequence_length = 0;
    Index embedding_dimension = 0;

    EmbeddingLookupOperator embedding_lookup;
    DropoutOperator         dropout;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
