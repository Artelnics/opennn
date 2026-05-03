//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "embedding_layer.h"
#include "neural_network.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "random_utilities.h"

namespace opennn
{

Embedding::Embedding(const Shape& new_input_shape,
                     Index new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);

    name = "Embedding";
    layer_type = LayerType::Embedding;
}

// Getters

Shape Embedding::get_output_shape() const
{
    return {sequence_length, embedding_dimension};
}

vector<Operator*> Embedding::get_operators()
{
    return {&embedding_lookup};
}

// Setters

void Embedding::set(const Index new_vocabulary_size,
                    Index new_sequence_length,
                    Index new_embedding_dimension,
                    const string& new_label)
{
    sequence_length = new_sequence_length;
    vocabulary_size = new_vocabulary_size;
    embedding_dimension = new_embedding_dimension;
    label = new_label;

    embedding_lookup.set(vocabulary_size, sequence_length, embedding_dimension);
}

// link_parameters() and link_states() are inherited from Layer; the base
// auto-distributes slices to embedding_lookup. init_positional_encoding() is
// called from inside EmbeddingLookup::link_states().

// Parameter initialization

void Embedding::set_parameters_random()
{
    if (parameters[Weight].empty()) return;

    MatrixMap weights = parameters[Weight].as_matrix();
    set_random_normal(weights, 0.0f, 1.0f);

    weights.row(0).setZero();
}

void Embedding::set_parameters_glorot()
{
    if (parameters[Weight].empty()) return;

    const float limit = sqrt(6.0f / (vocabulary_size + embedding_dimension));

    MatrixMap weights = parameters[Weight].as_matrix();

    weights.setRandom();
    weights *= limit;

    weights.row(0).setZero();
}

// Forward / back propagation

void Embedding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    embedding_lookup.apply(forward_views[Input][0], forward_views[Output][0]);

    if (is_training && dropout.active())
        dropout.apply(forward_views[Output][0]);
}

void Embedding::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& delta_views = back_propagation.delta_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    TensorView& output_delta = delta_views[OutputDelta][0];

    if (dropout.active())
        dropout.apply_delta(output_delta);

    embedding_lookup.apply_delta(forward_views[Input][0], output_delta, gradient_views[Weight]);
}

// Serialization

void Embedding::from_JSON(const JsonDocument& document)
{
    const Json* embedding_layer_element = get_json_root(document, "Embedding");

    const string new_label = read_json_string(embedding_layer_element, "Label");
    const Index new_vocabulary_size = read_json_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_json_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_json_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_label);

    set_scale_embedding(read_json_bool(embedding_layer_element, "ScaleEmbedding"));
    set_add_positional_encoding(read_json_bool(embedding_layer_element, "AddPositionalEncoding"));
}

void Embedding::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Embedding");

    write_json(printer, {
        {"Label", label},
        {"VocabularySize", to_string(get_vocabulary_size())},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingSize", to_string(get_embedding_dimension())},
        {"ScaleEmbedding", to_string(embedding_lookup.scale_embedding)},
        {"AddPositionalEncoding", to_string(embedding_lookup.add_positional_encoding)}
    });

    printer.close_element();
}

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
