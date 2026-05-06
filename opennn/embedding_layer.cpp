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
    name = "Embedding";
    layer_type = LayerType::Embedding;

    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);
}

Shape Embedding::get_output_shape() const
{
    return {sequence_length, embedding_dimension};
}

vector<Operator*> Embedding::get_operators()
{
    return {&embedding_lookup, &dropout};
}

vector<pair<Shape, Type>> Embedding::get_forward_specs(Index batch_size) const
{
    return {{{batch_size, sequence_length, embedding_dimension}, compute_dtype}}; // Output
}

void Embedding::set(Index new_vocabulary_size,
                    Index new_sequence_length,
                    Index new_embedding_dimension,
                    const string& new_label)
{
    vocabulary_size     = new_vocabulary_size;
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;

    set_label(new_label);

    embedding_lookup.set(vocabulary_size, sequence_length, embedding_dimension);

    embedding_lookup.input_slots  = {Input};
    embedding_lookup.output_slots = {Output};

    dropout.input_slots  = {Output};
    dropout.output_slots = {Output};
}

void Embedding::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& delta_views = back_propagation.delta_views[layer];

    TensorView& output_delta = delta_views[OutputDelta][0];

    if (dropout.active())
        dropout.apply_delta(output_delta);

    embedding_lookup.apply_delta(forward_views[Input][0], output_delta);
}

void Embedding::read_JSON_body(const Json* embedding_layer_element)
{
    const string new_label = read_json_string(embedding_layer_element, "Label");
    const Index new_vocabulary_size = read_json_index(embedding_layer_element, "VocabularySize");
    const Shape new_output_shape = string_to_shape(read_json_string(embedding_layer_element, "OutputDimensions"));

    set(new_vocabulary_size,
        new_output_shape.empty() ? Index(0) : new_output_shape[0],
        new_output_shape.rank >= 2 ? new_output_shape[1] : Index(0),
        new_label);

    set_scale_embedding(read_json_bool(embedding_layer_element, "ScaleEmbedding"));
    set_add_positional_encoding(read_json_bool(embedding_layer_element, "AddPositionalEncoding"));
}

void Embedding::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"VocabularySize", to_string(get_vocabulary_size())},
        {"ScaleEmbedding", to_string(embedding_lookup.scale_embedding)},
        {"AddPositionalEncoding", to_string(embedding_lookup.add_positional_encoding)}
    });
}

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
