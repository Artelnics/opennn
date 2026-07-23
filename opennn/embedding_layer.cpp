//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_types.h"
#include "embedding_layer.h"

namespace opennn
{

Embedding::Embedding(const Shape& new_input_shape,
                     Index new_embedding_dimension,
                     const string& new_label)
    : Layer(LayerType::Embedding)
{
    operators = {&embedding_lookup, &dropout};
    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);
}

Shape Embedding::get_output_shape() const
{
    return {sequence_length, embedding_dimension};
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

    dropout.input_slots  = {Output};
}

void Embedding::read_JSON_body(const Json* embedding_layer_element)
{
    const Index new_vocabulary_size = read_json_index(embedding_layer_element, "VocabularySize");
    const Shape new_output_shape = string_to_shape(read_json_string(embedding_layer_element, "OutputDimensions"));

    set(new_vocabulary_size,
        new_output_shape.dim_or_zero(0),
        new_output_shape.dim_or_zero(1),
        get_label());

    if (embedding_layer_element->has("LearnedPositional"))
        set_learned_positional(read_json_bool(embedding_layer_element, "LearnedPositional"));
    set_scale_embedding(read_json_bool(embedding_layer_element, "ScaleEmbedding"));
    set_add_positional_encoding(read_json_bool(embedding_layer_element, "AddPositionalEncoding"));
    if (embedding_layer_element->has("ExportValidLengths"))
        set_export_valid_lengths(read_json_bool(embedding_layer_element, "ExportValidLengths"));
}

void Embedding::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"VocabularySize", to_string(get_vocabulary_size())},
        {"OutputDimensions", shape_to_string(get_output_shape())},
        {"ScaleEmbedding", to_string(embedding_lookup.scale_embedding)},
        {"AddPositionalEncoding", to_string(embedding_lookup.add_positional_encoding)},
        {"LearnedPositional", to_string(embedding_lookup.positional_trainable)},
        {"ExportValidLengths", to_string(embedding_lookup.export_valid_lengths)}
    });
}

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
