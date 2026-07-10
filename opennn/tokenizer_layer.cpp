//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tokenizer_layer.h"
#include "string_utilities.h"
#include "json.h"

namespace opennn
{

Tokenizer::Tokenizer(const Shape& new_input_shape, const string& new_label)
    : Layer(LayerType::Tokenizer, false)
{
    input_shape = new_input_shape;
    set_label(new_label);

    check_rank(input_shape, {1}, "Tokenizer", "input");
}

void Tokenizer::set_input_shape(const Shape& new_input_shape)
{
    input_shape = new_input_shape;

    check_rank(input_shape, {1}, "Tokenizer", "input");
}

void Tokenizer::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    tokenizer = move(new_tokenizer);

    operators.clear();
    if (tokenizer) operators = {tokenizer.get()};
}

void Tokenizer::set_vocabulary(const vector<string>& new_vocabulary)
{
    if (!tokenizer) set_tokenizer(make_unique<WordLevelTokenizer>());

    tokenizer->set_vocabulary(new_vocabulary);
}

const vector<string>& Tokenizer::get_vocabulary() const
{
    static const vector<string> empty_vocabulary;

    return tokenizer ? tokenizer->get_vocabulary() : empty_vocabulary;
}

const unordered_map<string, Index>& Tokenizer::get_vocabulary_map() const
{
    static const unordered_map<string, Index> empty_vocabulary_map;

    return tokenizer ? tokenizer->get_vocabulary_map() : empty_vocabulary_map;
}

Index Tokenizer::get_vocabulary_size() const
{
    return tokenizer ? tokenizer->get_vocabulary_size() : 0;
}

void Tokenizer::read_JSON_body(const Json* tokenizer_layer_element)
{
    if (!tokenizer_layer_element->has("TokenizerKind")) return;

    set_tokenizer(make_tokenizer_operator(
        read_json_string(tokenizer_layer_element, "TokenizerKind")));
}

void Tokenizer::write_JSON_body(JsonWriter& printer) const
{
    if (!tokenizer) return;

    write_json(printer, {{"TokenizerKind", tokenizer->get_kind()}});
}

REGISTER(Layer, Tokenizer, "Tokenizer")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
