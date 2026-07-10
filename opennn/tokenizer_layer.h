//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tokenizer_operator.h"

namespace opennn
{

// Non-trainable passthrough layer that carries the tokenizer and its
// vocabulary inside the network, so text inference never needs a dataset.
// Token ids flow through unchanged; encode/decode of strings happens in the
// network-level text entry points, never in the tensor forward pass.
class Tokenizer final : public Layer
{
public:

    Tokenizer(const Shape& = {}, const string& = "tokenizer");

    Shape get_output_shape() const noexcept override { return input_shape; }

    void set_input_shape(const Shape&) override;

    vector<TensorSpec> get_forward_specs(Index) const override { return {}; }

    void set_tokenizer(unique_ptr<TokenizerOperator>);
    const TokenizerOperator* get_tokenizer() const noexcept { return tokenizer.get(); }

    void set_vocabulary(const vector<string>&);
    const vector<string>& get_vocabulary() const;
    const unordered_map<string, Index>& get_vocabulary_map() const;
    Index get_vocabulary_size() const;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    unique_ptr<TokenizerOperator> tokenizer;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
