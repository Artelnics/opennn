// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/core/variable.h"
#include "opennn/network/layers/layer.h"
#include "opennn/network/operators/tokenizer_operator.h"

namespace opennn
{

class Tokenizer final : public Layer
{
public:

    Tokenizer(const Shape& = {},
              const string& = "tokenizer",
              VariableRole = VariableRole::Input);

    Shape get_output_shape() const noexcept override { return input_shape; }

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 1); }
    bool allows_bf16_input_cast(size_t) const noexcept override { return false; }

    vector<TensorSpec> get_forward_specs(Index) const override { return {}; }

    void set_tokenizer(unique_ptr<TokenizerOperator>);
    const TokenizerOperator* get_tokenizer() const noexcept { return tokenizer.get(); }
    VariableRole get_variable_role() const noexcept { return variable_role; }

    void set_vocabulary(const vector<string>&);
    const vector<string>& get_vocabulary() const;
    const TokenizerOperator::VocabularyMap& get_vocabulary_map() const;
    Index get_vocabulary_size() const { return tokenizer ? tokenizer->get_vocabulary_size() : 0; }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    unique_ptr<TokenizerOperator> tokenizer;
    VariableRole variable_role = VariableRole::Input;
};

}
