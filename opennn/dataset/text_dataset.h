// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/dataset/dataset.h"
#include "opennn/core/io_utilities.h"
#include "opennn/network/operators/tokenizer_operator.h"

namespace opennn
{

class TextDataset final : public Dataset
{
public:
    enum class Task { Classification, SequenceToSequence, NextToken };
    enum class InputLayout { Tokens, TokensAndMask };

    struct Options
    {
        Task task = Task::Classification;
        InputLayout input_layout = InputLayout::Tokens;
        Index sequence_length = 0;
        Index maximum_vocabulary_size = 20000;
        Index minimum_token_frequency = 1;
    };

    TextDataset();
    explicit TextDataset(Options);

    const Options& get_options() const noexcept { return options; }
    void read_txt(const filesystem::path&);
    void set_tokenizer(unique_ptr<TokenizerOperator>, VariableRole = VariableRole::Input);
    const TokenizerOperator* get_tokenizer(VariableRole = VariableRole::Input) const noexcept;
    const vector<string>& get_vocabulary(VariableRole = VariableRole::Input) const noexcept;
    Index get_vocabulary_size(VariableRole role = VariableRole::Input) const noexcept
    {
        return Index(get_vocabulary(role).size());
    }
    Index get_sequence_length(VariableRole role = VariableRole::Input) const
    {
        const Shape& shape = get_shape(role);
        return shape.empty() ? 0 : shape.size();
    }

    const TokenizerOperator* get_training_tokenizer() const override { return get_tokenizer(); }
    const TokenizerOperator* get_training_tokenizer(VariableRole role) const override { return get_tokenizer(role); }
    bool supports_bf16_inputs() const override { return options.input_layout == InputLayout::TokensAndMask; }
    VectorI calculate_target_distribution() const override;

    using Dataset::set_storage_mode;
    void set_storage_mode(StorageMode) override;
    void to_JSON(JsonWriter&) const override;
    void from_JSON(const JsonDocument&) override;

    void fill_inputs(const vector<Index>&, const vector<Index>&, float*, FillMode,
                     ColumnContiguity = ColumnContiguity::Unknown) const override;
    void fill_targets(const vector<Index>&, const vector<Index>&, float*, FillMode,
                      ColumnContiguity = ColumnContiguity::Unknown) const override;
    void fill_decoder(const vector<Index>&, const vector<Index>&, float*, FillMode,
                      ColumnContiguity = ColumnContiguity::Unknown) const override;

private:
    struct Segment
    {
        VariableRole role;
        Index length;
        Index offset;
        Index prefix = -1;
    };

    VariableRole token_role() const noexcept;
    void configure(Index samples, Index input_length, Index target_length);
    void load_documents(vector<vector<string>>&, vector<vector<string>>&) const;
    void build_labels(const vector<vector<string>>&);
    void read_rows();
    void read_corpus();
    bool uses_cache() const noexcept;
    void store_matrix_record(Index, span<const int32_t>);
    void write_records(Index, const function<void(Index, span<int32_t>)>&);
    void fill_sequences(VariableRole, const vector<Index>&, const vector<Index>&,
                        float*, ColumnContiguity) const;
    void write_configuration(JsonWriter&) const;
    void read_configuration(const Json*);
    bool load_cache(const filesystem::path&);
    void save_cache(const filesystem::path&) const;

    Options options;
    unique_ptr<TokenizerOperator> tokenizer;
    unique_ptr<TokenizerOperator> target_tokenizer;
    bool fixed_vocabulary = false;
    bool fixed_target_vocabulary = false;
    string tokenizer_identity;
    string target_tokenizer_identity;
    vector<string> labels;
    vector<Segment> segments;
    Index record_tokens = 0;
    filesystem::path cache_path;
    mutable FileReader cache_reader;
};

}
