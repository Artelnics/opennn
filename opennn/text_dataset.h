//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <memory>

#include "dataset.h"
#include "io_utilities.h"
#include "tokenizer_operator.h"

namespace opennn
{

class TextDataset : public Dataset
{
public:

    enum class Task
    {
        Classification,
        SequenceToSequence,
        CausalLanguageModel,
        BertClassification
    };

    TextDataset(const filesystem::path& = "",
                Index sequence_length = 256,
                Index maximum_vocabulary_size = 20000,
                Index minimum_token_frequency = 1);

    static unique_ptr<TextDataset> from_causal_corpus(
        const filesystem::path& data_path,
        Index sequence_length = 256,
        Index maximum_vocabulary_size = 20000,
        Index minimum_token_frequency = 1);

    static unique_ptr<TextDataset> from_classification(
        const filesystem::path& data_path,
        Index maximum_vocabulary_size = 20000,
        Index minimum_token_frequency = 1);

    static unique_ptr<TextDataset> from_sequence_to_sequence(
        const filesystem::path& data_path,
        Index maximum_vocabulary_size = 20000,
        Index minimum_token_frequency = 1);

    static unique_ptr<TextDataset> from_bert_classification(
        const filesystem::path& text_file,
        const filesystem::path& vocabulary_file,
        Index sequence_length);

    virtual Task get_task() const noexcept { return task; }

    virtual const vector<string>& get_input_vocabulary() const noexcept { return vocabulary; }
    virtual const vector<string>& get_target_vocabulary() const noexcept { return vocabulary; }
    virtual unique_ptr<TokenizerOperator> clone_input_tokenizer() const;
    const vector<string>& get_vocabulary() const noexcept { return get_input_vocabulary(); }
    Index get_vocabulary_size() const noexcept { return ssize(get_input_vocabulary()); }
    const unordered_map<string, Index>& get_vocabulary_map() const noexcept { return vocabulary_map; }
    virtual Index get_sequence_length() const noexcept { return sequence_length; }

    void set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer) { tokenizer = move(new_tokenizer); }
    const TokenizerOperator* get_tokenizer() const noexcept { return tokenizer.get(); }

    void set_vocabulary(const vector<string>&);
    void set_maximum_vocabulary_size(Index new_maximum) { maximum_vocabulary_size = new_maximum; }
    void set_minimum_token_frequency(Index new_minimum) { minimum_token_frequency = new_minimum; }

    void read_txt();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     FillMode,
                     int = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      FillMode,
                      int = -1) const override;

    bool supports_bf16_inputs() const override { return false; }

    static constexpr string_view PAD_TOKEN = "[PAD]";
    static constexpr string_view UNK_TOKEN = "[UNK]";
    static constexpr float UNK_INDEX = 1.0f;
    inline static const vector<string> reserved_tokens = {string(PAD_TOKEN), string(UNK_TOKEN)};

protected:

    explicit TextDataset(Task new_task) : task(new_task) {}

private:

    void fill_blocks(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     int,
                     Index,
                     const char*) const;

    void create_vocabulary(const vector<string_view>&);
    void update_vocabulary_map();
    vector<Index> encode_corpus(const vector<string_view>&) const;
    void write_binary_cache(const vector<Index>&, Index);

    unique_ptr<TokenizerOperator> tokenizer;
    vector<string> vocabulary;
    unordered_map<string, Index> vocabulary_map;

    Index sequence_length = 256;
    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;

    filesystem::path cache_path;
    mutable FileReader cache_reader;
    Task task = Task::CausalLanguageModel;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
