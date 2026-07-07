//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   G E N E R A T I O N   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <memory>

#include "dataset.h"
#include "io_utilities.h"
#include "tokenizer.h"

namespace opennn
{

class TextGenerationDataset final : public Dataset
{

public:

    TextGenerationDataset(const filesystem::path& = "",
                          Index sequence_length = 256,
                          Index maximum_vocabulary_size = 20000,
                          Index minimum_token_frequency = 1);

    const vector<string>& get_vocabulary() const noexcept { return vocabulary; }
    Index get_vocabulary_size() const noexcept { return vocabulary.size(); }

    const unordered_map<string, Index>& get_vocabulary_map() const noexcept { return vocabulary_map; }

    Index get_sequence_length() const noexcept { return sequence_length; }

    // Optional subword tokenizer (e.g. BytePairTokenizer). When set with a
    // non-empty vocabulary the corpus is encoded through it and the built-in
    // whitespace/word-level path is bypassed; null keeps word-level tokenization.
    void set_tokenizer(unique_ptr<Tokenizer> new_tokenizer) { tokenizer = move(new_tokenizer); }
    const Tokenizer* get_tokenizer() const noexcept { return tokenizer.get(); }

    void set_vocabulary(const vector<string>&);

    void set_maximum_vocabulary_size(Index new_maximum) { maximum_vocabulary_size = new_maximum; }
    void set_minimum_token_frequency(Index new_minimum) { minimum_token_frequency = new_minimum; }

    void read_txt();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool,
                     int = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool,
                      int = -1) const override;

    bool supports_bf16_inputs() const override { return false; }

    static constexpr string_view PAD_TOKEN = "[PAD]";
    static constexpr string_view UNK_TOKEN = "[UNK]";

    static constexpr float UNK_INDEX = 1.0f;

    inline static const vector<string> reserved_tokens = {string(PAD_TOKEN), string(UNK_TOKEN)};

private:

    void fill_blocks(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     int,
                     Index,
                     const char*) const;

    void create_vocabulary(const vector<string_view>&);

    void update_vocabulary_map();

    void read_file(string&) const;

    vector<Index> encode_corpus(const vector<string_view>&) const;

    void write_binary_cache(const vector<Index>&, Index);

    unique_ptr<Tokenizer> tokenizer;

    vector<string> vocabulary;

    unordered_map<string, Index> vocabulary_map;

    Index sequence_length = 256;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;

    filesystem::path cache_path;
    mutable FileReader cache_reader;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
