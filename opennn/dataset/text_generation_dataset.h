//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   G E N E R A T I O N   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <memory>

#include "opennn/dataset/dataset.h"
#include "opennn/core/io_utilities.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"

namespace opennn
{

class TextGenerationDataset final : public Dataset
{

public:

    TextGenerationDataset(const filesystem::path& = "",
                          Index sequence_length = 256,
                          Index maximum_vocabulary_size = 20000,
                          Index minimum_token_frequency = 1);

    const vector<string>& get_vocabulary() const noexcept { return tokenizer->get_vocabulary(); }
    Index get_vocabulary_size() const noexcept { return tokenizer->get_vocabulary_size(); }

    Index get_sequence_length() const noexcept { return sequence_length; }

    void set_tokenizer(unique_ptr<TokenizerOperator>);
    const TokenizerOperator* get_tokenizer() const noexcept { return tokenizer.get(); }

    void set_vocabulary(const vector<string>&);

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

    static constexpr Index UNK_INDEX = 1;

    inline static const vector<string> reserved_tokens = {string(PAD_TOKEN), string(UNK_TOKEN)};

private:

    void configure(Index);
    bool load_cache_metadata(const filesystem::path&, uint64_t);
    void save_cache_metadata(const filesystem::path&, uint64_t, Index) const;

    void fill_blocks(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     int,
                     Index,
                     const char*) const;

    void create_vocabulary(const vector<string_view>&);

    vector<Index> encode_corpus(const vector<string_view>&) const;

    void write_binary_cache(const vector<Index>&, Index);

    unique_ptr<TokenizerOperator> tokenizer =
        make_unique<WordLevelTokenizer>(reserved_tokens);
    bool fixed_vocabulary = false;

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
