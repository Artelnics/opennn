//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/dataset/dataset.h"
#include "opennn/core/io_utilities.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"

namespace opennn
{

class LanguageDataset final : public Dataset
{

public:

    LanguageDataset(const filesystem::path& = "",
                    Index maximum_vocabulary_size = 20000,
                    Index minimum_token_frequency = 1);

    const vector<string>& get_input_vocabulary() const noexcept { return input_tokenizer->get_vocabulary(); }
    const vector<string>& get_target_vocabulary() const noexcept { return target_tokenizer->get_vocabulary(); }

    Index get_input_vocabulary_size() const noexcept { return input_tokenizer->get_vocabulary_size(); }
    Index get_target_vocabulary_size() const noexcept { return target_tokenizer->get_vocabulary_size(); }

    const TokenizerOperator& get_input_tokenizer() const noexcept { return *input_tokenizer; }

    Index get_maximum_input_sequence_length() const noexcept { return maximum_input_sequence_length; }
    Index get_maximum_target_sequence_length() const noexcept { return maximum_target_sequence_length; }

    void set_classification_target(bool new_classification_target) { classification_target = new_classification_target; }

    void set_input_sequence_length_limit(Index new_limit) { input_sequence_length_limit = new_limit; }

    VectorI calculate_target_distribution() const override;

    void read_txt();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     FillMode,
                     ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      FillMode,
                      ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      FillMode,
                      ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    bool supports_bf16_inputs() const override { return false; }

    static constexpr string_view PAD_TOKEN   = "[PAD]";
    static constexpr string_view UNK_TOKEN   = "[UNK]";
    static constexpr string_view START_TOKEN = "[START]";
    static constexpr string_view END_TOKEN   = "[END]";

    static constexpr Index UNK_INDEX = TokenizerOperator::UNK_INDEX;
    static constexpr Index START_INDEX = TokenizerOperator::START_INDEX;
    static constexpr Index END_INDEX = TokenizerOperator::END_INDEX;

    inline static const vector<string> reserved_tokens = {string(PAD_TOKEN), string(UNK_TOKEN), string(START_TOKEN), string(END_TOKEN)};

private:

    void configure(Index samples_number, bool has_decoder);
    bool load_cache_metadata(const filesystem::path&);
    void save_cache_metadata(const filesystem::path& metadata_path, Index samples_number, bool has_decoder) const;

    void fill_sequences(const vector<Index>&,
                        const vector<Index>&,
                        float*,
                        ColumnContiguity column_contiguity,
                        Index,
                        Index,
                        Index,
                        const char*) const;

    void load_documents(vector<vector<string>>&,
                        vector<vector<string>>&) const;

    void encode_streaming(const vector<vector<string>>&,
                          const vector<vector<string>>&,
                          vector<vector<Index>>&,
                          vector<vector<Index>>&) const;

    void write_binary_cache(const vector<vector<Index>>&,
                            const vector<vector<Index>>&);

    unique_ptr<TokenizerOperator> input_tokenizer  = make_unique<WordLevelTokenizer>();
    unique_ptr<TokenizerOperator> target_tokenizer = make_unique<WordLevelTokenizer>();

    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;

    bool classification_target = false;

    Index input_sequence_length_limit = 0;

    filesystem::path cache_path;
    mutable FileReader cache_reader;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
