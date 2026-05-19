//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"
#include "io_utilities.h"

namespace opennn
{

/// @brief Token-based language dataset with input/target vocabularies and binary token cache.
class LanguageDataset final : public Dataset
{

public:

    /// @brief Creates a language dataset, optionally reading documents from the given path.
    LanguageDataset(const filesystem::path& = "");
    /// @brief Creates a language dataset capped at the given vocabulary size.
    LanguageDataset(const filesystem::path&, Index maximum_vocabulary_size);
    /// @brief Creates a language dataset with vocabulary size and minimum token frequency filters.
    LanguageDataset(const filesystem::path&, Index maximum_vocabulary_size, Index minimum_token_frequency);
    /// @brief Creates a synthetic language dataset with the given sample count and vocabulary controls.
    LanguageDataset(const Index, Index, Index);

    const vector<string>& get_input_vocabulary() const { return input_vocabulary; }
    const vector<string>& get_target_vocabulary() const { return target_vocabulary; }

    Index get_input_vocabulary_size() const { return input_vocabulary.size(); }
    Index get_target_vocabulary_size() const { return target_vocabulary.size(); }

    const unordered_map<string, Index>& get_input_vocabulary_map() const { return input_vocabulary_map; }
    const unordered_map<string, Index>& get_target_vocabulary_map() const { return target_vocabulary_map; }
    const unordered_map<Index, string>& get_target_inverse_vocabulary_map() const { return target_inverse_vocabulary_map; }

    Index get_maximum_input_sequence_length() const { return maximum_input_sequence_length; }
    Index get_maximum_target_sequence_length() const { return maximum_target_sequence_length; }

    /// @brief Replaces the input vocabulary and rebuilds its lookup map.
    void set_input_vocabulary(const vector<string>&);
    /// @brief Replaces the target vocabulary and rebuilds its lookup maps.
    void set_target_vocabulary(const vector<string>&);

    void set_maximum_vocabulary_size(Index new_maximum) { maximum_vocabulary_size = new_maximum; }
    void set_minimum_token_frequency(Index new_minimum) { minimum_token_frequency = new_minimum; }

    /// @brief Returns the number of token sequences in the dataset.
    Index get_samples_number() const override;
    using Dataset::get_samples_number;

    /// @brief Returns the distribution of target tokens or classes.
    VectorI calculate_target_distribution() const override;

    /// @brief Reads the configured text corpus into the dataset and builds vocabularies.
    void read_txt();

    /// @brief Builds a vocabulary from the tokenized documents, filtered by frequency and size limits.
    /// @param documents Tokenized input or target documents.
    /// @param vocabulary Output vector receiving the resulting vocabulary.
    void create_vocabulary(const vector<vector<string_view>>&, vector<string>&) const;

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    /// @brief Streams input token indices of the selected samples into the destination buffer.
    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     bool parallelize = true,
                     int = -1) const override;

    /// @brief Streams target token indices of the selected samples into the destination buffer.
    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int = -1) const override;

    /// @brief Streams decoder token indices (target shifted right) into the destination buffer.
    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int = -1) const override;

    inline static const string PAD_TOKEN   = "[PAD]";     // 0
    inline static const string UNK_TOKEN   = "[UNK]";     // 1
    inline static const string START_TOKEN = "[START]";   // 2
    inline static const string END_TOKEN   = "[END]";     // 3

    inline static const float UNK_INDEX = 1.0f;
    inline static const float START_INDEX = 2.0f;
    inline static const float END_INDEX = 3.0f;

    inline static const vector<string> reserved_tokens = {PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN};

private:

    void update_input_vocabulary_map();
    void update_target_vocabulary_maps();

    unordered_map<string_view, Index> create_vocabulary_map(const vector<string>& vocabulary) const;

    void load_documents(string& buffer,
                        vector<vector<string_view>>& input_documents,
                        vector<vector<string_view>>& target_documents,
                        bool has_header_line,
                        bool strict_field_count) const;

    void encode_streaming(const vector<vector<string_view>>&,
                          const vector<vector<string_view>>&,
                          vector<vector<Index>>& in_idx,
                          vector<vector<Index>>& tgt_idx) const;

    void write_binary_cache(const vector<vector<Index>>& in_idx,
                            const vector<vector<Index>>& tgt_idx,
                            bool has_decoder);

    bool try_load_binary_cache(Index expected_samples);

    vector<string> input_vocabulary;
    vector<string> target_vocabulary;

    unordered_map<string, Index> input_vocabulary_map;
    unordered_map<string, Index> target_vocabulary_map;
    unordered_map<Index, string> target_inverse_vocabulary_map;

    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;

    // Binary streaming cache: <data_path>.cache/tokens.bin
    // Header(64B) + offsets table (int64[N][4]) + concat int32 tokens.
    filesystem::path cache_path;
    mutable FileReader cache_reader;
    uint64_t cache_data_off_ = 0;
    vector<array<int64_t, 4>> offsets_table;   // (in_off, in_len, tgt_off, tgt_len)
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
