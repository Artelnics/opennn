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

class LanguageDataset final : public Dataset
{

public:

    LanguageDataset(const filesystem::path& = "",
                    Index maximum_vocabulary_size = 20000,
                    Index minimum_token_frequency = 1);

    const vector<string>& get_input_vocabulary() const { return input_vocabulary; }
    const vector<string>& get_target_vocabulary() const { return target_vocabulary; }

    Index get_input_vocabulary_size() const { return input_vocabulary.size(); }
    Index get_target_vocabulary_size() const { return target_vocabulary.size(); }

    const unordered_map<string, Index>& get_input_vocabulary_map() const { return input_vocabulary_map; }
    const unordered_map<string, Index>& get_target_vocabulary_map() const { return target_vocabulary_map; }

    Index get_maximum_input_sequence_length() const { return maximum_input_sequence_length; }
    Index get_maximum_target_sequence_length() const { return maximum_target_sequence_length; }

    void set_input_vocabulary(const vector<string>&);
    void set_target_vocabulary(const vector<string>&);

    void set_maximum_vocabulary_size(Index new_maximum) { maximum_vocabulary_size = new_maximum; }
    void set_minimum_token_frequency(Index new_minimum) { minimum_token_frequency = new_minimum; }

    VectorI calculate_target_distribution() const override;

    void read_txt();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     int = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      int = -1) const override;

    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      int = -1) const override;

    bool supports_bf16_inputs() const override { return false; }

    inline static const string PAD_TOKEN   = "[PAD]";
    inline static const string UNK_TOKEN   = "[UNK]";
    inline static const string START_TOKEN = "[START]";
    inline static const string END_TOKEN   = "[END]";

    inline static const float UNK_INDEX = 1.0f;
    inline static const float START_INDEX = 2.0f;
    inline static const float END_INDEX = 3.0f;

    inline static const vector<string> reserved_tokens = {PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN};

private:

    void fill_sequences(const vector<Index>& sample_indices,
                        const vector<Index>& variable_indices,
                        float* output_data,
                        int contiguous,
                        Index sequence_length,
                        Index record_offset,
                        Index shift,
                        const char* context) const;

    void create_vocabulary(const vector<vector<string_view>>&, vector<string>&) const;

    void update_input_vocabulary_map();
    void update_target_vocabulary_map();

    unordered_map<string_view, Index> create_vocabulary_map(const vector<string>& vocabulary) const;

    void load_documents(string& buffer,
                        vector<vector<string_view>>& input_documents,
                        vector<vector<string_view>>& target_documents) const;

    void encode_streaming(const vector<vector<string_view>>&,
                          const vector<vector<string_view>>&,
                          vector<vector<Index>>& input_indices,
                          vector<vector<Index>>& target_indices) const;

    void write_binary_cache(const vector<vector<Index>>& input_indices,
                            const vector<vector<Index>>& target_indices);

    vector<string> input_vocabulary;
    vector<string> target_vocabulary;

    unordered_map<string, Index> input_vocabulary_map;
    unordered_map<string, Index> target_vocabulary_map;

    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;

    filesystem::path cache_path;
    mutable FileReader cache_reader;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
