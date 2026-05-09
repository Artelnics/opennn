//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "materialized_dataset.h"

namespace opennn
{

class LanguageDataset final : public MaterializedDataset
{

public:

    LanguageDataset(const filesystem::path& = "", bool streaming = false);
    LanguageDataset(const Index, Index, Index);

    const vector<string>& get_input_vocabulary() const { return input_vocabulary; }
    const vector<string>& get_target_vocabulary() const { return target_vocabulary; }

    Index get_input_vocabulary_size() const { return input_vocabulary.size(); }
    Index get_target_vocabulary_size() const { return target_vocabulary.size(); }

    Index get_maximum_input_sequence_length() const { return maximum_input_sequence_length; }
    Index get_maximum_target_sequence_length() const { return maximum_target_sequence_length; }

    void set_input_vocabulary(const vector<string>& new_vocabulary) { input_vocabulary = new_vocabulary; }
    void set_target_vocabulary(const vector<string>& new_vocabulary) { target_vocabulary = new_vocabulary; }

    bool is_streaming() const { return streaming; }
    void set_streaming(bool b) { streaming = b; }

    Index get_samples_number() const override;
    using MaterializedDataset::get_samples_number;

    void read_csv() override;

    void create_vocabulary(const vector<vector<string>>&, vector<string>&) const;

    void encode_input(const vector<vector<string>>&);
    void encode_decoder_target_sequence_to_sequence(const vector<vector<string>>&);
    void encode_target_classification(const vector<vector<string>>&);

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool = true,
                     int = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool = true,
                      int = -1) const override;

    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool = true,
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

    unordered_map<string, Index> create_vocabulary_map(const vector<string>& vocabulary);

    void encode_streaming(const vector<vector<string>>&,
                          const vector<vector<string>>&);

    vector<string> input_vocabulary;
    vector<string> target_vocabulary;

    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;

    bool streaming = false;

    vector<vector<Index>> sample_input_indices;
    vector<vector<Index>> sample_target_indices;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
