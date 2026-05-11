//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"

namespace opennn
{

class LanguageDataset final : public Dataset
{

public:

    LanguageDataset(const filesystem::path& = "");
    LanguageDataset(const filesystem::path&, Index maximum_vocabulary_size);
    LanguageDataset(const filesystem::path&, Index maximum_vocabulary_size, Index minimum_token_frequency);
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

    void set_input_vocabulary(const vector<string>&);
    void set_target_vocabulary(const vector<string>&);

    void set_maximum_vocabulary_size(Index new_maximum) { maximum_vocabulary_size = new_maximum; }
    void set_minimum_token_frequency(Index new_minimum) { minimum_token_frequency = new_minimum; }

    void read_csv() override;

    void create_vocabulary(const vector<vector<string>>&, vector<string>&) const;

    void encode_input(const vector<vector<string>>&);
    void encode_decoder_target_sequence_to_sequence(const vector<vector<string>>&);
    void encode_target_classification(const vector<vector<string>>&);

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

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

    vector<string> input_vocabulary;
    vector<string> target_vocabulary;

    unordered_map<string, Index> input_vocabulary_map;
    unordered_map<string, Index> target_vocabulary_map;
    unordered_map<Index, string> target_inverse_vocabulary_map;

    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
