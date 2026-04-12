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
    LanguageDataset(const Index, Index, Index);

    const vector<string>& get_input_vocabulary() const { return input_vocabulary; }
    const vector<string>& get_target_vocabulary() const { return target_vocabulary; }

    Index get_input_vocabulary_size() const { return input_vocabulary.size(); }
    Index get_target_vocabulary_size() const { return target_vocabulary.size(); }

    Index get_maximum_input_sequence_length() const { return maximum_input_sequence_length; }
    Index get_maximum_target_sequence_length() const { return maximum_target_sequence_length; }

    void set_input_vocabulary(const vector<string>& v) { input_vocabulary = v; }
    void set_target_vocabulary(const vector<string>& v) { target_vocabulary = v; }

    void read_csv() override;

    void create_vocabulary(const vector<vector<string>>&, vector<string>&) const;

    void encode_input(const vector<vector<string>>&);
    void encode_decoder_target_sequence_to_sequence(const vector<vector<string>>&);
    void encode_target_classification(const vector<vector<string>>&);

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

    inline static const string PAD_TOKEN   = "[PAD]";     // 0
    inline static const string UNK_TOKEN   = "[UNK]";     // 1
    inline static const string START_TOKEN = "[START]";   // 2
    inline static const string END_TOKEN   = "[END]";     // 3

    inline static const type UNK_INDEX = 1.0f;
    inline static const type START_INDEX = 2.0f;
    inline static const type END_INDEX = 3.0f;

    inline static const vector<string> reserved_tokens = {PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN};

private:

    unordered_map<string, Index> create_vocabulary_map(const vector<string>& vocabulary);

    vector<string> input_vocabulary;
    vector<string> target_vocabulary;

    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;

    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
