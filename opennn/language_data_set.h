//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LANGUAGEDATASET_H
#define LANGUAGEDATASET_H

#include "data_set.h"

namespace opennn
{

class LanguageDataSet : public DataSet
{

public:

    LanguageDataSet(const dimensions& input_dims = dimensions(0), const dimensions& target_dims = dimensions(0));
    LanguageDataSet(const filesystem::path&);

    const unordered_map<string, Index>& get_input_vocabulary() const;
    const unordered_map<string, Index>& get_target_vocabulary() const;

    Index get_input_vocabulary_size() const;
    Index get_target_vocabulary_size() const;

    Index get_input_size() const;
    Index get_target_size() const;

    void set_input_vocabulary(const unordered_map<string, Index>&);
    void set_target_vocabulary(const unordered_map<string, Index>&);

    void set_data_random() override;

    void read_csv() override;

    Index count_non_empty_lines() const;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    vector<string> tokenize(const string& document);

    unordered_map<string, Index> create_vocabulary(const vector<vector<string>>& document_tokens);

    void print() const override;

    void print_vocabulary(const unordered_map<string, Index>& vocabulary);

    inline static const string PAD_TOKEN = "[PAD]";     // 0
    inline static const string UNK_TOKEN = "[UNK]";     // 1
    inline static const string START_TOKEN = "[START]"; // 2
    inline static const string END_TOKEN = "[END]";     // 3

    inline static const vector<string> RESERVED_TOKENS = {
        PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN
    };

private:

    vector<vector<string>> input_tokens;
    vector<vector<string>> target_tokens;

    unordered_map<string, Index> input_vocabulary;
    unordered_map<string, Index> target_vocabulary;

    Index maximum_input_size = 0;
    Index maximum_target_size = 0;

    Index target_vocabulary_size = 0;
    Index input_vocabulary_size = 0;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
