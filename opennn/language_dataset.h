//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LANGUAGEDataset_H
#define LANGUAGEDataset_H

#include "dataset.h"

namespace opennn
{

class LanguageDataset : public Dataset
{

public:

    LanguageDataset(const filesystem::path& = "");
    LanguageDataset(const Index&, const Index&, const Index&);

    const vector<string>& get_input_vocabulary() const;
    const vector<string>& get_target_vocabulary() const;

    Index get_input_vocabulary_size() const;
    Index get_target_vocabulary_size() const;

    Index get_input_sequence_length() const;
    Index get_target_sequence_length() const;

    void set_input_vocabulary(const vector<string>&);
    void set_target_vocabulary(const vector<string>&);

    void read_csv() override;

    Index count_non_empty_lines() const;

    void create_vocabulary(const vector<vector<string>>&, vector<string>&) const;

    void encode_input_data(const vector<vector<string>>&);
    void encode_target_data(const vector<vector<string>>&);

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print_input_vocabulary() const;
    void print_target_vocabulary() const;

    inline static const string PAD_TOKEN = "[PAD]";     // 0
    inline static const string UNK_TOKEN = "[UNK]";     // 1
    inline static const string START_TOKEN = "[START]"; // 2
    inline static const string END_TOKEN = "[END]";     // 3

    inline static const vector<string> reserved_tokens = {PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN};

    dimensions get_input_dimensions() const
    {
        return dimensions({get_input_vocabulary_size(), get_input_sequence_length()});
    }


    dimensions get_target_dimensions() const
    {
        // @todo
        //get_variables_number()

        return dimensions({1});
    }

private:

    vector<string> input_vocabulary;
    vector<string> target_vocabulary;

    Index maximum_input_length = 0;
    Index maximum_target_length = 0;

    Index minimum_word_frequency = 2;
    Index maximum_vocabulary_size = 1000;
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
