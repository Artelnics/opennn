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

class  LanguageDataSet : public DataSet
{

public:

    explicit LanguageDataSet(const dimensions& = {0}, const dimensions& = {0});

    explicit LanguageDataSet(const filesystem::path& = filesystem::path());

    const vector<string>& get_input_vocabulary() const;
    const vector<string>& get_target_vocabulary() const;

    Index get_input_vocabulary_size() const;
    Index get_target_vocabulary_size() const;

    Index get_input_length() const;
    Index get_target_length() const;

    void set_default_raw_variables_uses() override;
    void set_raw_variable_uses(const vector<string>&) override;
    void set_raw_variable_uses(const vector<VariableUse>&) override;

    void set_input_vocabulary(const vector<string>&);
    void set_target_vocabulary(const vector<string>&);

    void set_data_random() override;

    void set_default() override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    // void save_vocabulary(const filesystem::path&, const vector<string>&);
    void import_vocabulary(const filesystem::path&, vector<string>&);

    // void save_lengths(const filesystem::path&, const Index&, const Index&);
    void import_lengths(const filesystem::path&, Index&, Index&);

    vector<string> create_vocabulary(const vector<vector<string>>& tokens,
                                        const Index& vocabulary_size,
                                        const vector<string>& reserved_tokens,
                                        const Index& upper_threshold = 10000000,
                                        const Index& lower_threshold = 10,
                                        const Index& iterations_number = 4,
                                        const Index& max_input_tokens = 5000000,
                                        const Index& max_token_length = 50,
                                        const Index& max_unique_chars = 1000,
                                        const float& slack_ratio = 0.05,
                                        const bool& include_joiner_token = true,
                                        const string& joiner = "##");

    void load_documents();

    void read_csv_1();

    void read_csv_2_simple();

    void read_csv_3_language_model();

    void read_csv() override;

    // Empieza por aquí. 

    void read_txt();

//    void write_data_file_whitespace(ofstream&, const vector<vector<string>>&, const vector<vector<string>>&);
    void write_data_file_wordpiece(ofstream&, const vector<vector<string>>&, const vector<vector<string>>&);

    Index count_non_empty_lines() const;

private:

    vector<string> input_vocabulary;

    vector<string> target_vocabulary;

    Index maximum_input_length = 0;

    Index maximum_target_length = 0;

    const Index target_vocabulary_size = 8000;
    const vector<string> reserved_tokens = { "[PAD]", "[UNK]", "[START]", "[END]" };
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
