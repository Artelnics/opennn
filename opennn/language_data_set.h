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

    // explicit LanguageDataSet(const Index& = 0,
    //                          const dimensions& = {0},
    //                          const dimensions& = {0});


    explicit LanguageDataSet(const filesystem::path& = filesystem::path());

    const vector<string>& get_context_vocabulary() const;
    const vector<string>& get_completion_vocabulary() const;

    Index get_context_vocabulary_size() const;
    Index get_completion_vocabulary_size() const;

    Index get_context_length() const;
    Index get_completion_length() const;

    const dimensions& get_context_dimensions() const;
    const dimensions& get_completion_dimensions() const;

    const vector<vector<string>>& get_documents() const;
    const vector<vector<string>>& get_targets() const;

    void set_default_raw_variables_uses() override;
    void set_raw_variable_uses(const vector<string>&) override;
    void set_raw_variable_uses(const vector<VariableUse>&) override;

    void set_context_dimensions(const dimensions&);

    void set_context_vocabulary_path(const string&);
    void set_completion_vocabulary_path(const string&);

    void set_context_vocabulary(const vector<string>&);
    void set_completion_vocabulary(const vector<string>&);

    void set_data_random() override;

    void set_default() override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    // void save_vocabulary(const filesystem::path&, const vector<string>&);
    void import_vocabulary(const filesystem::path&, vector<string>&);

    // void save_lengths(const filesystem::path&, const Index&, const Index&);
    void import_lengths(const filesystem::path&, Index&, Index&);

    vector<string> calculate_vocabulary(const vector<vector<string>>& tokens,
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

    void load_documents(const string&);

    void read_csv_1();

    void read_csv_2_simple();

    void read_csv_3_language_model();

    void read_csv() override;

    void read_txt();

//    void write_data_file_whitespace(ofstream&, const vector<vector<string>>&, const vector<vector<string>>&);
    void write_data_file_wordpiece(ofstream&, const vector<vector<string>>&, const vector<vector<string>>&);

private:

    dimensions completion_dimensions;
    dimensions context_dimensions;

    vector<string> context_vocabulary;

    string context_vocabulary_path;

    vector<string> completion_vocabulary;

    string completion_vocabulary_path;

    Index maximum_completion_length = 0;

    Index maximum_context_length = 0;

    vector<vector<string>> documents;

    vector<vector<string>> targets;

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
