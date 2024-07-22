//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LANGUAGEDATASET_H
#define LANGUAGEDATASET_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

// OpenNN includes

#include "config.h"
#include "data_set.h"
#include "strings_utilities.h"

namespace opennn
{

class LanguageDataSet : public DataSet
{

public:

    // DEFAULT CONSTRUCTOR

    explicit LanguageDataSet();

    string get_text_separator_string() const;

    Tensor<string, 1> get_context_vocabulary() const;
    Tensor<string, 1> get_completion_vocabulary() const;

    Index get_context_vocabulary_size() const;
    Index get_completion_vocabulary_size() const;

    Index get_context_length() const;
    Index get_completion_length() const;

    Index get_context_variables_number() const;
    const Tensor<Index, 1>& get_context_variables_dimensions() const;
    Tensor<Index, 1> get_context_variables_indices() const;
    Index get_context_raw_variables_number() const;
    Tensor<Index, 1> get_context_raw_variables_indices() const;

    Tensor<type, 2> get_testing_context_data() const;

    const Tensor<Tensor<string, 1>, 1> get_documents() const;
    const Tensor<Tensor<string, 1>, 1> get_targets() const;

    Tensor<type, 2> get_context_data() const;

    void set_default_raw_variables_uses();
    void set_raw_variables_uses(const Tensor<string, 1>& new_raw_variables_uses);
    void set_raw_variables_uses(const Tensor<VariableUse, 1>& new_raw_variables_uses);

    void set_context_variables_dimensions(const Tensor<Index, 1>& new_context_dimensions);

    void set_text_separator(const Separator&);
    void set_text_separator(const string&);

    void set_context_vocabulary_path(const string&);
    void set_completion_vocabulary_path(const string&);

    void set_data_random_language_model(const Index&, const Index&, const Index&, const Index&, const Index&);

    void set_default();

    Tensor<string, 2> get_text_data_file_preview() const;

    void from_XML(const tinyxml2::XMLDocument&);
    void write_XML(tinyxml2::XMLPrinter&) const;

    void import_vocabulary(const string&, Tensor<string, 1>&);
    const Tensor<string, 1> calculate_vocabulary(const Tensor<Tensor<string, 1>, 1>& tokens,
                                                 int vocabulary_size,
                                                 const vector<string>& reserved_tokens,
                                                 int upper_threshold = 10000000,
                                                 int lower_threshold = 10,
                                                 int iterations_number = 4,
                                                 int max_input_tokens = 5000000,
                                                 int max_token_length = 50,
                                                 int max_unique_chars = 1000,
                                                 float slack_ratio = 0.05,
                                                 bool include_joiner_token = true,
                                                 const string& joiner = "##");

    void load_documents(const string&);
    void read_csv_3_language_model();

    void read_csv_language_model();

    void read_txt_language_model();
    void write_data_file_whitespace(ofstream&, const Tensor<Tensor<string, 1>, 1>&, const Tensor<Tensor<string, 1>, 1>&);
    void write_data_file_wordpiece(ofstream&, const Tensor<Tensor<string, 1>, 1>&, const Tensor<Tensor<string, 1>, 1>&);

private:

    Separator text_separator = Separator::Tab;

    Tensor<string, 2> text_data_file_preview;

    // LARGE LANGUAGE MODEL

    Tensor<string, 1> context_vocabulary;

    string context_vocabulary_path = "";

    Tensor<string, 1> completion_vocabulary;

    string completion_vocabulary_path = "";

    Tensor<Index, 1> context_variables_dimensions = Tensor<Index, 1>(1);

    Index max_completion_length = 0;

    Index max_context_length = 0;

    Tensor<Tensor<string, 1>, 1> documents;

    Tensor<Tensor<string, 1>, 1> targets;
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
