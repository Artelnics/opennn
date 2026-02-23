//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

    void prepare_line(string&);
    Index count_non_empty_lines(const filesystem::path&);

    Index count_tokens(const string&, const string&);

    vector<string> get_tokens(const string&, const string&);

    vector<string> tokenize(const string&);

    vector<string> convert_string_vector(const vector<vector<string>>&, const string&);

    VectorR to_type_vector(const string&, const string&);

    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);

    enum DateFormat {AUTO, DMY, MDY, YMD};

    time_t date_to_timestamp(const string&, Index = 0, const DateFormat& format = AUTO);

    void replace_all_appearances(string&, const string&, const string&);
    void replace_all_word_appearances(string&, const string&, const string&);

    void trim(string&);
    void erase(string&, const char&);

    void replace_first_and_last_char_with_missing_label(string&, char, const string&, const string&);

    string get_trimmed(const string&);

    bool has_numbers(const vector<string>&);

    void replace(string&, const string&, const string&);
    void replace_double_char_with_label(string&, const string&, const string&);
    void replace_substring_within_quotes(string&, const string&, const string&);
    void replace_substring_in_string (vector<string>&, string&, const string&);

    void display_progress_bar(const int&, const int&);

    bool contains(vector<string>&, const string&);
    string get_first_word(string&);

    void sort_string_vector(vector<string>&);
    vector<string> concatenate_string_vectors (const vector<string>&, const vector<string>&);

    enum Language {ENG, SPA};

    void set_language(const Language&);

    void set_language(const string&);

    void tokenize_whitespace(const vector<string>&, Tensor2&);
    void tokenize_wordpiece(const vector<string>&, Tensor2&);
    void detokenize_whitespace(Tensor2&, ostringstream&);
    void detokenize_wordpiece(Tensor2&, ostringstream&);

    vector<string> preprocess_language_document(const string&, bool);

    string formatNumber(type, int);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
