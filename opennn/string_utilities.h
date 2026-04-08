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
    vector<string_view> get_tokens_fast(string_view, string_view);

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

    string write_time(type);

    string get_first_word(string&);

    void tokenize_whitespace(const vector<string>&, Tensor2&);
    void tokenize_wordpiece(const vector<string>&, Tensor2&);
    void detokenize_whitespace(Tensor2&, ostringstream&);
    void detokenize_wordpiece(Tensor2&, ostringstream&);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
