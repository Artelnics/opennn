//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef STRINGS_H
#define STRINGS_H

#include "pch.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

    void prepare_line(string&);
    Index count_non_empty_lines(const filesystem::path&);

    Index count_tokens(const string&, const string&);

    vector<string> get_tokens(const string&, const string&);

    vector<string> tokenize(const string&);

    vector<string> convert_string_vector(const vector<vector<string>>&, const string&);

    Tensor<type, 1> to_type_vector(const string&, const string&);

    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);

    enum DateFormat {AUTO, DMY, MDY, YMD};

    time_t date_to_timestamp(const string&, const Index& = 0, const DateFormat& format = AUTO);

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

    void print_tokens(const vector<vector<string>>&);
}

#endif // OPENNNSTRINGS_H
