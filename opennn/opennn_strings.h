//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N  S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef OPENNNSTRINGS_H
#define OPENNNSTRINGS_H

// System includes

#include <math.h>
#include <regex>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <string_view>
#include <cctype>
#include <iomanip>

// Eigen includes

#include "config.h"

namespace opennn
{
    Index count_tokens(const string&, const char& separator = ' ');
    Tensor<string, 1> get_tokens(const string&, const char& delimiter=' ');
    void fill_tokens(const string&, const char&, Tensor<string, 1>&);

    Index count_tokens(const string&, const string&);
    Tensor<string, 1> get_tokens(const string&, const string&);

    Tensor<type, 1> to_type_vector(const string&, const char&);
    Tensor<Index, 1> to_index_vector(const string&, const char&);

    Tensor<string, 1> get_unique_elements(const Tensor<string,1>&);
    Tensor<Index, 1> count_unique(const Tensor<string,1>&);

    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);
    bool is_email(const string&);
    bool contains_number(const string&);

    bool starts_with(const string& word, const string& starting);
    bool ends_with(const string&, const string&);
    bool ends_with(const string&, const Tensor<string,1>&);

    bool is_constant_numeric(const Tensor<type, 1>&);
    bool is_constant_string(const Tensor<string, 1>&);

    time_t date_to_timestamp(const string& date, const Index& gmt = 0);

    bool contains_substring(const string&, const string&);

    void replace_all_appearances(std::string& s, std::string const& toReplace, std::string const& replaceWith);
    string replace_non_allowed_programming_expressions(std::string& s);
    vector<string> get_words_in_a_string(string str);

    Tensor<string, 1> fix_write_expression_outputs(const string&, const Tensor<string, 1>&, const string&);
    Tensor<Tensor<string,1>, 1> fix_input_output_variables(Tensor<string, 1>&, Tensor<string, 1>&, ostringstream&);

    int WordOccurrence(char *sentence, char *word);

    void trim(string&);
    void erase(string&, const char&);
    void replace_first_char_with_missing_label(string &str, char target_char, const string &missing_label);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const Tensor<string, 1>&);
    bool has_strings(const Tensor<string, 1>&);

    bool is_numeric_string_vector(const Tensor<string, 1>&);

    bool is_not_numeric(const Tensor<string, 1>&);
    bool is_mixed(const Tensor<string, 1>&);

    void remove_non_printable_chars( std::string&);

    void replace(string&, const string&, const string&);
    void replace_substring(Tensor<string, 1>&, const string& , const string&);

    bool isNotAlnum(char &c);
    void remove_not_alnum(string &str);

    bool find_string_in_tensor(Tensor<string, 1>& t, string val);
    string get_word_from_token(string&);
    Tensor<string, 1> push_back_string (Tensor<string, 1>&, const string&);

    string round_to_precision_string(type,const int&);
    Tensor<string,2> round_to_precision_string_matrix(Tensor<type,2>, const int&);
}

#endif // OPENNNSTRINGS_H
