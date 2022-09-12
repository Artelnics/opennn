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
#include <iostream>
#include <cctype>

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

    void trim(string&);
    void erase(string&, const char&);

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
}

#endif // OPENNNSTRINGS_H
