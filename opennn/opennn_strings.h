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

// Eigen includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace OpenNN
{
    Index count_tokens(const string&, const char&);

    Tensor<string, 1> get_tokens(const string&, const char&);
    void fill_tokens(const string&, const char&, Tensor<string, 1>&);

    Tensor<type, 1> to_type_vector(const string&, const char&);

//    inline bool is_digit_string(const char str) {return std::isdigit(str);}
    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);
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

    void replace(string& source, const string& find, const string& replace);
    void replace_substring(Tensor<string, 1>&, const string& , const string&);
}

#endif // OPENNNSTRINGS_H
