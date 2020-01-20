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

// Eigen includes

#include "config.h"
#include "../eigen/Eigen/Eigen"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace Eigen;

namespace OpenNN
{
    int count_tokens(const string&, const char&);

    vector<string> get_tokens(const string&, const char&);

    Tensor<type, 1> to_double_vector(const string&, const char&);

//    inline bool is_digit_string(const char str) {return std::isdigit(str);}
    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);

    time_t date_to_timestamp(const string& date, const int& gmt = 0);

    bool contains_substring(const string&, const string&);

    void trim(string&);
    void erase(string&, const char&);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const vector<string>&);
    bool has_strings(const vector<string>&);

    bool is_numeric_string_vector(const vector<string>&);

    bool is_not_numeric(const vector<string>&);
    bool is_mixed(const vector<string>&);

    void replace(string& source, const string& find, const string& replace);
    void replace_substring(vector<string>&, const string& , const string&);
}

#endif // OPENNNSTRINGS_H
