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

// OpenNN includes

#include "vector.h"

using namespace std;

namespace OpenNN
{
    size_t count_tokens(const string&, const char&);

    Vector<string> get_tokens(const string&, const char&);

//    inline bool is_digit_string(const char str) {return std::isdigit(str);}
    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);

    time_t date_to_timestamp(const string& date, const int& gmt = 0);

    Vector<double> to_double_vector(const string& , const char&);

    bool contains_substring(const string&, const string&);

    void trim(string&);
    void erase(string&, const char&);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const Vector<string>&);
    bool has_strings(const Vector<string>&);

    bool is_numeric_string_vector(const Vector<string>&);

    bool is_not_numeric(const Vector<string>&);
    bool is_mixed(const Vector<string>&);

    void replace(string& source, const string& find, const string& replace);
    void replace_substring(Vector<string>&, const string& , const string& );
}

#endif // OPENNNSTRINGS_H
