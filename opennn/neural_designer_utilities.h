#pragma once

#include "pch.h"

namespace opennn
{

/// @brief Sorts a vector of strings in place in ascending lexicographical order.
void sort_string_vector(vector<string>&);

/// @brief Concatenates two string vectors and returns the resulting vector.
vector<string> concatenate_string_vectors(const vector<string>&, const vector<string>&);

/// @brief Formats a floating point number as a string using the given number of decimal digits.
string formatNumber(float, int);

/// @brief Rounds the given floating point value to the requested number of significant digits.
float round_to_precision(float, const int&);

}
