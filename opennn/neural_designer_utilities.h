#pragma once

#include "pch.h"

namespace opennn
{

void sort_string_vector(vector<string>&);
vector<string> concatenate_string_vectors(const vector<string>&, const vector<string>&);
string formatNumber(float, int);

float round_to_precision(float, const int&);

}
