#pragma once

#include "pch.h"

namespace opennn
{

void sort_string_vector(vector<string>&);
vector<string> concatenate_string_vectors(const vector<string>&, const vector<string>&);
string formatNumber(type, int);

type round_to_precision(type, const int&);

}
