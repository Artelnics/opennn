
#ifndef TENSORUTILITIES_H
#define TENSORUTILITIES_H

// System includes

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
#include <numeric>

// OpenNN includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace OpenNN
{

void initialize_sequential(Tensor<type, 1>&);

void multiply_rows(Tensor<type, 2>&, const Tensor<type, 1>&);
void divide_columns(Tensor<type, 2>&, const Tensor<type, 1>&);

bool is_zero(const Tensor<type, 1>&);

bool is_equal(const Tensor<type, 2>&, const type&, const type& = 0.0);

bool are_equal(const Tensor<type, 1>&, const Tensor<type, 1>&, const type& = 0.0);
bool are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&, const type& = 0.0);

bool is_false(const Tensor<bool, 1>& tensor);

void save_csv(const Tensor<type,2>&, const string&);

Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>&);
Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>&);


}

#endif
