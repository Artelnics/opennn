
#ifndef TENSORUTILITIES_H
#define TENSORUTILITIES_H

// System includes

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>

// OpenNN includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace OpenNN
{

void initialize_sequential(Tensor<type, 1>&);

void multiply_rows(Tensor<type, 2>&, const Tensor<type, 1>&);

bool is_zero(const Tensor<type, 1>&);



}

#endif
