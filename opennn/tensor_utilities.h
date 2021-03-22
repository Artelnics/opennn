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
/*
void multiply_rows(Tensor<type, 2> & matrix, const Tensor<type, 1> & vector)
{
    const Index columns_number = matrix.dimension(1);
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
           matrix(i,j) *= vector(j);
        }
    }
}
*/
};

#endif
