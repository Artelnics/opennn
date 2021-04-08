//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"

namespace OpenNN
{

void initialize_sequential(Tensor<type, 1>& vector)
{
    for(Index i = 0; i < vector.size(); i++) vector(i) = i;
}


void multiply_rows(Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index columns_number = matrix.dimension(1);
    const Index rows_number = matrix.dimension(0);

//    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
           matrix(i,j) *= vector(j);
        }
    }
}


bool is_zero(const Tensor<type, 1>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(tensor[i]) > numeric_limits<type>::min()) return false;
    }

    return true;
}


bool is_false(const Tensor<bool, 1>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if(tensor(i)) return false;
    }

    return true;
}


bool is_equal(const Tensor<type, 2>& matrix, const type& value, const type& tolerance)
{
    const Index size = matrix.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(matrix(i) - value) > tolerance) return false;
    }

    return true;
}


bool are_equal(const Tensor<type, 1>& vector_1, const Tensor<type, 1>& vector_2, const type& tolerance)
{
    const Index size = vector_1.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(vector_1(i) - vector_2(i)) > tolerance) return false;
    }

    return true;
}


bool are_equal(const Tensor<type, 2>& matrix_1, const Tensor<type, 2>& matrix_2, const type& tolerance)
{
    const Index size = matrix_1.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(matrix_1(i) - matrix_2(i)) > tolerance) return false;
    }

    return true;
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
