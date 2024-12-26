#include "pch.h"

#include "../opennn/tensors.h"

/*

void TensorsTest::test_fill_tensor_data()
{
    Tensor<type, 2> submatrix;

    Tensor<Index, 1> rows_indices;
    Tensor<Index, 1> columns_indices;

    // Test

    matrix.resize(1, 1);
    matrix.setConstant(type(3.1416));

    rows_indices.resize(1);
    rows_indices.setZero();

    columns_indices.resize(1);
    columns_indices.setZero();

    submatrix.resize(1, 1);

    fill_tensor_data(matrix, rows_indices, columns_indices, submatrix.data());

    EXPECT_EQ(is_equal(submatrix, type(3.1416)));
}


void TensorsTest::test_calculate_rank()
{
    Tensor<Index, 1> rank_greater;
    Tensor<Index, 1> rank_less;

    // Test

    vector.resize(3);
    vector.setValues({type(4),type(2),type(3)});

    rank_greater = calculate_rank_greater(vector);
    rank_less = calculate_rank_less(vector);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
*/