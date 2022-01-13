//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   T E S T   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities_test.h"

TensorUtilitiesTest::TensorUtilitiesTest() : UnitTesting()
{
}


TensorUtilitiesTest::~TensorUtilitiesTest()
{
}


void TensorUtilitiesTest::test_fill_submatrix()
{
    cout << "test_fill_submatrix\n";

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

    fill_submatrix(matrix, rows_indices, columns_indices, submatrix.data());

    assert_true(is_equal(submatrix, type(3.1416)), LOG);
}


void TensorUtilitiesTest::test_calculate_rank()
{
    cout << "test_calculate_rank\n";

    Tensor<Index, 1> rank_greater;
    Tensor<Index, 1> rank_less;

    // Test

    vector.resize(3);
    vector.setValues({ type(4),type(2),type(3)});

    rank_greater = calculate_rank_greater(vector);
    rank_less = calculate_rank_less(vector);
}


void TensorUtilitiesTest::run_test_case()
{
    cout << "Running tensor utilities test case...\n";

    test_fill_submatrix();

    test_calculate_rank();

    cout << "End of tensor utilities test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
