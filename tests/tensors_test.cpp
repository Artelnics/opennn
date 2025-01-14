#include "pch.h"

#include "../opennn/tensors.h"

TEST(Tensors, Fill)
{
    Tensor<type, 2> submatrix;

    vector<Index> rows_indices;
    vector<Index> columns_indices;

    Tensor<type, 2> matrix(1, 1);
    matrix.setConstant(type(3.1416));

    rows_indices.resize(1, 0);

    columns_indices.resize(1, 0);

    submatrix.resize(1, 1);

//    fill_tensor_data(matrix, rows_indices, columns_indices, submatrix.data());

//    EXPECT_EQ(is_equal(submatrix, type(3.1416)), true);

}




