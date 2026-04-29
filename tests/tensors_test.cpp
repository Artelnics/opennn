#include "pch.h"

#include "../opennn/tensor_utilities.h"
using namespace opennn;

TEST(Tensors, Fill)
{
    MatrixR submatrix;

    vector<Index> rows_indices;
    vector<Index> columns_indices;

    MatrixR matrix(1, 1);
    matrix.setConstant(type(3.1416));

    rows_indices.resize(1, 0);

    columns_indices.resize(1, 0);

    submatrix.resize(1, 1);

    fill_tensor_data(matrix, rows_indices, columns_indices, submatrix.data());

    EXPECT_LT((submatrix.array() - type(3.1416)).abs().maxCoeff(), type(1e-6));
}




