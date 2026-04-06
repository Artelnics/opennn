//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "statistics.h"
#include "scaling.h"

namespace opennn
{


void scale_mean_standard_deviation(MatrixMap matrix,
                                   Index column_index,
                                   const Descriptives& column_descriptives)
{
    if(column_descriptives.standard_deviation > EPSILON)
        matrix.col(column_index).array() = (matrix.col(column_index).array() - column_descriptives.mean) / column_descriptives.standard_deviation;
    else
        matrix.col(column_index).setZero();
}


void scale_standard_deviation(MatrixMap matrix,
                              Index column_index,
                              const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation > EPSILON)
        ? type(1) / column_descriptives.standard_deviation
        : type(0);

    matrix.col(column_index) *= slope;
}


void scale_minimum_maximum(MatrixMap matrix,
                           Index column_index,
                           const Descriptives& column_descriptives,
                           type min_range,
                           type max_range)
{
    const type range = column_descriptives.maximum - column_descriptives.minimum;

    if (range < EPSILON)
    {
        matrix.col(column_index).setZero();
        return;
    }

    matrix.col(column_index).array() =
        (matrix.col(column_index).array() - column_descriptives.minimum) / range * (max_range - min_range) + min_range;
}


void scale_logarithmic(MatrixMap matrix, Index column_index)
{
    auto col = matrix.col(column_index).array();

    const type min_val = (col.isFinite()).select(col, MAX).minCoeff();

    if (min_val <= 0)
    {
        const type offset = abs(min_val) + 1.0 + EPSILON;

        col = (col.isNaN()).select(col, col + offset);
    }

    col = col.log();
}


void scale_mean_standard_deviation(MatrixR& matrix, Index column_index, const Descriptives& column_descriptives)
{
    // Create a Map view of the Tensor and call the implementation
    MatrixMap map(matrix.data(), matrix.rows(), matrix.cols());
    scale_mean_standard_deviation(map, column_index, column_descriptives);
}


void scale_minimum_maximum(MatrixR& matrix, Index column_index, const Descriptives& column_descriptives, type min_range, type max_range)
{
    MatrixMap map(matrix.data(), matrix.rows(), matrix.cols());
    scale_minimum_maximum(map, column_index, column_descriptives, min_range, max_range);
}

void scale_logarithmic(MatrixR& matrix, Index column_index)
{
    MatrixMap map(matrix.data(), matrix.rows(), matrix.cols());
    scale_logarithmic(map, column_index);
}


void scale_standard_deviation(MatrixR& matrix, Index column_index, const Descriptives& column_descriptives)
{
    MatrixMap map(matrix.data(), matrix.rows(), matrix.cols());
    scale_standard_deviation(map, column_index, column_descriptives);
}


void unscale_minimum_maximum(MatrixMap matrix,
                             Index column_index,
                             const Descriptives& column_descriptives,
                             type min_range,
                             type max_range)
{
    const type minimum = column_descriptives.minimum;
    const type maximum = column_descriptives.maximum;

    if(max_range - min_range < EPSILON)
        throw runtime_error("The range values are not valid.");

    #pragma omp parallel for
    for(Index i = 0; i < matrix.rows(); i++)
    {
        type normalized = (matrix(i, column_index) - min_range) / (max_range - min_range);
        matrix(i,column_index) = normalized * (maximum - minimum) + minimum;
    }
}


void unscale_mean_standard_deviation(MatrixMap matrix, Index column_index, const Descriptives& column_descriptives)
{
    const type mean = column_descriptives.mean;
    type standard_deviation = column_descriptives.standard_deviation;

    if(standard_deviation < EPSILON)
    {
        #pragma omp parallel for
        for(Index i = 0; i < matrix.rows(); i++)
            matrix(i, column_index) = mean;

        return;
    }

    #pragma omp parallel for
    for(Index i = 0; i < matrix.rows(); i++)
        matrix(i, column_index) = mean + matrix(i, column_index)*standard_deviation;
}


void unscale_standard_deviation(MatrixMap matrix, Index column_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.standard_deviation) < EPSILON
            ? type(1)
            : column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.rows(); i++)
        matrix(i, column_index) = matrix(i, column_index) * slope;
}


void unscale_logarithmic(MatrixMap matrix, Index column_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.rows(); i++)
        matrix(i, column_index) = exp(matrix(i, column_index));
}


void unscale_image_minimum_maximum(MatrixMap matrix, Index column_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.rows(); i++)
        matrix(i, column_index) *= type(255);
}


void unscale_minimum_maximum(MatrixR& matrix,
                             Index column_index,
                             const Descriptives& column_descriptives,
                             type min_range,
                             type max_range)
{
    MatrixMap map(matrix.data(), matrix.rows(), matrix.cols());
    unscale_minimum_maximum(map, column_index, column_descriptives, min_range, max_range);
}


void unscale_mean_standard_deviation(MatrixR& matrix, Index column_index, const Descriptives& column_descriptives)
{
    unscale_mean_standard_deviation(MatrixMap(matrix.data(), matrix.rows(), matrix.cols()), column_index, column_descriptives);
}


void unscale_standard_deviation(MatrixR& matrix, Index column_index, const Descriptives& descriptives)
{
    unscale_standard_deviation(MatrixMap(matrix.data(), matrix.rows(), matrix.cols()), column_index, descriptives);
}


void unscale_logarithmic(MatrixR& matrix, Index column_index)
{
    unscale_logarithmic(MatrixMap(matrix.data(), matrix.rows(), matrix.cols()), column_index);
}


void unscale_image_minimum_maximum(MatrixR& matrix, Index column_index)
{
    unscale_image_minimum_maximum(MatrixMap(matrix.data(), matrix.rows(), matrix.cols()), column_index);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
