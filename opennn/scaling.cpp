//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "statistics.h"
#include "scaling.h"

namespace opennn
{

void scale_mean_standard_deviation(MatrixMap matrix,
                                   Index column_index,
                                   const Descriptives& column_descriptives)
{
    if (column_descriptives.standard_deviation > EPSILON)
        matrix.col(column_index).array() = (matrix.col(column_index).array() - column_descriptives.mean) / column_descriptives.standard_deviation;
    else
        matrix.col(column_index).setZero();
}

void scale_standard_deviation(MatrixMap matrix,
                              Index column_index,
                              const Descriptives& column_descriptives)
{
    const float slope = (column_descriptives.standard_deviation > EPSILON)
        ? 1.0f / column_descriptives.standard_deviation
        : 0.0f;

    matrix.col(column_index) *= slope;
}

void scale_minimum_maximum(MatrixMap matrix,
                           Index column_index,
                           const Descriptives& column_descriptives,
                           float min_range,
                           float max_range)
{
    const float range = column_descriptives.maximum - column_descriptives.minimum;

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
    matrix.col(column_index).array() = matrix.col(column_index).array().max(EPSILON).log();
}

void unscale_minimum_maximum(MatrixMap matrix,
                             Index column_index,
                             const Descriptives& column_descriptives,
                             float min_range,
                             float max_range)
{
    throw_if(max_range - min_range < EPSILON,
             "The range values are not valid.");

    matrix.col(column_index).array() =
        (matrix.col(column_index).array() - min_range) / (max_range - min_range)
        * (column_descriptives.maximum - column_descriptives.minimum) + column_descriptives.minimum;
}

void unscale_mean_standard_deviation(MatrixMap matrix, Index column_index, const Descriptives& column_descriptives)
{
    if (column_descriptives.standard_deviation < EPSILON)
    {
        matrix.col(column_index).setConstant(column_descriptives.mean);
        return;
    }

    matrix.col(column_index).array()
        = column_descriptives.mean + matrix.col(column_index).array() * column_descriptives.standard_deviation;
}

void unscale_standard_deviation(MatrixMap matrix, Index column_index, const Descriptives& column_descriptives)
{
    const float slope = abs(column_descriptives.standard_deviation) < EPSILON
            ? 1.0f
            : column_descriptives.standard_deviation;

    matrix.col(column_index) *= slope;
}

void unscale_logarithmic(MatrixMap matrix, Index column_index)
{
    matrix.col(column_index).array() = matrix.col(column_index).array().exp();
}

void unscale_image_minimum_maximum(MatrixMap matrix, Index column_index)
{
    matrix.col(column_index) *= 255.0f;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
