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


void scale_mean_standard_deviation(Tensor<type, 2>& matrix,
                                   const Index& column_index,
                                   const Descriptives& column_descriptives)
{
    const type mean = column_descriptives.mean;
    const type standard_deviation = column_descriptives.standard_deviation;

    if(standard_deviation > NUMERIC_LIMITS_MIN)
    {
        #pragma omp parallel for
        for(Index i = 0; i < matrix.dimension(0); i++)
            matrix(i, column_index) = (matrix(i, column_index) - mean) / standard_deviation;
    }
    else
    {
        #pragma omp parallel for
        for(Index i = 0; i < matrix.dimension(0); i++)
            matrix(i, column_index) = type(0);
    }
}


void scale_standard_deviation(Tensor<type, 2>& matrix,
                              const Index& column_index,
                              const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation > NUMERIC_LIMITS_MIN)
        ? type(1) / column_descriptives.standard_deviation
        : type(0);

    #pragma omp parallel for
    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = (matrix(i, column_index)) * slope;
}


void scale_minimum_maximum(Tensor<type, 2>& matrix,
                           const Index& column_index,
                           const Descriptives& column_descriptives,
                           const type& min_range,
                           const type& max_range)
{
    const type minimum = column_descriptives.minimum;
    const type maximum = column_descriptives.maximum;
    const type range = maximum - minimum;

    if(max_range - min_range < NUMERIC_LIMITS_MIN)
        throw runtime_error("The range values for scaling are not valid.");

    if(range > NUMERIC_LIMITS_MIN)
    {
        #pragma omp parallel for
        for(Index i = 0; i < matrix.dimension(0); i++)
            matrix(i, column_index) = (matrix(i, column_index) - minimum) / range;
    }
    else
    {
        #pragma omp parallel for
        for(Index i = 0; i < matrix.dimension(0); i++)
            matrix(i, column_index) = type(0);
    }
}


void scale_logarithmic(Tensor<type, 2>& matrix, const Index& column_index)
{
    type min_value = numeric_limits<type>::max();

    for(Index i = 0; i < matrix.dimension(0); i++)
        if(!isnan(matrix(i, column_index)) && matrix(i,column_index) < min_value)
            min_value = matrix(i,column_index);

    if(min_value <= type(0))
    {
        const type offset = abs(min_value) + type(1) + NUMERIC_LIMITS_MIN;

        for(Index i = 0; i < matrix.dimension(0); i++)
            if(!isnan(matrix(i,column_index)))
                matrix(i, column_index) += offset;
    }

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = log(matrix(i, column_index));
}

void scale_mean_standard_deviation_3d(Tensor<type, 3>& tensor,
                                      const Index& feature_index,
                                      const Descriptives& feature_descriptives)
{
    const type mean = feature_descriptives.mean;
    const type standard_deviation = feature_descriptives.standard_deviation;
    const type epsilon = type(1e-7);

    const Index batch_size = tensor.dimension(0);
    const Index time_steps = tensor.dimension(1);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; b++)
        for(Index t = 0; t < time_steps; t++)
            tensor(b, t, feature_index) = (tensor(b, t, feature_index) - mean) / (standard_deviation + epsilon);
}


void scale_standard_deviation_3d(Tensor<type, 3>& tensor,
                                 const Index& feature_index,
                                 const Descriptives& feature_descriptives)
{
    const type slope = (feature_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
                           ? type(1)
                           : type(1) / feature_descriptives.standard_deviation;

    const Index batch_size = tensor.dimension(0);
    const Index time_steps = tensor.dimension(1);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; b++)
        for(Index t = 0; t < time_steps; t++)
            tensor(b, t, feature_index) *= slope;
}


void scale_minimum_maximum_3d(Tensor<type, 3>& tensor,
                              const Index& feature_index,
                              const Descriptives& feature_descriptives,
                              const type& min_range,
                              const type& max_range)
{
    const type minimum = feature_descriptives.minimum;
    const type maximum = feature_descriptives.maximum;
    const type range_diff = maximum - minimum;

    if(max_range - min_range < NUMERIC_LIMITS_MIN)
        throw runtime_error("The range values are not valid.");

    const Index batch_size = tensor.dimension(0);
    const Index time_steps = tensor.dimension(1);

    if(range_diff > NUMERIC_LIMITS_MIN)
    {
        #pragma omp parallel for
        for(Index b = 0; b < batch_size; b++)
            for(Index t = 0; t < time_steps; t++)
            {
                type scaled_01 = (tensor(b, t, feature_index) - minimum) / range_diff;
                tensor(b, t, feature_index) = scaled_01 * (max_range - min_range) + min_range;
            }
    }
    else
    {
        const type mid_range = min_range + (max_range - min_range) / 2.0;

        #pragma omp parallel for
        for(Index b = 0; b < batch_size; b++)
            for(Index t = 0; t < time_steps; t++)
                tensor(b, t, feature_index) = mid_range;
    }
}


void scale_logarithmic_3d(Tensor<type, 3>& tensor, const Index& feature_index)
{
    type min_value = numeric_limits<type>::max();
    const Index batch_size = tensor.dimension(0);
    const Index time_steps = tensor.dimension(1);

    for(Index b = 0; b < batch_size; b++)
        for(Index t = 0; t < time_steps; t++)
            if(!isnan(tensor(b, t, feature_index)) && tensor(b, t, feature_index) < min_value)
                min_value = tensor(b, t, feature_index);

    if(min_value <= type(0))
    {
        const type offset = abs(min_value) + type(1) + NUMERIC_LIMITS_MIN;
        for(Index b = 0; b < batch_size; b++)
            for(Index t = 0; t < time_steps; t++)
                    tensor(b, t, feature_index) += offset;
    }

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; b++)
        for(Index t = 0; t < time_steps; t++)
            tensor(b, t, feature_index) = log(tensor(b, t, feature_index));
}


void unscale_minimum_maximum(Tensor<type, 2>& matrix,
                             const Index& column_index,
                             const Descriptives& column_descriptives,
                             const type& min_range,
                             const type& max_range)
{
    const type minimum = column_descriptives.minimum;
    const type maximum = column_descriptives.maximum;

    if(max_range - min_range < NUMERIC_LIMITS_MIN)
        throw runtime_error("The range values are not valid.");

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i,column_index) = (matrix(i, column_index)*(maximum-minimum)+minimum);
}


void unscale_mean_standard_deviation(Tensor<type, 2>& matrix, const Index& column_index, const Descriptives& column_descriptives)
{
    const type mean = column_descriptives.mean;
    type standard_deviation = column_descriptives.standard_deviation;

    if(standard_deviation < NUMERIC_LIMITS_MIN)
    {
        #pragma omp parallel for
        for(Index i = 0; i < matrix.dimension(0); i++)
            matrix(i, column_index) = mean;

        return;
    }

    #pragma omp parallel for
    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = mean + matrix(i, column_index)*standard_deviation;
}


void unscale_standard_deviation(Tensor<type, 2>& matrix, const Index& column_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
            ? type(1)
            : column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = matrix(i, column_index) * slope;
}


void unscale_logarithmic(Tensor<type, 2>& matrix, const Index& column_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = exp(matrix(i, column_index));
}


void unscale_image_minimum_maximum(Tensor<type, 2>& matrix, const Index& column_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) *= type(255);
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
