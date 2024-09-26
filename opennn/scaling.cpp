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

void scale_minimum_maximum_binary(Tensor<type, 2>& matrix,
                                  const type& value_1,
                                  const type& value_2,
                                  const Index& raw_variable_index)
{
    const Index rows_number = matrix.dimension(0);

    type slope = type(0);
    type intercept = type(0);

    if(value_1 > value_2)
    {
        slope = type(1)/(value_1-value_2);
        intercept = -value_2/(value_1-value_2);
    }
    else
    {
        slope = type(1)/(value_2 - value_1);
        intercept = -value_1/(value_2-value_1);
    }

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        matrix(i, raw_variable_index) = slope*matrix(i, raw_variable_index)+intercept;
    }
}


void scale_mean_standard_deviation(Tensor<type, 2>& matrix,
                                   const Index& raw_variable_index,
                                   const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation) < type(NUMERIC_LIMITS_MIN)
            ? type(1)
            : type(1)/column_descriptives.standard_deviation;

    const type intercept = (column_descriptives.standard_deviation) < type(NUMERIC_LIMITS_MIN)
            ? type(0)
            : -type(1)*column_descriptives.mean/column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index) * slope + intercept;
    }
}


void scale_standard_deviation(Tensor<type, 2>& matrix,
                              const Index& raw_variable_index,
                              const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation) < type(NUMERIC_LIMITS_MIN)
            ? type(1)
            : type(1)/column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = (matrix(i, raw_variable_index)) * slope;
    }
}


void scale_minimum_maximum(Tensor<type, 2>& matrix,
                           const Index& raw_variable_index,
                           const Descriptives& column_descriptives,
                           const type& min_range, const type& max_range)
{
    const type slope = abs(column_descriptives.maximum-column_descriptives.minimum) < type(1e-3) ?
                       type(0) :
                       type(max_range-min_range)/(column_descriptives.maximum-column_descriptives.minimum);

    const type intercept = abs(column_descriptives.maximum-column_descriptives.minimum) < type(1e-3) ?
        type(0) :
                type(min_range*column_descriptives.maximum-max_range*column_descriptives.minimum)/(column_descriptives.maximum-column_descriptives.minimum);

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index)*slope + intercept;
    }
}


Tensor<type, 1> scale_minimum_maximum(const Tensor<type, 1>& x)
{
    const Tensor<type, 0> minimum = x.minimum();
    const Tensor<type, 0> maximum = x.maximum();

    const type min_range = type(-1);
    const type max_range = type(1);

    const type slope = (max_range-min_range)/(maximum()-minimum());
    const type intercept = (min_range*maximum()-max_range*minimum())/(maximum()-minimum());

    Tensor<type, 1> scaled_x(x.size());

    #pragma omp parallel for

    for(Index i = 0; i < scaled_x.size(); i++)
    {
        scaled_x(i) = slope*x(i)+intercept;
    }

    return scaled_x;
}

/*
void scale_image_minimum_maximum(Tensor<type, 2>& matrix, const Index& raw_variable_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) /= type(255);
    }
}
*/

Tensor<type, 2> scale_minimum_maximum(const Tensor<type, 2>& x)
{
    const Index rows_number = x.dimension(0);
    const Index raw_variables_number = x.dimension(1);

    Tensor<type, 2> scaled_x(rows_number, raw_variables_number);

    const Tensor<type, 1> columns_minimums = opennn::columns_minimums(x);

    const Tensor<type, 1> columns_maximums = opennn::columns_maximums(x);

    const type min_range = type(-1);
    const type max_range = type(1);

    #pragma omp parallel for

    for(Index j = 0; j < raw_variables_number; j++)
    {
        const type minimum = columns_minimums(j);
        const type maximum = columns_maximums(j);

        const type slope = (max_range-min_range)/(maximum - minimum);
        const type intercept = (min_range*maximum-max_range*minimum)/(maximum - minimum);

        for(Index i = 0; i < rows_number; i++)
        {
            scaled_x(i,j) = slope*x(i,j)+intercept;
        }
    }

    return scaled_x;
}


void scale_logarithmic(Tensor<type, 2>& matrix, const Index& raw_variable_index)
{
    type min_value = numeric_limits<type>::max();

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        if(!isnan(matrix(i,raw_variable_index)) && matrix(i,raw_variable_index) < min_value)
        {
            min_value = matrix(i,raw_variable_index);
        }
    }

    if(min_value <= type(0))
    {
        const type offset = abs(min_value) + type(1) + NUMERIC_LIMITS_MIN;

        for(Index i = 0; i < matrix.dimension(0); i++)
        {
            if(!isnan(matrix(i,raw_variable_index)))
            {
                matrix(i,raw_variable_index) += offset;
            }
        }
    }

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i,raw_variable_index) = log(matrix(i,raw_variable_index));
    }

}


void unscale_minimum_maximum(Tensor<type, 2>& matrix,
                             const Index& raw_variable_index,
                             const Descriptives& column_descriptives,
                             const type& min_range, const type& max_range)
{
    const type slope = abs(max_range-min_range) < type(1e-3)
            ? type(0)
            : (column_descriptives.maximum-column_descriptives.minimum)/type(max_range-min_range);

    const type intercept = abs(max_range-min_range) < type(1e-3)
            ? type(0)
            : -(min_range*column_descriptives.maximum-max_range*column_descriptives.minimum)/type(max_range-min_range);

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index)*slope + intercept;
    }
}


void unscale_mean_standard_deviation(Tensor<type, 2>& matrix, const Index& raw_variable_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.standard_deviation) < type(NUMERIC_LIMITS_MIN)
            ? type(0)
            : column_descriptives.standard_deviation;

    const type intercept = abs(column_descriptives.mean) < type(NUMERIC_LIMITS_MIN)
            ? type(0)
            : column_descriptives.mean;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index)*slope + intercept;
    }
}


void unscale_standard_deviation(Tensor<type, 2>& matrix, const Index& raw_variable_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.standard_deviation) < type(NUMERIC_LIMITS_MIN)
            ? type(0)
            : column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index) * slope;
    }
}


void unscale_logarithmic(Tensor<type, 2>& matrix, const Index& raw_variable_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) = exp(matrix(i, raw_variable_index));
    }
}

void unscale_image_minimum_maximum(Tensor<type, 2>& matrix, const Index& raw_variable_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, raw_variable_index) *= type(255);
    }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
