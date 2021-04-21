//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "scaling.h"

namespace OpenNN
{

void scale_minimum_maximum_binary(Tensor<type, 2>& matrix,
                                  const type& value_1,
                                  const type& value_2,
                                  const Index& column_index)
{
    const Index rows_number = matrix.dimension(0);

    type slope = 0;
    type intercept = 0;

    if(value_1 > value_2)
    {
        slope = 1/(value_1-value_2);
        intercept = -value_2/(value_1-value_2);
    }
    else
    {
        slope = 1/(value_2 - value_1);

        intercept = -value_1/(value_2-value_1);
    }

    for(Index i = 0; i < rows_number; i++)
    {
        matrix(i, column_index) = slope*matrix(i, column_index)+intercept;
    }
}


/// Scales the given input variables with given mean and standard deviation values.
/// It updates the input variable of the matrix matrix.
/// @param column_descriptives vector of descriptives structures for the input variables.
/// @param column_index Index of the input to be scaled.

void scale_mean_standard_deviation(Tensor<type, 2>& matrix,
                                   const Index& column_index,
                                   const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation -0) < static_cast<type>(1e-3) ?
                0 :
                static_cast<type>(1)/column_descriptives.standard_deviation;

    const type intercept = (column_descriptives.standard_deviation -0) < static_cast<type>(1e-3) ?
                0 :
                -static_cast<type>(1)*column_descriptives.mean/column_descriptives.standard_deviation;

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = matrix(i, column_index)*slope + intercept;
    }
}


/// Scales the given input variables with given standard deviation values.
/// It updates the input variable of the matrix matrix.
/// @param inputs_statistics vector of descriptives structures for the input variables.
/// @param column_index Index of the input to be scaled.

void scale_standard_deviation(Tensor<type, 2>& matrix,
                                     const Index& column_index,
                                     const Descriptives& column_descriptives)
{
    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = static_cast<type>(2)*(matrix(i, column_index)) / column_descriptives.standard_deviation;
    }
}


/// Scales the given input variable with given minimum and maximum values.
/// It updates the input variables of the matrix matrix.
/// @param column_descriptives vector with the descriptives of the input variable.
/// @param column_index Index of the input to be scaled.

void scale_minimum_maximum(Tensor<type, 2>& matrix,
                           const Index& column_index,
                           const Descriptives& column_descriptives,
                           const Index& min_range, const Index& max_range)
{
    const type slope = abs(column_descriptives.maximum-column_descriptives.minimum) < static_cast<type>(1e-3) ?
                0 :
                (max_range-min_range)/(column_descriptives.maximum-column_descriptives.minimum);

    const type intercept = abs(column_descriptives.maximum-column_descriptives.minimum) < static_cast<type>(1e-3) ?
                0 :
                (min_range*column_descriptives.maximum-max_range*column_descriptives.minimum)/(column_descriptives.maximum-column_descriptives.minimum);

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = matrix(i, column_index)*slope + intercept;
    }

}

void scale_logarithmic(Tensor<type, 2>& matrix, const Index& target_index, const Descriptives& descriptives)
{
    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        if(abs(descriptives.standard_deviation-0) < static_cast<type>(1e-3))
        {
            matrix(i, target_index) = 0;
        }
        else
        {
            matrix(i, target_index)
                    = static_cast<type>(0.5)*(exp(matrix(i,target_index)-1))*(descriptives.maximum-descriptives.minimum)
                    + descriptives.minimum;
        }
    }
}


/// Unscales the given input variable with given minimum and maximum values.
/// It updates the input variables of the matrix matrix.
/// @param column_descriptives vector with the descriptives of the input variable.
/// @param column_index Index of the input to be scaled.

void unscale_minimum_maximum(Tensor<type, 2>& matrix,
                             const Index& column_index,
                             const Descriptives& column_descriptives,
                             const Index& min_range, const Index& max_range)
{
    const type slope = abs(max_range-min_range) < static_cast<type>(1e-3)
            ? 0
            : (column_descriptives.maximum-column_descriptives.minimum)/(max_range-min_range);

    const type intercept = abs(max_range-min_range) < static_cast<type>(1e-3)
            ? 0
            : -(min_range*column_descriptives.maximum-max_range*column_descriptives.minimum)/(max_range-min_range);

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = matrix(i, column_index)*slope + intercept;
    }
}


/// Uncales the given input variables with given mean and standard deviation values.
/// It updates the input variable of the matrix matrix.
/// @param column_descriptives vector of descriptives structures for the input variables.
/// @param column_index Index of the input to be scaled.

void unscale_mean_standard_deviation(Tensor<type, 2>& matrix, const Index& column_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.mean - 0) < static_cast<type>(1e-3) ? 0
            : column_descriptives.standard_deviation;

    const type intercept = abs(column_descriptives.mean-0) < static_cast<type>(1e-3)
            ? column_descriptives.minimum
            : column_descriptives.mean;

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = matrix(i, column_index)*slope + intercept;
    }
}


/// Unscales the given input variables with given standard deviation values.
/// It updates the input variable of the matrix matrix.
/// @param inputs_statistics vector of descriptives structures for the input variables.
/// @param column_index Index of the input to be scaled.

void unscale_standard_deviation(Tensor<type, 2>& matrix, const Index& column_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.mean-0) < static_cast<type>(1e-3)
            ? 0
            : column_descriptives.standard_deviation/static_cast<type>(2);

    const type intercept = abs(column_descriptives.mean-0) < static_cast<type>(1e-3)
            ? column_descriptives.minimum
            : 0;

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = matrix(i, column_index)*slope + intercept;
    }
}


/// Unscales the given input variables with given logarithmic values.
/// It updates the input variable of the matrix matrix.
/// @param inputs_statistics vector of descriptives structures for the input variables.
/// @param column_index Index of the input to be scaled.
///
void unscale_logarithmic(Tensor<type, 2>& matrix, const Index& target_index, const Descriptives& target_statistics)
{
    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        if(abs(target_statistics.maximum - target_statistics.minimum) < static_cast<type>(1e-3))
        {
            matrix(i, target_index) = target_statistics.minimum;
        }
        else
        {
            matrix(i, target_index) = log(static_cast<type>(2)*(matrix(i,target_index)-target_statistics.minimum)/(target_statistics.maximum-target_statistics.minimum));
        }
    }
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
