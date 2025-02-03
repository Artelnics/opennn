//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "pch.h"
#include "statistics.h"
#include "scaling.h"

namespace opennn
{
// void scale_minimum_maximum_binary(Tensor<type, 2>& matrix,
//                                   const type& value_1,
//                                   const type& value_2,
//                                   const Index& raw_variable_index)
// {
//     const Index rows_number = matrix.dimension(0);

//     type slope = type(0);
//     type intercept = type(0);

//     if(value_1 > value_2)
//     {
//         slope = type(1)/(value_1-value_2);
//         intercept = -value_2/(value_1-value_2);
//     }
//     else
//     {
//         slope = type(1)/(value_2 - value_1);
//         intercept = -value_1/(value_2-value_1);
//     }

//     #pragma omp parallel for

//     for(Index i = 0; i < rows_number; i++)
//         matrix(i, raw_variable_index) = slope*matrix(i, raw_variable_index)+intercept;
// }

    string scaler_to_string(const Scaler& scaler)
    {
        switch (scaler)
        {
        case Scaler::None:
            return "None";
        case Scaler::MinimumMaximum:
            return "MinimumMaximum";
        case Scaler::MeanStandardDeviation:
            return "MeanStandardDeviation";
        case Scaler::StandardDeviation:
            return "StandardDeviation";
        case Scaler::Logarithm:
            return "Logarithm";
        case Scaler::ImageMinMax:
            return "ImageMinMax";
        default:
            throw runtime_error("Unknown scaler\n");
        }
    }


    Scaler string_to_scaler(const string& new_scaler)
    {
        if (new_scaler == "None")
            return Scaler::None;
        else if (new_scaler == "MinimumMaximum")
            return Scaler::MinimumMaximum;
        else if (new_scaler == "MeanStandardDeviation")
            return Scaler::MeanStandardDeviation;
        else if (new_scaler == "StandardDeviation")
            return Scaler::StandardDeviation;
        else if (new_scaler == "Logarithm")
            return Scaler::Logarithm;
        else if (new_scaler == "ImageMinMax")
            return Scaler::ImageMinMax;
        else
            throw runtime_error("Unknown scaler: " + new_scaler + "\n");

    }

    void scale_mean_standard_deviation(Tensor<type, 2>& matrix,
                                   const Index& raw_variable_index,
                                   const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
            ? type(1)
            : type(1)/column_descriptives.standard_deviation;

    const type intercept = (column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
            ? type(0)
            : -type(1)*column_descriptives.mean/column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index) * slope + intercept;
}


void scale_standard_deviation(Tensor<type, 2>& matrix,
                              const Index& raw_variable_index,
                              const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
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


void scale_logarithmic(Tensor<type, 2>& matrix, const Index& raw_variable_index)
{
    type min_value = numeric_limits<type>::max();

    for(Index i = 0; i < matrix.dimension(0); i++)
        if(!isnan(matrix(i,raw_variable_index)) && matrix(i,raw_variable_index) < min_value)
            min_value = matrix(i,raw_variable_index);

    if(min_value <= type(0))
    {
        const type offset = abs(min_value) + type(1) + NUMERIC_LIMITS_MIN;

        for(Index i = 0; i < matrix.dimension(0); i++)
            if(!isnan(matrix(i,raw_variable_index)))
                matrix(i,raw_variable_index) += offset;
    }

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i,raw_variable_index) = log(matrix(i,raw_variable_index));
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
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index)*slope + intercept;
}


void unscale_mean_standard_deviation(Tensor<type, 2>& matrix, const Index& raw_variable_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
            ? type(0)
            : column_descriptives.standard_deviation;

    const type intercept = abs(column_descriptives.mean) < NUMERIC_LIMITS_MIN
            ? type(0)
            : column_descriptives.mean;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index)*slope + intercept;
}


void unscale_standard_deviation(Tensor<type, 2>& matrix, const Index& raw_variable_index, const Descriptives& column_descriptives)
{
    const type slope = abs(column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
            ? type(0)
            : column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, raw_variable_index) = matrix(i, raw_variable_index) * slope;
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
