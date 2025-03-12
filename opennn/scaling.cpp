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
                                   const Index& column_index,
                                   const Descriptives& column_descriptives)
{
    const type mean = column_descriptives.mean;
    const type standard_deviation = column_descriptives.standard_deviation;

    // if(abs(standard_deviation) < NUMERIC_LIMITS_MIN)
    //     throw runtime_error("Standard deviation is zero.");

    #pragma omp parallel for
    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = (matrix(i, column_index) - mean)/standard_deviation;
}


void scale_standard_deviation(Tensor<type, 2>& matrix,
                              const Index& column_index,
                              const Descriptives& column_descriptives)
{
    const type slope = (column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN
            ? type(1)
            : type(1)/column_descriptives.standard_deviation;

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) = (matrix(i, column_index)) * slope;
    }
}


void scale_minimum_maximum(Tensor<type, 2>& matrix,
                           const Index& column_index,
                           const Descriptives& column_descriptives,
                           const type& min_range,
                           const type& max_range)
{

    const type minimum = column_descriptives.minimum;
    const type maximum = column_descriptives.maximum;

    if(abs(maximum - minimum) < NUMERIC_LIMITS_MIN)
        //throw runtime_error("The difference between the maximum and minimum value is too small.");

    if(max_range - min_range < NUMERIC_LIMITS_MIN)
        throw runtime_error("The range values are not valid.");

    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
        matrix(i, column_index) = (matrix(i, column_index) - minimum) / (maximum - minimum);
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

    if(abs(column_descriptives.standard_deviation) < NUMERIC_LIMITS_MIN)
        return;
        // throw runtime_error("Standard deviation is zero.");

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
    {
        matrix(i, column_index) = exp(matrix(i, column_index));
    }
}

void unscale_image_minimum_maximum(Tensor<type, 2>& matrix, const Index& column_index)
{
    #pragma omp parallel for

    for(Index i = 0; i < matrix.dimension(0); i++)
    {
        matrix(i, column_index) *= type(255);
    }
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
