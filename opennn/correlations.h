//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CORRELATIONS_H
#define CORRELATIONS_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>
#include <algorithm>

// OpenNN includes

#include "statistics.h"
#include "config.h"

namespace opennn
{

enum class CorrelationMethod{Pearson, Spearman};

/// This enumeration represents the different regression methods provided by OpenNN.

enum class CorrelationType{Linear, Logistic, Logarithmic, Exponential, Power};


/// This structure provides the results obtained from the regression analysis.

struct Correlation
{
    explicit Correlation() {}

    string write_correlation_method() const
    {
        switch(correlation_type)
        {
        case CorrelationType::Linear: return "linear";
        case CorrelationType::Logistic: return "logistic";
        case CorrelationType::Logarithmic: return "logarithmic";
        case CorrelationType::Exponential: return "exponential";
        case CorrelationType::Power: return "power";
        default:
            return string();
        }
    }

    void print() const
    {
        cout << "Correlation" << endl;
        cout << "Type: " << write_correlation_method() << endl;
        cout << "a: " << a << endl;
        cout << "b: " << b << endl;
        cout << "r: " << r << endl;
        cout << "Lower confidence: " << lower_confidence << endl;
        cout << "Upper confidence: " << upper_confidence << endl;
    }

    /// Independent coefficient of the regression function.

    type a = static_cast<type>(NAN);

    /// x coefficient of the regression function.

    type b = static_cast<type>(NAN);

    /// Correlation coefficient of the regression.

    type r = static_cast<type>(NAN);

    type lower_confidence = static_cast<type>(NAN);
    type upper_confidence = static_cast<type>(NAN);

    /// Regression method type

    CorrelationMethod correlation_method = CorrelationMethod::Pearson;
    CorrelationType correlation_type = CorrelationType::Linear;

};


    // Pearson correlation methods

    Correlation linear_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation logarithmic_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation exponential_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation power_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation logistic_correlation_vector_vector(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation logistic_correlation_vector_matrix(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 2>&);

    Correlation logistic_correlation_matrix_vector(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 1>&);

    Correlation logistic_correlation_matrix_matrix(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

    Correlation correlation(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

    // Spearman correlation methods

    Correlation linear_correlation_spearman(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);
    Tensor<type, 1> calculate_spearman_ranks(const Tensor<type, 1>&);

    Correlation logistic_correlation_vector_vector_spearman(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation correlation_spearman(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

    // Confidence interval

    type r_correlation_to_z_correlation(const type&);
    type z_correlation_to_r_correlation(const type&);

    Tensor<type,1> confidence_interval_z_correlation(const type&, const Index&);

    template<typename T>
    const T& clamp(const T& value, const T& min, const T& max) {
        if (value < min) {
            return min;
        } else if (value > max) {
            return max;
        } else {
            return value;
        }
    }


    // Time series correlation methods

    Tensor<type, 1> autocorrelations(const ThreadPoolDevice*,
                                     const Tensor<type, 1>&,
                                     const Index&  = 10);

    Tensor<type, 1> cross_correlations(const ThreadPoolDevice*,
                                       const Tensor<type, 1>&,
                                       const Tensor<type, 1>&,
                                       const Index&);

    Tensor<type, 2> get_correlation_values(const Tensor<Correlation, 2>&);

    // Missing values methods

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
    pair<Tensor<type, 1>, Tensor<type, 2>> filter_missing_values_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
    pair<Tensor<type, 1>, Tensor<type, 2>> filter_missing_values_matrix_vector(const Tensor<type, 2>&, const Tensor<type, 1>&);
    pair<Tensor<type, 2>, Tensor<type, 2>> filter_missing_values_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);
}


#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
