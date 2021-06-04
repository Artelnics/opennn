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

// OpenNN includes

#include "statistics.h"
#include "config.h"

namespace OpenNN
{

/// The enum CorrelationMethod represents the different regression methods provided by OpenNN.

enum CorrelationMethod{Linear, Logistic, Logarithmic, Exponential, Power, KarlPearson, OneWayAnova};

/// This structure provides the results obtained from the regression analysis.

struct Correlation
{
    explicit Correlation() {}

    virtual ~Correlation() {}

    string write_regression_type() const
    {
        switch(correlation_type)
        {
            case Linear: return "linear";
            case Logistic: return "logistic";
            case Logarithmic: return "logarithmic";
            case Exponential: return "exponential";
            case Power: return "power";
            case KarlPearson: return "KarlPearson";
            case OneWayAnova: return "one-way-anova";
        }

        return string();
    }

    /// Independent coefficient of the logistic function.

    type a = static_cast<type>(NAN);

    /// x coefficient of the logistic function.

    type b = static_cast<type>(NAN);

    /// Correlation coefficient of the  regression.

    type r =  static_cast<type>(NAN);

    /// Regression method type.

    CorrelationMethod correlation_type;
};

    // Correlation methods

    Correlation linear_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&, const bool& = true);

    Correlation logarithmic_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation exponential_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation power_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation logistic_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Correlation multiple_logistic_correlation(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

    // Time series correlation methods

    Tensor<type, 1> autocorrelations(const ThreadPoolDevice*,
                                     const Tensor<type, 1>&,
                                     const Index&  = 10);

    Tensor<type, 1> cross_correlations(const ThreadPoolDevice*,
                                       const Tensor<type, 1>&,
                                       const Tensor<type, 1>&,
                                       const Index&);

    // Logistic function methods

    Tensor<type, 1> logistic(const type&, const type&, const Tensor<type, 1>&);

    // Missing values methods

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);
    pair<Tensor<type, 2>, Tensor<type, 2>> filter_missing_values(const Tensor<type, 2>&, const Tensor<type, 2>&);

}


#endif


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
