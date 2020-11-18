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


//using namespace std;
//using namespace Eigen;

namespace OpenNN
{

/// The enum RegressionMethod represents the different regression methods provided by OpenNN.

enum RegressionMethod{Linear, Logistic, Logarithmic, Exponential, Power, KarlPearson, OneWayAnova};

/// This structure provides the results obtained from the regression analysis.

struct RegressionResults
{
    explicit RegressionResults() {}

    virtual ~RegressionResults() {}

    string write_regression_type() const
    {
        switch(regression_type)
        {
            case Linear: return "linear";
            case Logistic: return "logistic";
            case Logarithmic: return "logarithmic";
            case Exponential: return "exponential";
            case Power: return "power";
            case KarlPearson: return "KarlPearson";
            case OneWayAnova: return "one-way-anova";
        }
    }

    /// Independent coefficient of the logistic function.

    type a = static_cast<type>(NAN);

    /// x coefficient of the logistic function.

    type b = static_cast<type>(NAN);

    /// Correlation coefficient of the  regression.

    type correlation =  static_cast<type>(NAN);

    /// Regression method type.

    RegressionMethod regression_type;
};


/// The enum CorrelationType represents the different correlations methods provided by OpenNN.

enum CorrelationType{Linear_correlation, Logistic_correlation, Logarithmic_correlation, Exponential_correlation, Power_correlation, KarlPearson_correlation, OneWayAnova_correlation, Gauss_correlation};


/// This structure provides the results obtained from the correlations.

struct CorrelationResults
{
    explicit CorrelationResults(){}

    virtual ~CorrelationResults() {}

    string write_correlation_type() const
    {
        switch(correlation_type)
        {
            case Linear_correlation: return "Linear";
            case Logistic_correlation: return "Logistic";
            case Logarithmic_correlation: return "Logarithmic";
            case Exponential_correlation: return "Exponential";
            case Power_correlation: return "Power";
            case KarlPearson_correlation: return "Karl-Pearson";
            case OneWayAnova_correlation: return "One-way Anova";
            case Gauss_correlation: return "Gauss";
        }

        return "";
    }

    /// Correlation coefficient.

    type correlation = static_cast<type>(NAN);

    /// Correlation type.

    CorrelationType correlation_type;
};
    // Linear

    type linear_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&, const bool& = true);

    // Rank linear

    type rank_linear_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);
    type rank_linear_correlation_missing_values(const ThreadPoolDevice*, const Tensor<type, 1>&x, const Tensor<type, 1>&);

    // Exponential

    type exponential_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Logarithmic

    type logarithmic_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Rank Logistic

    type rank_logistic_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Power

    type power_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Karl Pearson

    type karl_pearson_correlation(const ThreadPoolDevice*, const Tensor<type,2>&, const Tensor<type,2>&);

    // Time series correlation methods

    Tensor<type, 1> autocorrelations(const Tensor<type, 1>&, const Index & = 10);
    Tensor<type, 1> cross_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Index & = 10);

    // Logistic error methods

    type logistic(const type&, const type&, const type&);
    Tensor<type, 1> logistic(const type&, const type&, const Tensor<type, 1>&);

    Tensor<type, 2> logistic(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 2>&, const Tensor<type, 2>&);

    type logistic_error(const type&, const type&, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 1> logistic_error_gradient(const type&, const type&, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Regression methods

    RegressionResults linear_regression(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&, const bool& = true);

    RegressionResults logarithmic_regression(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults exponential_regression(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults power_regression(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults logistic_regression(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Correlation methods

    CorrelationResults linear_correlations(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults logarithmic_correlations(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults exponential_correlations(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults power_correlations(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults logistic_correlations(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults multiple_logistic_correlations(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

    CorrelationResults karl_pearson_correlations(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

    CorrelationResults gauss_correlations(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Covariance

    type covariance(const Tensor<type, 1>&, const Tensor<type, 1>&);
    type covariance_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 2> covariance_matrix(const Tensor<type, 2>&);

    Tensor<type, 1> less_rank_with_ties(const Tensor<type, 1>&);

    // Contingency tables

    Tensor<Index, 2> contingency_table(const Tensor<string, 1>&, const Tensor<string, 1>&);
    Tensor<Index, 2> contingency_table(Tensor<string, 2>&);
    Tensor<Index, 2> contingency_table(const Tensor<type, 2>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    type chi_square_test(const Tensor<type, 2>&);

    type chi_square_critical_point(const type&, const type&);


    // Missing values methods

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);
    pair<Tensor<type, 2>, Tensor<type, 2>> filter_missing_values(const Tensor<type, 2>&, const Tensor<type, 2>&);

    Index count_NAN(const Tensor<type, 1>&);

    // Other methods

    Tensor<type, 1> scale_minimum_maximum(const Tensor<type, 1>&);
    Tensor<type, 2> scale_minimum_maximum(const Tensor<type, 2>&);

    vector<int> get_indices_sorted(Tensor<type, 1>&);
}


#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
