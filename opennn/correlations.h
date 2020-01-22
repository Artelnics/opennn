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
#include "transformations.h"
#include "config.h"

#include "../eigen/Eigen/Eigen"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"



using namespace std;
using namespace Eigen;

namespace OpenNN
{

/// This class represents the space of correlations and regression analysis.


/// The enum RegressionMethod represents the different regression methods provided by OpenNN.

enum RegressionMethod{Linear, Logistic, Logarithmic, Exponential, Power, KarlPearson, OneWayAnova};

/// This structure provides the results obtained from the regression analysis.

struct RegressionResults
{
    explicit RegressionResults() {}

    virtual ~RegressionResults() {}

    /// Independent coefficient of the logistic function.

    double a = static_cast<double>(NAN);

    /// x coefficient of the logistic function.

    double b = static_cast<double>(NAN);

    /// Correlation coefficient of the  regression.

    double correlation =  static_cast<double>(NAN);

    /// Regression method type.

    RegressionMethod regression_type;
};


/// The enum CorrelationType represents the different correlations methods provided by OpenNN.

enum CorrelationType{Linear_correlation, Logistic_correlation, Logarithmic_correlation, Exponential_correlation, Power_correlation, KarlPearson_correlation, OneWayAnova_correlation};


/// This structure provides the results obtained from the correlations.

struct CorrelationResults
{
    explicit CorrelationResults(){}

    virtual ~CorrelationResults() {}

    /// Correlation coefficient.

    double correlation = static_cast<double>(NAN);

    /// Correlation type.

    CorrelationType correlation_type;
};


    // Linear

    double linear_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double linear_correlation_missing_values(const Tensor<type, 1>&x, const Tensor<type, 1>&);

    // Rank linear

    double rank_linear_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double rank_linear_correlation_missing_values(const Tensor<type, 1>&x, const Tensor<type, 1>&);

    // Exponential

    double exponential_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double exponential_correlation_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Logarithmic

    double logarithmic_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double logarithmic_correlation_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Logistic

    double logistic_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double logistic_correlation_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Rank Logistic

    double rank_logistic_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Power

    double power_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double power_correlation_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Time series correlation methods

    Tensor<type, 1> autocorrelations(const Tensor<type, 1>&, const int & = 10);
    Tensor<type, 1> cross_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&, const int & = 10);

    // Logistic error methods

    double logistic(const double&, const double&, const double&);

    double logistic_error(const double&, const double&, const Tensor<type, 1>&, const Tensor<type, 1>&);
    double logistic_error_missing_values(const double&, const double&, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 1> logistic_error_gradient(const double&, const double&, const Tensor<type, 1>&, const Tensor<type, 1>&);
    Tensor<type, 1> logistic_error_gradient_missing_values(const double&, const double&, const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Regression methods

    RegressionResults linear_regression(const Tensor<type, 1>&, const Tensor<type, 1>&);
    RegressionResults linear_regression_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults logarithmic_regression(const Tensor<type, 1>&, const Tensor<type, 1>&);
    RegressionResults logarithmic_regression_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults exponential_regression(const Tensor<type, 1>&, const Tensor<type, 1>&);
    RegressionResults exponential_regression_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults power_regression(const Tensor<type, 1>&, const Tensor<type, 1>&);
    RegressionResults power_regression_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    RegressionResults logistic_regression(const Tensor<type, 1>&, const Tensor<type, 1>&);
    RegressionResults logistic_regression_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Correlation methods

    CorrelationResults linear_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&);
    CorrelationResults linear_correlations_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults logarithmic_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&);
    CorrelationResults logarithmic_correlations_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults exponential_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&);
    CorrelationResults exponential_correlations_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults power_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&);
    CorrelationResults power_correlations_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults logistic_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&);
    CorrelationResults logistic_correlations_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    CorrelationResults karl_pearson_correlations(const Tensor<type, 2>&, const Tensor<type, 2>&);
    CorrelationResults karl_pearson_correlations_missing_values(const Tensor<type, 2>&, const Tensor<type, 2>&);

    CorrelationResults one_way_anova_correlations(const Tensor<type, 2>&, const Tensor<type, 1>&);
    CorrelationResults one_way_anova_correlations_missing_values(const Tensor<type, 2>&, const Tensor<type, 1>&);

    // Covariance

    double covariance(const Tensor<type, 1>&, const Tensor<type, 1>&);
    double covariance_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 2> covariance_matrix(const Tensor<type, 2>&);

    Tensor<type, 1> less_rank_with_ties(const Tensor<type, 1>&);

    // Contingency tables

    Matrix<int, Dynamic, Dynamic> contingency_table(const Tensor<string, 1>&, const Tensor<string, 1>&);
    Matrix<int, Dynamic, Dynamic> contingency_table(Matrix<string, Dynamic, Dynamic>&);
    Matrix<int, Dynamic, Dynamic> contingency_table(const Tensor<type, 2>&, const Tensor<int, 1>&, const Tensor<int, 1>&);

    double chi_square_test(const Tensor<type, 2>&);

    double chi_square_critical_point(const double&, const double&);

    double karl_pearson_correlation(const Tensor<string, 1>&, const Tensor<string, 1>&);
    double karl_pearson_correlation(const Tensor<type, 2>&, const Tensor<type, 2>&);
    double karl_pearson_correlation_missing_values(const Tensor<type, 2>&, const Tensor<type, 2>&);

    //One way ANOVA

    double one_way_anova(const Tensor<type, 2>&, const Tensor<type, 1>&);
    double one_way_anova(const Tensor<type, 2>& ,const int&, const Tensor<int, 1>&);

    double one_way_anova_correlation(const Tensor<type, 2>&, const Tensor<type, 1>&);
    double one_way_anova_correlation_missing_values(const Tensor<type, 2>&, const Tensor<type, 1>&);

    double f_snedecor_critical_point(const Tensor<type, 2>&, const double&);
    double f_snedecor_critical_point(const Matrix<string, Dynamic, Dynamic>&, const double&);
    double f_snedecor_critical_point(const Tensor<type, 2>&);
    double f_snedecor_critical_point_missing_values(const Tensor<type, 2>&);

    double one_way_anova_correlation(const Tensor<type, 2>& ,const int& , const Tensor<int, 1>&);

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values(const Tensor<type, 1>&, const Tensor<type, 1>&);

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
