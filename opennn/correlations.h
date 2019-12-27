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

#include "vector.h"
#include "matrix.h"
#include "statistics.h"
#include "transformations.h"

using namespace std;

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

    double linear_correlation(const Vector<double>&, const Vector<double>&);
    double linear_correlation_missing_values(const Vector<double>&x, const Vector<double>&);

    // Rank linear

    double rank_linear_correlation(const Vector<double>&, const Vector<double>&);
    double rank_linear_correlation_missing_values(const Vector<double>&x, const Vector<double>&);

    // Exponential

    double exponential_correlation(const Vector<double>&, const Vector<double>&);
    double exponential_correlation_missing_values(const Vector<double>&, const Vector<double>&);

    // Logarithmic

    double logarithmic_correlation(const Vector<double>&, const Vector<double>&);
    double logarithmic_correlation_missing_values(const Vector<double>&, const Vector<double>&);

    // Logistic

    double logistic_correlation(const Vector<double>&, const Vector<double>&);
    double logistic_correlation_missing_values(const Vector<double>&, const Vector<double>&);

    // Rank Logistic

    double rank_logistic_correlation(const Vector<double>&, const Vector<double>&);

    // Power

    double power_correlation(const Vector<double>&, const Vector<double>&);
    double power_correlation_missing_values(const Vector<double>&, const Vector<double>&);

    // Time series correlation methods

    Vector<double> autocorrelations(const Vector<double>&, const size_t & = 10);
    Vector<double> cross_correlations(const Vector<double>&, const Vector<double>&, const size_t & = 10);

    // Logistic error methods

    double logistic(const double&, const double&, const double&);

    double logistic_error(const double&, const double&, const Vector<double>&, const Vector<double>&);
    double logistic_error_missing_values(const double&, const double&, const Vector<double>&, const Vector<double>&);

    Vector<double> logistic_error_gradient(const double&, const double&, const Vector<double>&, const Vector<double>&);
    Vector<double> logistic_error_gradient_missing_values(const double&, const double&, const Vector<double>&, const Vector<double>&);

    // Regression methods

    RegressionResults linear_regression(const Vector<double>&, const Vector<double>&);
    RegressionResults linear_regression_missing_values(const Vector<double>&, const Vector<double>&);

    RegressionResults logarithmic_regression(const Vector<double>&, const Vector<double>&);
    RegressionResults logarithmic_regression_missing_values(const Vector<double>&, const Vector<double>&);

    RegressionResults exponential_regression(const Vector<double>&, const Vector<double>&);
    RegressionResults exponential_regression_missing_values(const Vector<double>&, const Vector<double>&);

    RegressionResults power_regression(const Vector<double>&, const Vector<double>&);
    RegressionResults power_regression_missing_values(const Vector<double>&, const Vector<double>&);

    RegressionResults logistic_regression(const Vector<double>&, const Vector<double>&);
    RegressionResults logistic_regression_missing_values(const Vector<double>&, const Vector<double>&);

    // Correlation methods

    CorrelationResults linear_correlations(const Vector<double>&, const Vector<double>&);
    CorrelationResults linear_correlations_missing_values(const Vector<double>&, const Vector<double>&);

    CorrelationResults logarithmic_correlations(const Vector<double>&, const Vector<double>&);
    CorrelationResults logarithmic_correlations_missing_values(const Vector<double>&, const Vector<double>&);

    CorrelationResults exponential_correlations(const Vector<double>&, const Vector<double>&);
    CorrelationResults exponential_correlations_missing_values(const Vector<double>&, const Vector<double>&);

    CorrelationResults power_correlations(const Vector<double>&, const Vector<double>&);
    CorrelationResults power_correlations_missing_values(const Vector<double>&, const Vector<double>&);

    CorrelationResults logistic_correlations(const Vector<double>&, const Vector<double>&);
    CorrelationResults logistic_correlations_missing_values(const Vector<double>&, const Vector<double>&);

    CorrelationResults karl_pearson_correlations(const Matrix<double>&, const Matrix<double>&);
    CorrelationResults karl_pearson_correlations_missing_values(const Matrix<double>&, const Matrix<double>&);

    CorrelationResults one_way_anova_correlations(const Matrix<double>&, const Vector<double>&);
    CorrelationResults one_way_anova_correlations_missing_values(const Matrix<double>&, const Vector<double>&);

    // Covariance

    double covariance(const Vector<double>&, const Vector<double>&);
    double covariance_missing_values(const Vector<double>&, const Vector<double>&);

    Matrix<double> covariance_matrix(const Matrix<double>&);

    Vector<double> less_rank_with_ties(const Vector<double>&);

    // Contingency tables

    Matrix<size_t> contingency_table(const Vector<string>&, const Vector<string>&);
    Matrix<size_t> contingency_table(Matrix<string>&);
    Matrix<size_t> contingency_table(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>& );

    double chi_square_test(const Matrix<double>&);

    double chi_square_critical_point(const double&, const double&);

    double karl_pearson_correlation(const Vector<string>&, const Vector<string>&);
    double karl_pearson_correlation(const Matrix<double>&, const Matrix<double>&);
    double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>&);

    //One way ANOVA

    double one_way_anova(const Matrix<double>&, const Vector<double>&);
    double one_way_anova(const Matrix<double>& ,const size_t&, const Vector<size_t>& );

    double one_way_anova_correlation(const Matrix<double>&, const Vector<double>&);
    double one_way_anova_correlation_missing_values(const Matrix<double>&, const Vector<double>&);

    double f_snedecor_critical_point(const Matrix<double>&, const double&);
    double f_snedecor_critical_point(const Matrix<string>&, const double&);    
    double f_snedecor_critical_point(const Matrix<double>&);
    double f_snedecor_critical_point_missing_values(const Matrix<double>&);

    double one_way_anova_correlation(const Matrix<double>& ,const size_t& , const Vector<size_t>& );

    pair<Vector<double>, Vector<double>> filter_missing_values(const Vector<double>&, const Vector<double>&);

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
