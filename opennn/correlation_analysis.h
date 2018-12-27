/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O R R E L A T I O N   A N A L Y S I S                                                                    */
/*                                                                                                              */
/*   Javier Sanchez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   javiersanchez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef CORRELATION_ANALYSIS_H
#define CORRELATION_ANALYSIS_H

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
#include "opennn.h"


using namespace std;

namespace OpenNN
{

class CorrelationAnalysis
{

public:

    // DEFAULT CONSTRUCTOR

    explicit CorrelationAnalysis();


    // DESTRUCTOR

    virtual ~CorrelationAnalysis();


    // Methods

    // Linear

    static double calculate_linear_correlation(const Vector<double>&, const Vector<double>&);

    static double calculate_linear_correlation_missing_values(const Vector<double> &x, const Vector<double>&, const Vector<size_t> &);

    // Rank linear

    static double calculate_rank_linear_correlation(const Vector<double>&, const Vector<double>&);

    static double calculate_rank_linear_correlation_missing_values(const Vector<double> &x, const Vector<double>&, const Vector<size_t> &);

    // Point biserial

    static double calculate_point_biserial_correlation(const Vector<double>&, const Vector<double>&);

    static double calculate_point_biserial_correlation_missing_values(const Vector<double>&, const Vector<double>&, const Vector<size_t>&);

    // Exponential

    static double calculate_exponential_correlation(const Vector<double>&, const Vector<double>&);

    static double calculate_exponential_correlation_missing_values(const Vector<double>&, const Vector<double>&, const Vector<size_t>&);

    // Logarithmic

    static double calculate_logarithmic_correlation(const Vector<double>&, const Vector<double>&);

    static double calculate_logarithmic_correlation_missing_values(const Vector<double>&, const Vector<double>&, const Vector<size_t>&);

    // Logistic

    static double calculate_logistic_correlation(const Vector<double>&, const Vector<double>&);

    static double calculate_logistic_correlation_missing_values(const Vector<double>&, const Vector<double>&, const Vector<size_t> &);

    // Rank Logistic correlation methods

    static double calculate_rank_logistic_correlation(const Vector<double>&, const Vector<double>&);

    // Power

    // Polynomial

    static double calculate_polynomial_correlation(const Vector<double>&, const Vector<double>&, const size_t&);

    static size_t calculate_best_correlation_order(const Vector<double>&, const Vector<double>&, const size_t& = 5);

    // Multiple linear

    static double calculate_multivariate_linear_correlation(const Matrix<double>&, const Vector<double>&);

    static double calculate_multivariate_linear_correlation_missing_values(const Matrix<double>&, const Vector<double>&, const Vector<size_t>&);

    // Logistic regression methods

    static Vector<double> calculate_logistic_regression(const Matrix<double>&, const Vector<double>&, const double& =1e-7);

    static Vector<double> calculate_logistic_regression(const Vector<double>&, const Vector<double>&, const double& =1e-7);

    // Time series correlation methods

    static Vector<double> calculate_autocorrelations(const Vector<double>&, const size_t & = 10);

    static Vector<double> calculate_cross_correlations(const Vector<double> &, const Vector<double> &, const size_t & = 10);


    // General correlation methods

    static double calculate_correlation(const Vector<double>&, const Vector<double>&);

    // Matrix Methods

    static Matrix<double> calculate_correlations(const Matrix<double>&);

    static Vector<double> calculate_correlations(const Matrix<double>&, const size_t&);

    static Matrix<double> calculate_correlations(const Matrix<double>&, const Vector<size_t>&);

    // Multiple correlation methods

    static Matrix<double> calculate_multiple_linear_correlations(const DataSet&, const Vector<size_t>&);

    static double calculate_multiple_linear_correlation(const Matrix<double>&, const Vector<double>&);

    // Logistic error methods

    static double calculate_logistic_function(const Vector<double>&, const Vector<double> &);

    static Vector<double> calculate_logistic_error_gradient(const Vector<double>&, const Vector<double>&, const Vector<double>&);

    // Remove methods

    static Matrix<double> remove_correlations(const Matrix<double>&, const size_t&, const double&);

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
