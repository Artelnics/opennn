//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CORRELATIONS_TESTS_H
#define CORRELATIONS_TESTS_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class CorrelationsTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

    // DEFAULT CONSTRUCTOR

    explicit CorrelationsTest();

    virtual ~CorrelationsTest();

    // Linear correlation methods

    void test_linear_correlation();
    void test_spearman_linear_correlation();
    void test_rank_linear_correlation();

    // Logistic correlation methods

    void test_logistic_correlation();
    void test_rank_logistic_correlation();
    void test_logistic_function();
    void test_logistic_error_gradient();

    // Point-Biserial correlation methods

    void test_point_biserial_correlation();

    // Logarithmic correlation methods

    void test_logarithmic_correlation();

    // Exponential correlation methods

    void test_exponential_correlation();

    // Regressions methods

    void test_linear_regression();

    void test_exponential_regression();

    void test_logarithmic_regression();

    void test_power_regression();

    void test_logistic_regression();

    // Time series correlation methods

    void test_autocorrelation();
    void test_cross_correlations();

    // Covariance

    void test_covariance();
    void test_covariance_matrix();
    void test_less_rank_with_ties();

    //Contingency table

    void test_contingency_table();
    void test_chi_square_test();
    void test_chi_square_critical_point();
    void test_karl_pearson_correlation();

    // Unit tseting method

    void run_test_case();

};


#endif



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
