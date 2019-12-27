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
    void test_linear_correlation_missing_values();
    void test_rank_linear_correlation();
    void test_rank_linear_correlation_missing_values();

    // Logistic correlation methods

    void test_logistic_correlation();
    void test_rank_logistic_correlation();
    void test_logistic_function();
    void test_logistic_error_gradient();
    void test_logistic_error_gradient_missing_values();
    void test_logistic_correlation_draft();
    void test_logistic_correlation_missing_values();

    // Point-Biserial correlation methods

    void test_point_biserial_correlation();
    void test_point_biserial_correlation_missing_values();

    // Logarithmic correlation methods

    void test_logarithmic_correlation();
    void test_logarithmic_correlation_missing_values();

    // Exponential correlation methods

    void test_exponential_correlation();
    void test_exponential_correlation_missing_values();

    // Regressions methods

    void test_linear_regression();
    void test_linear_regression_missing_values();

    void test_exponential_regression();
    void test_exponential_regression_missing_values();

    void test_logarithmic_regression();
    void test_logarithmic_regression_missing_values();

    void test_power_regression();
    void test_power_regression_missing_values();

    void test_logistic_regression();

    // Time series correlation methods

    void test_autocorrelation();
    void test_cross_correlations();

    // Covariance

    void test_covariance();
    void test_covariance_missing_values();
    void test_covariance_matrix();
    void test_less_rank_with_ties();

    // Remove methods

    void test_remove_correlations();

    //Contingency table

    void test_contingency_table();
    void test_chi_square_test();
    void test_chi_square_critical_point();
    void test_karl_pearson_correlation();
    void test_karl_pearson_correlation_missing_values();

    // One way anova

    void test_one_way_anova();
    void test_one_way_anova_correlation();
    void test_one_way_anova_correlation_missing_values();

    void test_f_snedecor_critical_point();
    void test_f_snedecor_critical_point_missing_values();

    // Unit tseting method

    void run_test_case();

};


#endif



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
