/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O R R E L A T I O N   A N A L Y S I S   T E S T   C L A S S                                              */
/*                                                                                                              */
/*   Javier Sanchez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   javiersanchez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __CORRELATIONANALYSIS_H__
#define __CORRELATIONANALYSIS_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class CorrelationAnalysisTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

    // DEFAULT CONSTRUCTOR

    explicit CorrelationAnalysisTest();


    // DESTRUCTOR

    virtual ~CorrelationAnalysisTest();

    // Methods

    void test_constructor();
    void test_destructor();

    // Linear correlation methods

    void test_calculate_linear_correlation();

    void test_calculate_spearman_linear_correlation();

    void test_calculate_linear_correlation_missing_values();

    // Multivariate Correlation Methods

    void test_calculate_multivariate_correlation();

    // Logistic correlation methods

    void test_calculate_logistic_correlation();

    void test_calculate_rank_logistic_correlation();

    void test_calculate_logistic_function();

    void test_calculate_logistic_error_gradient();

    void test_calculate_logistic_correlation_draft();

    void test_calculate_logistic_correlation_missing_values();

    // Point-Biserial correlation methods

    void test_calculate_point_biserial_correlation();

    void test_calculate_point_biserial_correlation_missing_values();

    // Logaithmic correlation methods

    void test_caculate_logarithmic_correlation();

    // Exponential correlation methods

    void test_calculate_exponential_correlation();

    // Time series correlation methods

    void test_calculate_autocorrelation();

    void test_calculate_cross_correlation();

    // General correlation methods

    void test_calculate_correlation();

    void test_calculate_correlations();

//    void test_calculate_correlations() const;

    // Unit tseting method

    void run_test_case();
};


#endif



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
