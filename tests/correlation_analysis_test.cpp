/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O R R E L A T I O N   A N A L Y S I S   T E S T   C L A S S                                              */
/*                                                                                                              */
/*   Javier Sanchez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   E-mail: javiersanchez@artelnics.com                                                                        */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "correlation_analysis_test.h"
#include "correlation_analysis.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

CorrelationAnalysisTest::CorrelationAnalysisTest() : UnitTesting()
{
}


// DESTRUCTOR

CorrelationAnalysisTest::~CorrelationAnalysisTest()
{
}


// METHODS

void CorrelationAnalysisTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   CorrelationAnalysis ca1;

}


void CorrelationAnalysisTest::test_destructor()
{
   message += "test_destructor\n";
}

void CorrelationAnalysisTest::test_calculate_linear_correlation()
{
    message += "test_calculate_linear_correlation\n";

    size_t test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);

    double correlation;

    for(int i = 0; i < test_size; i++) y[i] = 2*x[i];

    correlation = CorrelationAnalysis::calculate_linear_correlation(x,y);
    assert_true(correlation == 1, LOG);
    assert_true(abs(correlation) <= 1, LOG);

    for(int i = 0; i <test_size; i++) y[i] = -i;

    correlation = CorrelationAnalysis::calculate_linear_correlation(x,y);
    assert_true(correlation == -1, LOG);
    assert_true(abs(correlation) <= 1, LOG);

}


void CorrelationAnalysisTest::test_calculate_spearman_linear_correlation()
{
    message += "test_calculate_spearman_linear_correlation\n";

    size_t test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);

    for(int i = 0; i < test_size; i++) y[i] = 2*x[i];

    double correlation;

    correlation = CorrelationAnalysis::calculate_rank_linear_correlation(x,y);
    assert_true(correlation == 1, LOG);
    assert_true(abs(correlation) <= 1, LOG);

    for(int i = 0; i <test_size; i++) y[i] = -i;

    assert_true(CorrelationAnalysis::calculate_rank_linear_correlation(x,y) == -1, LOG);
    assert_true(abs(correlation) <= 1, LOG);

    y.initialize(0.1);
    correlation = CorrelationAnalysis::calculate_rank_linear_correlation(x,y);
    assert_true(correlation == 1, LOG);
    assert_true(abs(correlation) <= 1, LOG);
}


void CorrelationAnalysisTest::test_calculate_linear_correlation_missing_values()
{
    message += "test_calculate_linear_correlation_missing_values\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);
    Vector<size_t> z(10);

    z.randomize_uniform(0,test_size);

    double correlation;

    for(int i = 0; i < test_size; i++) y[i] = 2*x[i];

    correlation = CorrelationAnalysis::calculate_linear_correlation_missing_values(x,y,z);
    assert_true(correlation == 1, LOG);
    assert_true(abs(correlation) <= 1, LOG);

    for(int i = 0; i <test_size; i++) y[i] = -i;

    correlation = CorrelationAnalysis::calculate_linear_correlation_missing_values(x,y,z);
    assert_true(correlation == -1, LOG);
    assert_true(abs(correlation) <= 1, LOG);
}


void CorrelationAnalysisTest::test_calculate_multivariate_correlation()
{
    message += "test_calculate_multivariate_correlation\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.randomize_normal();
    Vector<double> z(test_size);
    z.initialize_sequential();
    Vector<double> y(test_size);

    Matrix<double> matrix(test_size, 2, 0.0);
    matrix.set_column(0, x);

    double correlation;

    for(int i = 0; i < y.size(); i++)
    {
        z[i] *= 0.1235;
        y[i] = y[i] = 1.32 + 5.6 * x[i] + 0.34 * z[i];
    }
    matrix.set_column(1, z);

    correlation = CorrelationAnalysis::calculate_multivariate_linear_correlation(matrix, y);
    assert_true(correlation > 0.99, LOG);

}


void CorrelationAnalysisTest::test_calculate_logistic_correlation()
{
    message += "test_calculate_logistic_correlation\n";

    const int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();

    Vector<double> y(test_size);

    double correlation;

    y.initialize(0);
    for(int i= test_size - (test_size/2); i <test_size; i++) y[i] = 1;
    correlation = CorrelationAnalysis::calculate_logistic_correlation(x,y);
    assert_true(correlation > 0.95, LOG);

    y.initialize(1);
    for(int i= test_size - (test_size/2); i <test_size; i++) y[i] = 0;
    correlation = CorrelationAnalysis::calculate_logistic_correlation(x,y);
    assert_true(correlation < -0.99, LOG);

    y.initialize(0.0);
    correlation = CorrelationAnalysis::calculate_logistic_correlation(x,y);
    assert_true(abs(correlation) == 1, LOG);
}


void CorrelationAnalysisTest::test_calculate_rank_logistic_correlation()
{
    message += "test_calculate_rank_logistic_correlation\n";

    const int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();

    Vector<double> y(test_size);

    double correlation;

    y.initialize(0);
    for(int i= test_size - (test_size/2); i <test_size; i++) y[i] = 1;
    correlation = CorrelationAnalysis::calculate_rank_logistic_correlation(x,y);
    assert_true(correlation > 0.95, LOG);

    y.initialize(1);
    for(int i= test_size - (test_size/2); i <test_size; i++) y[i] = 0;
    correlation = CorrelationAnalysis::calculate_rank_logistic_correlation(x,y);
    assert_true(correlation < -0.99, LOG);

    y.initialize(0.0);
    correlation = CorrelationAnalysis::calculate_rank_logistic_correlation(x,y);
    assert_true(abs(correlation) == 1, LOG);
}


void CorrelationAnalysisTest::test_calculate_logistic_correlation_missing_values()
{
    message += "test_calculate_logistic_correlation_missing_values\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);
    Vector<size_t> z(10);
    z.randomize_uniform(0,test_size);

    double correlation;

    for(int i = 0; i < test_size/2; i++) y[i] = 0;

    for(int i = test_size - (test_size/2); i <test_size; i++) y[i] = 1;

    correlation = CorrelationAnalysis::calculate_logistic_correlation_missing_values(x,y,z);
    assert_true(correlation > 0.95, LOG);

    for(int i = 0; i < test_size/2; i++) y[i] = 1;

    for(int i= test_size - (test_size/2); i <test_size; i++) y[i] = 0;

    correlation = CorrelationAnalysis::calculate_logistic_correlation_missing_values(x,y,z);
    assert_true(correlation < -0.95, LOG);

    y.initialize(0.0);
    correlation = CorrelationAnalysis::calculate_logistic_correlation_missing_values(x,y,z);
    assert_true(abs(correlation) == 1, LOG);

    y.randomize_binary();
    correlation = CorrelationAnalysis::calculate_logistic_correlation_missing_values(x,y,z);
    assert_true(correlation < 0.3, LOG);
}


void CorrelationAnalysisTest::test_calculate_autocorrelation()
{
    message += "test_calculate_autocorrelation\n";

    size_t test_size = 1000;
    Vector<double> x(test_size);
    x.initialize_sequential();

    Vector<double> correlations;

    correlations = CorrelationAnalysis::calculate_autocorrelations(x, test_size/100);
    assert_true(correlations.calculate_minimum() > 0.9, LOG);
}


void CorrelationAnalysisTest::test_calculate_point_biserial_correlation()
{
    message += "test_calculate_point_biserial_correlation\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);

    double correlation;

    y.initialize(1);
    for(int i = 0; i < y.size(); i++)
    {
        if(i%2 == 0) y[i] = 0;
    }
    correlation = CorrelationAnalysis::calculate_point_biserial_correlation(x,y);
    assert_true(abs(correlation) < 0.1, LOG);

    for(int i = 0; i < test_size/2; i++)
    {
        y[i] = 0;
    }
    for(int i = test_size - (test_size/2); i <test_size; i++)
    {
        y[i] = 1;
    }
    correlation = CorrelationAnalysis::calculate_point_biserial_correlation(x,y);
    assert_true(correlation > 0.8, LOG);

    for(int i = 0; i < test_size/2; i++)
    {
        y[i] = 1;
    }
    for(int i = test_size - (test_size/2); i <test_size; i++)
    {
        y[i] = 0;
    }
    correlation = CorrelationAnalysis::calculate_point_biserial_correlation(x,y);
    assert_true(correlation < -0.8, LOG);
}


void CorrelationAnalysisTest::test_calculate_point_biserial_correlation_missing_values()
{
    message += "test_calculate_point_biserial_correlation_missing_values\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);
    Vector<size_t> z(10);

    z.randomize_uniform(0,test_size);

    double correlation;

    y.initialize(1);
    for(int i = 0; i < y.size(); i++)
    {
        if(i%2 == 0) y[i] = 0;
    }
    correlation = CorrelationAnalysis::calculate_point_biserial_correlation_missing_values(x,y,z);
    assert_true(abs(correlation) < 0.1, LOG);

    for(int i = 0; i < test_size/2; i++)
    {
        y[i] = 0;
    }
    for(int i = test_size - (test_size/2); i <test_size; i++)
    {
        y[i] = 1;
    }
    correlation = CorrelationAnalysis::calculate_point_biserial_correlation_missing_values(x,y,z);
    assert_true(correlation > 0.8, LOG);

    for(int i = 0; i < test_size/2; i++)
    {
        y[i] = 1;
    }
    for(int i = test_size - (test_size/2); i <test_size; i++)
    {
        y[i] = 0;
    }
    correlation = CorrelationAnalysis::calculate_point_biserial_correlation_missing_values(x,y,z);
    assert_true(correlation < -0.8, LOG);
}

void CorrelationAnalysisTest::test_caculate_logarithmic_correlation()
{
    message += "test_calculate_logarithmic_correlation\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);

    for(int i = 0; i < test_size; i++)
    {
        y[i] = log(1.5 * x[i] + 2);
    }

    double correlation = CorrelationAnalysis::calculate_logarithmic_correlation(x,y);
    assert_true(correlation == 1, LOG);
}


void CorrelationAnalysisTest::test_calculate_exponential_correlation()
{
    message += "test_calculate_exponential_correlation\n";

    int test_size = 100;
    Vector<double> x(test_size);
    x.initialize_sequential();
    Vector<double> y(test_size);

    for(int i = 0; i < test_size; i++)
    {
        y[i] = exp(2.5 * x[i] + 1.4);
    }

    double correlation = CorrelationAnalysis::calculate_exponential_correlation(x,y);
    assert_true(correlation > 0.999999, LOG);
}

void CorrelationAnalysisTest::run_test_case()
{
   message += "Running correlation analysis test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Linear correlation methods

   test_calculate_linear_correlation();

   test_calculate_spearman_linear_correlation();

   test_calculate_linear_correlation_missing_values();

   // Multivariate correlation method

   test_calculate_multivariate_correlation();

   // Logistic correlation methods

   test_calculate_logistic_correlation();

   test_calculate_rank_logistic_correlation();

   test_calculate_logistic_correlation_missing_values();

   // Point-Biserial correlation methods

   test_calculate_point_biserial_correlation();

   test_calculate_point_biserial_correlation_missing_values();

   // Logarithmic correlation methods

   test_caculate_logarithmic_correlation();

   // Exponential correlation methods

   test_calculate_exponential_correlation();

   // Time series correlation methods

   test_calculate_autocorrelation();

   message += "End of correlation analysis test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C); 2005-2018 Artificial Intelligence Techniques, SL.
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
