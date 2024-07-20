//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   E-mail: artelnics@artelnics.com

#include "correlations_test.h"

CorrelationsTest::CorrelationsTest() : UnitTesting()
{
}


CorrelationsTest::~CorrelationsTest()
{
}


void CorrelationsTest::test_spearman_linear_correlation()
{
    cout << "test_spearman_linear_correlation\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    type solution;

    size = 10;

    x.resize(size);
    x.setValues({type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10)});

    y.resize(size);
    y.setValues({type(1), type(3), type(7), type(9), type(10), type(16), type(20), type(28), type(44), type(100)});

    solution = type(1);

    assert_true(linear_correlation_spearman(thread_pool_device, x, y).r - solution < type(NUMERIC_LIMITS_MIN), LOG);
}


void CorrelationsTest::test_linear_correlation()
{
    cout << "test_linear_correlation\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    type correlation;
    type solution;

    // Perfect case

    size = 10;

    x.resize(size);
    x.setValues({type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10)});

    y.resize(size);
    y.setValues({type(333), type(222), type(8),type( 4),type( 6),type( 5),type( 4),type( 3),type( 2),type( 50)});

    solution = type(1);

    assert_true(linear_correlation(thread_pool_device, x, y).r - solution < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).r - solution < type(NUMERIC_LIMITS_MIN), LOG);

    const Tensor<type, 1> x1 = calculate_rank_greater(x).cast<type>();
    const Tensor<type, 1> y1 = calculate_rank_greater(y).cast<type>();

    // Test

    y.setValues({type(10), type(9), type(8),type( 7),type( 6),type( 5),type( 4),type( 3),type( 2),type( 1)});

    assert_true(linear_correlation(thread_pool_device, x, y).r + solution < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y = type(2)*x;

    correlation = linear_correlation(thread_pool_device, x, y).r;

    assert_true(abs(correlation - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(abs(correlation) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    y = type(-1.0)*x;

    correlation = linear_correlation(thread_pool_device, x, y).r;
    assert_true(abs(correlation + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
}


void CorrelationsTest::test_logistic_correlation()
{
    cout << "test_logistic_correlation\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Correlation correlation;

    // Test

    size = 20;

    x.resize(size);
    x.setValues({-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9});

    y.resize(size);
    y.setValues({0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1});

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(abs(correlation.r) <= type(0.1), LOG);
    assert_true((correlation.form == Correlation::Form::Logistic), LOG);

    // Test

    size = 10;

    x.resize(size);
    x.setValues({-5,-4,-3,-2,-1,1,2,3,4,5});

    y.resize(size
             );
    y.setValues({0,0,0,0,0,1,1,1,1,1});

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(correlation.r >= type(0.9), LOG);
    assert_true((correlation.form == Correlation::Form::Logistic), LOG);

    y.setConstant(type(0));

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(correlation.r - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((correlation.form == Correlation::Form::Logistic), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);

    for(Index i = 0; i < size/2; i++) y[i] = 0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(correlation.r <= type(1), LOG);

    for(Index i = 0; i < size; i++)
    {
        y[i] = exp(type(2.5)*x[i] + type(1.4));
    }

    const int n = omp_get_max_threads();

    ThreadPool* thread_pool = new ThreadPool(n);

    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    // Test

    for(Index i = 0; i < size/2; i++) y[i] = 1.0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(abs(correlation.r) >= type(-0.95), LOG);

    // Test

    y.setConstant(type(0));

    correlation = logistic_correlation_vector_vector(thread_pool_device, x,y);

    assert_true(isnan(correlation.r), LOG);
}


void CorrelationsTest::test_logarithmic_correlation()
{
    cout << "test_logarithmic_correlation\n";

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Index size;

    Correlation correlation;

    type solution;

    // Perfect case

    size = 10;

    x.resize(size);
    x.setValues({type(1), type(2), type(3),type( 4),type( 5),type( 6),type( 7),type( 8),type( 9),type( 10)});

    y.resize(size);

    for(Index i = 0; i < size; i++) y[i] = type(4)*log(x[i]);

    correlation = logarithmic_correlation(thread_pool_device, x, y );

    solution = type(1);

    assert_true(abs(correlation.r - solution) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.b - type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.a - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void CorrelationsTest::test_exponential_correlation()
{
    cout << "test_exponential_correlation\n";

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Index size;

    Correlation correlation;

    // Test

    size = 10;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = type(1) * exp(type(0.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation.r - type(1))< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.a - type(1))< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.b - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test missing values

    size = 5;

    x.resize(size);
    x.setValues({ type(1),type(2),type(3),type(4),type(NAN)});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = type(1.4) * exp(type(2.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation.r - type(1)) < type(1.0e-3), LOG);
    assert_true(correlation.b - type(2.5)< type(NUMERIC_LIMITS_MIN), LOG);
}


void CorrelationsTest::test_power_correlation()
{
    cout << "test_power_regression\n";

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Index size;

    // Test

    size = 10;

    x.resize(size);
    for(Index i = 0; i < size; i++) x[i] = type(i+1);

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = type(1) * pow(x[i], type(2));

    Correlation correlation = power_correlation(thread_pool_device,x,y);

    // Test

    assert_true(correlation.r > type(0.999999), LOG);
    assert_true(correlation.a - type(1)< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(correlation.b - type(2)< type(NUMERIC_LIMITS_MIN), LOG);
}

void CorrelationsTest::test_autocorrelations()
{
    cout << "test_autocorrelations\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
    initialize_sequential(x);
    Tensor<type, 1> correlations;

    correlations = autocorrelations(thread_pool_device,x, size/100);
    assert_true(minimum(correlations) > type(0.9), LOG);
}


void CorrelationsTest::test_cross_correlations()
{
    cout << "test_cross_correlations\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
    Tensor<type, 1> y(size);

    initialize_sequential(x);
    for(Index i = 0; i < size; i++) y(i) = type(i);

    Tensor<type, 1> cros_correlations;

    cros_correlations = cross_correlations(thread_pool_device,x, y, 10);
    assert_true(cros_correlations(0) < 5.0, LOG);
    assert_true(cros_correlations(1) > 0.9, LOG);

}


void CorrelationsTest::run_test_case()
{
    cout << "Running correlation analysis test case...\n";

    // Correlation methods

    test_linear_correlation();

    test_spearman_linear_correlation();

    test_logistic_correlation();

    test_logarithmic_correlation();

    test_exponential_correlation();

    test_power_correlation();

    // Time series correlation methods

    test_autocorrelations();

    test_cross_correlations();

    cout << "End of correlation analysis test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C); 2005-2024 Artificial Intelligence Techniques, SL.
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
