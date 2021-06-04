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
    x.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    y.resize(size);
    y.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    solution = 1;

    assert_true(linear_correlation(thread_pool_device, x, y).r - solution < numeric_limits<type>::min(), LOG);

    // Test

    y.setValues({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});

    assert_true(linear_correlation(thread_pool_device, x, y).r + solution < numeric_limits<type>::min(), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y = 2*x;

    correlation = linear_correlation(thread_pool_device, x, y).r;

    assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    // Test

    y = -1.0*x;

    correlation = linear_correlation(thread_pool_device, x, y).r;
    assert_true(abs(correlation + static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
}

void CorrelationsTest::test_linear_regression()
{
    cout << "test_linear_regression\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    // Perfect case

    size = 10;

    x.resize(size);
    x.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    y.resize(size);
    y.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    assert_true(linear_correlation(thread_pool_device, x, y).a  < numeric_limits<type>::min(), LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).b - 1 < numeric_limits<type>::min(), LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).r - 1 < numeric_limits<type>::min(), LOG);

    // Constant case

    x.resize(size);
    x.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    y.resize(size);
    y.setValues({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    assert_true(linear_correlation(thread_pool_device, x, y).a - y(0) < numeric_limits<type>::min(), LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).b < numeric_limits<type>::min(), LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).r - 1 < numeric_limits<type>::min(), LOG);

    // Test

    x.resize(size);
    x.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    y.resize(size);
    y.setValues({2, 5, 8, 11, 14, 17, 20, 23, 26, 29});

    assert_true(linear_correlation(thread_pool_device, x, y).a == 2, LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).b == 3, LOG);
    assert_true(linear_correlation(thread_pool_device, x, y).r == 1, LOG);

}

void CorrelationsTest::test_logistic_correlation()
{
    cout << "test_logistic_correlation\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Correlation correlation;

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y.setConstant(0.0);

    for(Index i = size - size/2; i < size; i++) y[i] = 1;

    correlation = logistic_correlation(thread_pool_device, x, y);

    cout << "correlation r: " << correlation.r << endl;

    assert_true(correlation.r <= 0.95, LOG);

    y.setConstant(1.0);

    for(Index i = size - (size/2); i < size; i++) y[i] = 0;

    correlation = logistic_correlation(thread_pool_device, x, y);

    assert_true(correlation.r >= -0.99, LOG);

    y.setConstant(0.0);

    correlation = logistic_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation.r) < numeric_limits<type>::min(), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);

    for(Index i = 0; i < size/2; i++) y[i] = 0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation(thread_pool_device, x, y);

    cout << "logisitic correlation: " << correlation.r << endl;

    assert_true(correlation.r <= static_cast<type>(0.95), LOG);

    // Test

    for(Index i = 0; i < size/2; i++) y[i] = 1.0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation(thread_pool_device, x, y);

    assert_true(correlation.r >= static_cast<type>(-0.95), LOG);

    // Test

    y.setConstant(0.0);

    correlation = logistic_correlation(thread_pool_device, x,y);

    assert_true(abs(correlation.r) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    // Test

    for(Index i = 0; i < size; i++) i%2 == 0 ? y[i] = 0.0 : y[i] = 1.0;

    correlation = logistic_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation.r) < numeric_limits<type>::min(), LOG);
}


void CorrelationsTest::test_logarithmic_correlation()
{
    cout << "test_logarithmic_correlation\n";

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Index size = 10;

    Correlation correlation;

    type solution;

    // Perfect case

    x.resize(size);
    x.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = 4 * log(x[i]);

    correlation = logarithmic_correlation(thread_pool_device, x, y);

    solution = 1;

    assert_true(abs(correlation.r - solution) <= static_cast<type>(1.0e-6), LOG);
    assert_true(correlation.b == static_cast<type>(4), LOG);
    assert_true(correlation.a == static_cast<type>(0), LOG);
}


void CorrelationsTest::test_exponential_correlation()
{
    cout << "test_exponential_correlation\n";

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    Index size;

    Correlation correlation;

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = static_cast<type>(1) * exp(static_cast<type>(0.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    assert_true(correlation.r > static_cast<type>(0.999999), LOG);
    assert_true(correlation.a == static_cast<type>(1), LOG);
    assert_true(correlation.b == static_cast<type>(0.5), LOG);

    // Test missing values

    size = 5;

    x.resize(size);
    x.setValues({1,2,3,4,NAN});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = static_cast<type>(1.4) * exp(static_cast<type>(2.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation.r - 1.0) < 1.0e-3, LOG);
    assert_true(correlation.b == static_cast<type>(2.5), LOG);
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
    for(Index i = 0; i < size; i++)
    {
        x[i] = i+1;
    }

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = static_cast<type>(1) * pow(x[i], 2);

    Correlation correlation = power_correlation(thread_pool_device,x,y);

    // Test

    assert_true(correlation.r > static_cast<type>(0.999999), LOG);
    assert_true(correlation.a == static_cast<type>(1), LOG);
    assert_true(correlation.b == static_cast<type>(2), LOG);
}


void CorrelationsTest::test_autocorrelations()
{
    cout << "test_autocorrelations\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
    initialize_sequential(x);
    Tensor<type, 1> correlations;

//    correlations = autocorrelations(x, size/100);
    //@todo(assert_true(minimum(correlations) > static_cast<type>(0.9), LOG);)
}


void CorrelationsTest::test_cross_correlations()
{
    cout << "test_cross_correlations\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
    Tensor<type, 1> y(size);

    initialize_sequential(x);
    for(Index i = 0; i < size; i++) y(i) = i;

    Tensor<type, 1> cros_correlations;

//    cros_correlations = cross_correlations(x, y, 10);
    //@todo(assert_true(cros_correlations(0) < 5.0, LOG);)
    //@todo(assert_true(cros_correlations(1) > 0.9, LOG);)
}


void CorrelationsTest::run_test_case()
{
   cout << "Running correlation analysis test case...\n";

   // Correlation methods

//   test_linear_correlation();

//   test_linear_regression();

//   test_logistic_correlation();

//   test_logarithmic_correlation();

//   test_exponential_correlation();

//   test_power_correlation();

   // Time series correlation methods

//   test_autocorrelations();

//   test_cross_correlations();

   cout << "End of correlation analysis test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C); 2005-2021 Artificial Intelligence Techniques, SL.
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
