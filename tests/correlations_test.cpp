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

    Index size;

    Correlation correlation;

    type solution;

    // Perfect case

    x.resize(10);
    x.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    y.resize(10);
    y.setValues({0, 0.30103, 0.477121, 0.60206, 0.69897, 0.778151, 0.845098, 0.90309, 0.954243, 1});

    correlation = logarithmic_correlation(thread_pool_device, x, y);

    solution = 1;

    assert_true(abs(correlation.r - solution) <= static_cast<type>(1.0e-6), LOG);

    assert_true(logarithmic_correlation(thread_pool_device, x, y).r - solution <= static_cast<type>(0.00001), LOG);

    // Test missing values

    size = 5;

    x.resize(size);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    x[3] = 4;
    x[4] = static_cast<type>(NAN);

    y.resize(size);

    for(Index i = 0; i < size; i++) y[i] = log(static_cast<type>(1.5)*x[i] + 2);
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
    for(Index i = 0; i < size; i++) y[i] = exp(static_cast<type>(2.5)*x[i] + static_cast<type>(1.4));

    correlation = exponential_correlation(thread_pool_device, x, y);

    assert_true(correlation.r > static_cast<type>(0.999999), LOG);

    // Test missing values

    size = 5;

    x.resize(size);
    x.setValues({1,2,3,4,NAN});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = exp(static_cast<type>(2.5)*x[i] + static_cast<type>(1.4));

    correlation = exponential_correlation(thread_pool_device, x, y);
    assert_true(abs(correlation.r - 1.0) < 1.0e-3, LOG);
}


void CorrelationsTest::test_power_correlation()
{
    cout << "test_power_regression\n";

    Tensor<type, 1> x(5);
    x.setValues({10, 16, 25, 40, 60});
    Tensor<type, 1> y(5);
    y.setValues({94, 118, 147, 180, 230});

    type solution1 = static_cast<type>(30.213);
    type solution2 = static_cast<type>(0.491);
    type solution3 = static_cast<type>(0.998849);

    Correlation correlation = power_correlation(thread_pool_device,x,y);

    // Test

    assert_true(correlation.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(correlation.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(correlation.r - solution3 <= static_cast<type>(0.01), LOG);

    x.resize(6);
    x.setValues({10, 16, 25, 40, 60, NAN});
    y.resize(6);
    y.setValues({94, 118, 147, 180, 230, 100});

    solution1 = static_cast<type>(30.213);
    solution2 = static_cast<type>(0.491);
    solution3 = static_cast<type>(0.998849);

    // Test

    correlation = power_correlation(thread_pool_device, x,y);

    assert_true(correlation.a - solution1 <= 0.01, LOG);
    assert_true(correlation.b - solution2 <= 0.01, LOG);
    assert_true(correlation.r - solution3 <= 0.01, LOG);
}


void CorrelationsTest::test_contingency_table()
{
    cout << "test_contingency_table\n";

    Tensor<string, 1> x(4);
    x.setValues({"a", "b", "b", "a" });
    Tensor<string, 1> y(4);
    y.setValues({"c", "c", "d", "d" });

//    assert_true(contingency_table(x, y) == matrix, LOG);

}


void CorrelationsTest::test_chi_square_test()
{
    cout << "test_chi_square_test\n";

//    Tensor<string, 1> x({"a", "b", "b", "a" });
//    Tensor<string, 1> y({"c", "c", "d", "d" });

//    assert_true(abs(chi_square_test(contingency_table(x, y).to_type_matrix()) - 0.0) < 1.0e-3, LOG);

}


void CorrelationsTest::test_karl_pearson_correlation()
{
    cout << "test_karl_pearson_correlation\n";

    Tensor<type, 2> matrix1(4,2);
    Tensor<type, 2> matrix2(4,2);

//    matrix1.set_values(0, {1, 0, 1, 0});
//    matrix1.set_column(1, {0, 1, 0, 1});

//    matrix2.set_column(0, {1, 0, 1, 0});
//    matrix2.set_column(1, {0, 1, 0, 1});

    const type solution = 1;

//    const type correlation = karl_pearson_correlation(matrix1, matrix2);

//    assert_true(abs(correlation - solution) <= 0.001, LOG);

    matrix1.resize(5,2);
    matrix2.resize(5,2);

//    matrix1.set_column(0, {1, 0, 1, 0, 0});
//    matrix1.set_column(1, {0, 1, NAN, 1, NAN});

//    matrix2.set_column(0, {1, 0, 1, 0, 0});
//    matrix2.set_column(1, {0, NAN, 0, 1, 1});

//    const type solution = 1;

//    const type correlation = karl_pearson_correlation(matrix1, matrix2);

//    assert_true(abs(correlation - solution) <= 0.001, LOG);
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

   test_logistic_correlation();
/*
   test_logarithmic_correlation();

   test_exponential_correlation();

   test_power_correlation();

   test_contingency_table();

   test_chi_square_test();

   test_karl_pearson_correlation();

   // Time series correlation methods

   test_autocorrelations();
   test_cross_correlations();
*/
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
