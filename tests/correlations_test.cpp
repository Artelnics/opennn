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
//    cout << "test_spearman_linear_correlation\n";

//    Index size;

//    Tensor<type, 1> x;
//    Tensor<type, 1> y;

//    type correlation;
//    type solution;

//    // Perfect case

//    size = 500000;

//    x.resize(size);
//    x.setRandom();
////    x.setValues({type(5), type(1), type(2),type(1), type(2), type(1), type(1)});

//    cout << "----------------------------------------" << endl;

//    cout << "Old function" << endl;

//    auto start_1 = std::chrono::high_resolution_clock::now();

//    const int n = x.size();

//    Tensor<type,1> x_ranked_function(n);

//    int r, s;

//#pragma omp parallel for
//    for(int i = 0; i < n; i++)
//    {
//        s = std::count(x.data(), x.data() + x.size(), x[i]);
//        r = std::count_if(x.data(), x.data() + x.size(), [&x,i](type j) { return j < x[i];}) + 1;
//        x_ranked_function[i] = r + (s-1) * 0.5;
//    }

//    auto end_1 = std::chrono::high_resolution_clock::now();
//    auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1);

//    std::cout << "Duration _function (in microseconds): " << duration_1.count() << std::endl;

//    cout << "----------------------------------------" << endl;

//    cout << "New function" << endl;

//    auto start_2 = std::chrono::high_resolution_clock::now();

//    vector<pair<type, size_t> > v_sort(x.size());

//    for (size_t i = 0U; i < v_sort.size(); ++i)
//    {
//        v_sort[i] = make_pair(x[i], i);
//    }

//    sort(v_sort.begin(), v_sort.end());

//    vector<type> x_rank(x.size());
//    type rank = 1.0;

//    for (size_t i = 0U; i < v_sort.size(); ++i)
//    {
//        size_t repeated = 1U;
//        for (size_t j = i + 1U; j < v_sort.size() && v_sort[j].first == v_sort[i].first; ++j, ++repeated);
//        for (size_t k = 0; k < repeated; ++k)
//        {
//            x_rank[v_sort[i + k].second] = rank + (repeated - 1) / 2.0;
//        }
//        i += repeated - 1;
//        rank += repeated;
//    }

//    TensorMap<Tensor<type, 1>> tensor(x_rank.data(), x_rank.size());

//    // Code to be timed

//    auto end_2 = std::chrono::high_resolution_clock::now();
//    auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2);

//    std::cout << "Duration (in microseconds): " << duration_2.count() << std::endl;

//    cout << "----------------------------------------" << endl;

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

    assert_true(linear_correlation(thread_pool_device, x, y ).r + solution < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y = type(2)*x;

    correlation = linear_correlation(thread_pool_device, x, y ).r;

    assert_true(abs(correlation - static_cast<type>(1.0)) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(abs(correlation) - static_cast<type>(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    y = type(-1.0)*x;

    correlation = linear_correlation(thread_pool_device, x, y ).r;
    assert_true(abs(correlation + static_cast<type>(1.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation) - static_cast<type>(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
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
    assert_true((correlation.correlation_type == CorrelationType::Logistic), LOG);

    // Test

    size = 10;

    x.resize(size);
    x.setValues({-5,-4,-3,-2,-1,0,1,2,3,4});

    y.resize(size);
    y.setValues({0,0,0,0,0,1,1,1,1,1});

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(correlation.r >= type(0.999), LOG);
    assert_true((correlation.correlation_type == CorrelationType::Logistic), LOG);

    y.setConstant(type(0));

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(correlation.r - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((correlation.correlation_type == CorrelationType::Logistic), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);

    for(Index i = 0; i < size/2; i++) y[i] = 0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(correlation.r <= static_cast<type>(1.0), LOG);

    for(Index i = 0; i < size; i++)
    {
        y[i] = exp(static_cast<type>(2.5)*x[i] + static_cast<type>(1.4));
    }

    const int n = omp_get_max_threads();
    ThreadPool* thread_pool = new ThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    // Test

    for(Index i = 0; i < size/2; i++) y[i] = 1.0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    assert_true(abs(correlation.r) >= static_cast<type>(-0.95), LOG);

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
    assert_true(abs(correlation.b - static_cast<type>(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.a - static_cast<type>(0)) < type(NUMERIC_LIMITS_MIN), LOG);
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

    assert_true(abs(correlation.r - static_cast<type>(1))< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.a - static_cast<type>(1))< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(correlation.b - static_cast<type>(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test missing values

    size = 5;

    x.resize(size);
    x.setValues({ type(1),type(2),type(3),type(4),type(NAN)});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = static_cast<type>(1.4) * exp(static_cast<type>(2.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation.r - type(1)) < type(1.0e-3), LOG);
    assert_true(correlation.b - static_cast<type>(2.5)< type(NUMERIC_LIMITS_MIN), LOG);
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
    for(Index i = 0; i < size; i++) y[i] = static_cast<type>(1) * pow(x[i], type(2));

    Correlation correlation = power_correlation(thread_pool_device,x,y);

    // Test

    assert_true(correlation.r > static_cast<type>(0.999999), LOG);
    assert_true(correlation.a - static_cast<type>(1)< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(correlation.b - static_cast<type>(2)< type(NUMERIC_LIMITS_MIN), LOG);
}

void CorrelationsTest::test_autocorrelations()
{
    cout << "test_autocorrelations\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
    initialize_sequential(x);
    Tensor<type, 1> correlations;

    correlations = autocorrelations(thread_pool_device,x, size/100);
    assert_true(minimum(correlations) > static_cast<type>(0.9), LOG);
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
