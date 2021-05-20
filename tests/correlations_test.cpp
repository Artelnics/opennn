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

    assert_true(linear_correlation(thread_pool_device, x, y) - solution < numeric_limits<type>::min(), LOG);

    // Test

    y.setValues({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});

    assert_true(linear_correlation(thread_pool_device, x, y) + solution < numeric_limits<type>::min(), LOG);

    // General case

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y = 2*x;

    correlation = linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    y = -1.0*x;

    correlation = linear_correlation(thread_pool_device, x, y);
    assert_true(abs(correlation + static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

}


/// @todo

void CorrelationsTest::test_spearman_linear_correlation()
{
    cout << "test_spearman_linear_correlation\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    type correlation;

    // Test

    size = 10;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y = 2*x;

    correlation = rank_linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);

    // Test

    y = -1.0*x;

    assert_true(abs(rank_linear_correlation(thread_pool_device, x, y)) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation) <= static_cast<type>(1.0), LOG);

    //Test

    y.setConstant(static_cast<type>(0.1));

    correlation = rank_linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation) <= static_cast<type>(1.0), LOG);

    // Test missing values @todo

    x.resize(5);
    initialize_sequential(x);

    y.resize(5);
    y[0] = 1;
    y[1] = 2;
    y[2] = 3;
    y[3] = static_cast<type>(NAN);
    y[4] = 5;

//    type linear_correlation = linear_correlation(thread_pool_device, vector, target);

//    assert_true(abs(linear_correlation - 1.0) < 1.0e-3, LOG );
}


void CorrelationsTest::test_rank_linear_correlation()
{
    cout << "test_rank_linear_correlation()\n";

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    type correlation;

    // Test

    x.resize(10);
    initialize_sequential(x);

    y = 2*x;

    correlation = rank_linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);

    // Test

    y = -1.0*x;

    correlation = rank_linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    // Test missing values

    x.resize(5);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    x[3] = 4;
    x[4] = 5;

    y.resize(5);
    y[0] = 1;
    y[1] = 2;
    y[2] = 3;
    y[3] = static_cast<type>(NAN);
    y[4] = 5;

    correlation = rank_linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation - static_cast<type>(1.0)) < static_cast<type>(1.0e-3), LOG);

    // Test with ties

    x.resize(5);
    x[0] = 1;
    x[1] = 1;
    x[2] = 3;
    x[3] = static_cast<type>(NAN);
    x[4] = 5;

    y.resize(5);
    y[0] = 1;
    y[1] = 1;
    y[2] = 3;
    y[3] = static_cast<type>(NAN);
    y[4] = 5;

    correlation = rank_linear_correlation(thread_pool_device, x, y);

    assert_true(abs(correlation - static_cast<type>(1.0)) < static_cast<type>(1.0e-3), LOG );
}


void CorrelationsTest::test_logistic_correlation()
{
    cout << "test_logistic_correlation\n";

    Index size;

    Tensor<type, 1> x;
    Tensor<type, 1> y;

    type correlation;

    // Test

    size = 100;
    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y.setConstant(0.0);

    for(Index i = size - size/2; i < size; i++) y[i] = 1;

//    correlation = logistic_correlation(thread_pool_device, x, y);

//    assert_true(correlation <= 0.95, LOG);

//    y.setConstant(1.0);

//    for(Index i= size - (size/2); i < size; i++) y[i] = 0;

//    correlation = logistic_correlation(x.to_column_matrix(), y);

//    cout << correlation << endl;

//    assert_true(correlation >= -0.99, LOG);

//    y.setConstant(0.0);

//    correlation = logistic_correlation(x.to_column_matrix(), y);

//    assert_true(abs(correlation - 0.0) < numeric_limits<type>::min(), LOG);

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);

    for(Index i = 0; i < size/2; i++) y[i] = 0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

//    correlation = logistic_correlation(x, y);

    assert_true(correlation <= static_cast<type>(0.95), LOG);

    // Test

    for(Index i = 0; i < size/2; i++) y[i] = 1.0;

    for(Index i= size - (size/2); i < size; i++) y[i] = 0.0;

//    correlation = logistic_correlation(x.to_column_matrix(),y);

    assert_true(correlation >= static_cast<type>(-0.95), LOG);

    // Test

    y.setConstant(0.0);

//    correlation = logistic_correlation(x.to_column_matrix(),y);

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    // Test

//    y.randomize_binary();

//    correlation = logistic_correlation(x.to_column_matrix(),y);

    assert_true(correlation < static_cast<type>(0.3), LOG);

}


void CorrelationsTest::test_rank_logistic_correlation()
{
    cout << "test_rank_logistic_correlation\n";

    const Index size = 10;
    Tensor<type, 1> x(size);
    initialize_sequential(x);

    Tensor<type, 1> y(size);

    type correlation;

    y.setConstant(0.0);
    for(Index i= size - (size/2); i < size; i++) y[i] = 1;
//    correlation = rank_logistic_correlation(thread_pool_device, x,y);
//    assert_true(correlation <= 1, LOG);

//    y.setConstant(1.0);
//    for(Index i= size - (size/2); i < size; i++) y[i] = 0;
//    correlation = rank_logistic_correlation(x,y);
//    assert_true(correlation >= -1, LOG);

//    y.setConstant(0.0);
//    correlation = rank_logistic_correlation(x,y);
//    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

}


void CorrelationsTest::test_autocorrelation()
{
    cout << "test_autocorrelation\n";

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


void CorrelationsTest::test_logarithmic_correlation()
{
    cout << "test_logarithmic_correlation\n";

    // Perfect case

    const Tensor<type, 1> vector_1;//({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    const Tensor<type, 1> vector_2;//({0, 0.30103, 0.477121, 0.60206, 0.69897, 0.778151, 0.845098, 0.90309, 0.954243, 1});

    const type solution = 1;

    const type correl = logarithmic_correlation(thread_pool_device, vector_1, vector_2) - solution;

    assert_true(correl - solution <= static_cast<type>(0.0001), LOG);

    assert_true(logarithmic_correlation(thread_pool_device, vector_1, vector_2) - solution <= static_cast<type>(0.00001), LOG);

    // Test missing values

    Tensor<type, 1> vector;
    vector.resize(5);
    vector[0] = 1;
    vector[1] = 2;
    vector[2] = 3;
    vector[3] = 4;
    vector[4] = static_cast<type>(NAN);

    Tensor<type, 1> target(vector.size());

    for(Index i = 0; i < vector.size(); i++)
    {
        target[i] = log(static_cast<type>(1.5)*vector[i] + 2);
    }
}


void CorrelationsTest::test_exponential_correlation()
{
    cout << "test_exponential_correlation\n";

    const Index size = 100;
    Tensor<type, 1> x(size);

    initialize_sequential(x);
    Tensor<type, 1> y(size);

    for(Index i = 0; i < size; i++) y[i] = exp(static_cast<type>(2.5)*x[i] + static_cast<type>(1.4));

//    type correlation = exponential_correlation(thread_pool_device, x, y);
    //@todo(assert_true(correlation > static_cast<type>(0.999999), LOG);)

    // Test missing values

       Tensor<type, 1> vector;
       vector.resize(5);
       vector[0] = 1;
       vector[1] = 2;
       vector[2] = 3;
       vector[3] = 4;
       vector[4] = static_cast<type>(NAN);
       Tensor<type, 1> target(vector.size());

       for(Index i = 0; i < vector.size(); i++)
           target[i] = exp(static_cast<type>(2.5)*vector[i] + static_cast<type>(1.4));

//       type exponential_correlation = exponential_correlation(vector, target);
//       assert_true(abs(exponential_correlation - 1.0) < 1.0e-3, LOG );
}


void CorrelationsTest::test_linear_regression()
{
    cout << "test_linear_regression\n";

    // Device

//    Tensor<type, 1> vector_1(4);
//    vector_1.setValues({10, 16, 25, 40, 60});
//    Tensor<type, 1> vector_2(5);
//    vector_2.setValues({94, 118, 147, 180, 230});

//    const type solution1 = static_cast<type>(74.067);
//    const type solution2 = static_cast<type>(2.6402);
//    const type solution3 = static_cast<type>(0.99564);

//    RegressionResults lr = linear_regression(thread_pool_device, vector_1,vector_2);
//    assert_true(lr.a - solution1 <= 0.01, LOG);
//    assert_true(lr.b - solution2 <= 0.01, LOG);
//    assert_true(lr.correlation - solution3 <= 0.01, LOG);

    Tensor<type, 1> vector_1(6);
    vector_1.setValues({10, 16, 25, 40, 60, NAN});
    Tensor<type, 1> vector_2(6);
    vector_2.setValues({94, 118, 147, 180, 230, 100});

    const type solution1 = static_cast<type>(74.067);
    const type solution2 = static_cast<type>(2.6402);
    const type solution3 = static_cast<type>(0.99564);


    RegressionResults lr = linear_regression(thread_pool_device,vector_1,vector_2);

    //@todo(assert_true(abs(lr.a - solution1) <= 0.01, LOG);)
    //@todo(assert_true(abs(lr.b - solution2) <= 0.01, LOG);)
    assert_true(abs(lr.correlation - solution3) <= 0.01, LOG);
}


void CorrelationsTest::test_exponential_regression()
{
    cout << "test_exponential_regression\n";

    Tensor<type, 1> vector_1(5);
    vector_1.setValues({10, 16, 25, 40, 60});

    Tensor<type, 1> vector_2(5);
    vector_2.setValues({94, 118, 147, 180, 230});

    const type solution1 = static_cast<type>(87.805);
    const type solution2 = static_cast<type>(0.017);
    const type solution3 = static_cast<type>(0.9754);

    RegressionResults er = exponential_regression(thread_pool_device,vector_1,vector_2);

    assert_true(er.a - solution1 <= static_cast<type>(0.01), LOG);
    //@todo(assert_true(er.b - solution2 <= static_cast<type>(0.01), LOG);)
    assert_true(er.correlation - solution3 <= static_cast<type>(0.1), LOG);

    vector_1.resize(6);
    vector_1.setValues({10, 16, 25, 40, 60, NAN});

    vector_2.resize(6);
    vector_2.setValues({94, 118, 147, 180, 230, 100});

//    solution1 = static_cast<type>(87.805);
//    solution2 = static_cast<type>(0.017);
//    solution3 = static_cast<type>(0.9754);

//    RegressionResults er = exponential_regression_missing_values(vector_1,vector_2);

//    assert_true(er.a - solution1 <= 0.01, LOG);
//    assert_true(er.b - solution2 <= 0.01, LOG);
//    assert_true(er.correlation - solution3 <= 0.1, LOG);
}


void CorrelationsTest::test_logarithmic_regression()
{
    cout << "test_logarithmic_regression\n";

    Tensor<type, 1> vector_1(5);
    vector_1.setValues({10, 16, 25, 40, 60});

    Tensor<type, 1> vector_2(5);
    vector_2.setValues({94, 118, 147, 180, 230});

    type solution1 = static_cast<type>(-83.935);
    type solution2 = static_cast<type>(73.935);
    type solution3 = static_cast<type>(0.985799);

    RegressionResults lr = logarithmic_regression(thread_pool_device, vector_1,vector_2);

    assert_true(lr.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(lr.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(lr.correlation - solution3 <= static_cast<type>(0.01), LOG);

    vector_1.resize(6);
    vector_1.setValues({10, 16, 25, 40, 60, NAN});
    vector_2.resize(6);
    vector_2.setValues({94, 118, 147, 180, 230, 100});

    solution1 = static_cast<type>(-83.935);
    solution2 = static_cast<type>(73.935);
    solution3 = static_cast<type>(0.985799);

    lr = logarithmic_regression(thread_pool_device,vector_1,vector_2);

    assert_true(lr.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(lr.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(lr.correlation - solution3 <= static_cast<type>(0.01), LOG);
}


void CorrelationsTest::test_power_regression()
{
    cout << "test_power_regression\n";

    Tensor<type, 1> vector_1(5);
    vector_1.setValues({10, 16, 25, 40, 60});
    Tensor<type, 1> vector_2(5);
    vector_2.setValues({94, 118, 147, 180, 230});

    type solution1 = static_cast<type>(30.213);
    type solution2 = static_cast<type>(0.491);
    type solution3 = static_cast<type>(0.998849);

    RegressionResults regression_results = power_regression(thread_pool_device,vector_1,vector_2);

    assert_true(regression_results.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(regression_results.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(regression_results.correlation - solution3 <= static_cast<type>(0.01), LOG);

    vector_1.resize(6);
    vector_1.setValues({10, 16, 25, 40, 60, NAN});
    vector_2.resize(6);
    vector_2.setValues({94, 118, 147, 180, 230, 100});

    solution1 = static_cast<type>(30.213);
    solution2 = static_cast<type>(0.491);
    solution3 = static_cast<type>(0.998849);

//    RegressionResults regression_results = power_regression_missing_values(vector_1,vector_2);

//    assert_true(pr.a - solution1 <= 0.01, LOG);
//    assert_true(pr.b - solution2 <= 0.01, LOG);
//    assert_true(pr.correlation - solution3 <= 0.01, LOG);
}


void CorrelationsTest::test_logistic_regression()
{
    cout << "test_logistic_regression \n";

    const Index size = 100;

    Tensor<type, 1> x(size);

    Tensor<type, 1> y(size);

    for(Index i = 0; i < size; i++)
    {
        x[i] = i + 1;

        if(i < size/2) y[i] = 0;
        else y[i] = 1;
    }

//    RegressionResults log = logistic_regression(thread_pool_device, x, y);

//    assert_true(abs(log.correlation - static_cast<type>(0.95)) <= static_cast<type>(0.01), LOG);
}


void CorrelationsTest::test_covariance()
{
    cout << "test_covariance\n";

    Index size = 100;
    Tensor<type, 1> x(size);
    initialize_sequential(x);
    Tensor<type, 1> y(size);
    for(Index i = 0; i < size; i++) y(i) = i;

    type covariance = OpenNN::covariance(x,y);

    //@todo(assert_true(abs(covariance - static_cast<type>(841.6666666666666)) < static_cast<type>(1.0e-3), LOG);)
    //@todo(assert_true(covariance < 842, LOG);)

    x.resize(size);
    initialize_sequential(x);
    y.resize(size);
    initialize_sequential(x);

    covariance = OpenNN::covariance(x,y);
    covariance = OpenNN::covariance(x,y);

    //@todo(assert_true(abs(covariance_missing_values - static_cast<type>(841.6666666666666)) < static_cast<type>(1.0e-3), LOG);)
    //@todo(assert_true(abs(covariance_missing_values - covariance) < static_cast<type>(1.0e-3), LOG);)
}


void CorrelationsTest::test_covariance_matrix()
{
    cout << "test_covariance_matrix\n";

    Index size = 2;
    Tensor<type, 1> vector_1(size);
    vector_1.setConstant(1.0);
    Tensor<type, 1> vector_2(size);
    vector_2.setConstant( 1.0);

    Tensor<type, 2> matrix(size, 2);
//    matrix.set_column(0, vector_1);
//    matrix.set_column(1, vector_2);

    Tensor<type, 2> matrix_solution(2, 2);
    matrix_solution.setConstant( 0.0);
    Tensor<type, 2> covarianze_matrix;

    covarianze_matrix = covariance_matrix(matrix);

    assert_true(covarianze_matrix(0, 0) == matrix_solution(0, 0), LOG);
    assert_true(covarianze_matrix(0, 1) == matrix_solution(0, 1), LOG);
    assert_true(covarianze_matrix(1, 0) == matrix_solution(1, 0), LOG);
    assert_true(covarianze_matrix(1, 1) == matrix_solution(1, 1), LOG);
}


void CorrelationsTest::test_less_rank_with_ties()
{
    cout << "test_less_rank_with_ties\n";

    Tensor<type, 1> vector_1(3);
    vector_1.setValues({0.0, 1, 2.0});
    Tensor<type, 1> vector_2(3);
    vector_2.setValues({5, 2, 3});
    Tensor<type, 1> vector3(2);
    vector3.setValues({3, 4.0});
    Tensor<type, 1> vector_solution_1(3);
    vector_solution_1.setValues({0.0, 1, 2.0});
    Tensor<type, 1> vector_solution_2(3);
    vector_solution_2.setValues({2, 0, 1});

    Tensor<type, 1> ranks_1;
    Tensor<type, 1> ranks_2;
    Tensor<type, 1> ranks_3;

//    type average_ranks;

    ranks_1 = less_rank_with_ties(vector_1);
    ranks_2 = less_rank_with_ties(vector_2);
    ranks_3 = less_rank_with_ties(vector3);

//    average_ranks = mean(ranks_3);
//    Tensor<type, 1> vector_solution_3(3, average_ranks);

//    assert_true(ranks_1 == vector_solution_1 + 1, LOG);
//    assert_true(ranks_2 == vector_solution_2 + 1, LOG);
//    assert_true(ranks_3 == vector_solution_3 , LOG);
}


void CorrelationsTest::test_contingency_table()
{
    cout << "test_contingency_table\n";

    Tensor<string, 1> vector_1(4);
    vector_1.setValues({"a", "b", "b", "a" });
    Tensor<string, 1> vector_2(4);
    vector_2.setValues({"c", "c", "d", "d" });

//    assert_true(contingency_table(vector_1, vector_2) == matrix, LOG);

}


void CorrelationsTest::test_chi_square_test()
{
    cout << "test_chi_square_test\n";

//    Tensor<string, 1> vector_1({"a", "b", "b", "a" });
//    Tensor<string, 1> vector_2({"c", "c", "d", "d" });

//    assert_true(abs(chi_square_test(contingency_table(vector_1, vector_2).to_type_matrix()) - 0.0) < 1.0e-3, LOG);

}


void CorrelationsTest::test_karl_pearson_correlation()
{
    cout << "test_karl_pearson_correlation\n";

    Tensor<type, 2> matrix1(4,2);
    Tensor<type, 2> matrix2(4,2);

//    matrix1.set_column(0, {1, 0, 1, 0});
//    matrix1.set_column(1, {0, 1, 0, 1});

//    matrix2.set_column(0, {1, 0, 1, 0});
//    matrix2.set_column(1, {0, 1, 0, 1});

//    const type solution = 1;

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


void CorrelationsTest::run_test_case()
{
   cout << "Running correlation analysis test case...\n";

   // Linear correlation methods

   test_linear_correlation();
   test_spearman_linear_correlation();
   test_rank_linear_correlation();

   // Logistic correlation methods

   test_logistic_correlation();
   test_rank_logistic_correlation();

   // Logarithmic correlation methods

   test_logarithmic_correlation();

   // Exponential correlation methods

   test_exponential_correlation();

   // Regression Methods

   test_linear_regression();
   test_exponential_regression();
   test_logarithmic_regression();
   test_power_regression();
   test_logistic_regression();

   // Time series correlation methods

   test_autocorrelation();
   test_cross_correlations();

   // Covariance

   test_covariance();
   test_covariance_matrix();
   test_less_rank_with_ties();

   // Contingency table

   test_contingency_table();
   test_chi_square_test();
   test_karl_pearson_correlation();

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
