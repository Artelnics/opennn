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
    cout << "test_calculate_linear_correlation\n";

    // Perfect case

    Tensor<type, 1> vector1(10);
    vector1.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    Tensor<type, 1> vector2(10);
    vector2.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    const type solution = 1;
    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    assert_true(linear_correlation(thread_pool_device, vector1, vector2) - solution <= 0.000001, LOG);

    Tensor<type, 1> vector3(10);
    vector3.setValues({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});

    assert_true(linear_correlation(thread_pool_device, vector1, vector3) + solution <= static_cast<type>(0.000001), LOG);

    // General case

    Index size = 100;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> y(size);

    type correlation;

    for(Index i = 0; i < size; i++) y[i] = 2*x[i];

    correlation = linear_correlation(thread_pool_device,x,y);

   assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    y = -1.0*x;

    correlation = linear_correlation(thread_pool_device,x,y);
    assert_true(abs(correlation + static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

}


void CorrelationsTest::test_spearman_linear_correlation()
{
    cout << "test_calculate_spearman_linear_correlation\n";

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Index size = 100;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> y(size);

    type correlation;

    for(Index i = 0; i < size; i++) y[i] = 2*x[i];

    correlation = rank_linear_correlation(thread_pool_device,x,y);

   //@todo(assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)
   //@todo(assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)

    y = -1.0*x;

    assert_true(abs(rank_linear_correlation(thread_pool_device,x,y)) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
    assert_true(abs(correlation) <= static_cast<type>(1.0), LOG);

    y.setConstant(static_cast<type>(0.1));
    correlation = rank_linear_correlation(thread_pool_device,x,y);
    //@todo(assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)
    assert_true(abs(correlation) <= static_cast<type>(1.0), LOG);
}

void CorrelationsTest::test_linear_correlation_missing_values()
{
    cout << "test_calculate_linear_correlation_missing_values\n";

//Test missing values

    Tensor<type, 1> vector;
    vector.resize(5);
    vector[0] = 1;
    vector[1] = 2;
    vector[2] = 3;
    vector[3] = 4;
    vector[4] = 5;

    Tensor<type, 1> target;
    target.resize(5);
    target[0] = 1;
    target[1] = 2;
    target[2] = 3;
    target[3] = static_cast<type>(NAN);
    target[4] = 5;

    //@todo(type linear_correlation = linear_correlation_missing_values(thread_pool_device, vector, target);)

    //@todo(assert_true(abs(linear_correlation - 1.0) < 1.0e-3, LOG );)
}


void CorrelationsTest::test_rank_linear_correlation()
{
    cout << "test_calculate_rank_linear_correlation()\n";

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Index size = 10;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> y(size);

    type correlation;

    for(Index i = 0; i < size; i++) y[i] = 2*x[i];

    correlation = rank_linear_correlation(thread_pool_device,x,y);

    //@todo(assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    y = -1.0*x;

    correlation = rank_linear_correlation(thread_pool_device,x,y);
    //@todo(assert_true(abs(correlation + static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)
    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
}


void CorrelationsTest::test_logistic_correlation()
{
    cout << "test_logistic_correlation\n";

    const Index size = 100;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();

    Tensor<type, 1> y(size);
    type correlation;
    y.setConstant(0.0);

    for(Index i= size - (size/2); i < size; i++) y[i] = 1;

//    correlation = logistic_correlation_missing_values(x.to_column_matrix(), y);

//    assert_true(correlation <= 0.95, LOG);

//    y.setConstant(1.0);

//    for(Index i= size - (size/2); i < size; i++) y[i] = 0;

//    correlation = logistic_correlation(x.to_column_matrix(), y);

//    cout << correlation << endl;

//    assert_true(correlation >= -0.99, LOG);

//    y.setConstant(0.0);

//    correlation = logistic_correlation(x.to_column_matrix(), y);

//    assert_true(abs(correlation - 0.0) < numeric_limits<type>::min(), LOG);
}


void CorrelationsTest::test_rank_logistic_correlation()
{
    cout << "test_calculate_rank_logistic_correlation\n";

    const Index size = 10;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();

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


void CorrelationsTest::test_logistic_error_gradient()
{
    cout << "test_logistic_error_gradient\n";

    Tensor<type, 1> solution(2);

//    solution.setValues({0.09296,0.152088});

    const Tensor<type, 1> error_gradient;// = logistic_error_gradient(0,1,{1, 2, 3, 4, 5, 6}, {0, 0, 0, 1, 1, 1});

//    assert_true(abs(solution[0] - error_gradient[0]) <= static_cast<type>(0.001), LOG);
//    assert_true(abs(solution[1] - error_gradient[1]) <= static_cast<type>(0.001), LOG);

}


void CorrelationsTest::test_logistic_error_gradient_missing_values()
{
    cout << "test_logistic_error_gradient\n";

    Tensor<type, 1> solution(2);

//    solution.setValues({0.09296,0.152088});

    const Tensor<type, 1> error_gradient;// = logistic_error_gradient_missing_values(0,1,{1, 2, 3, 4, 5, 6, NAN}, {0, 0, 0, 1, 1, 1, 1});

    assert_true(abs(solution[0] - error_gradient[0]) <= static_cast<type>(0.001), LOG);
    assert_true(abs(solution[1] - error_gradient[1]) <= static_cast<type>(0.001), LOG);
}


void CorrelationsTest::test_rank_linear_correlation_missing_values()
{
    cout << "test_calculate_rank_linear_correlation_missing_values\n";

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Index size = 10;

    Tensor<type, 1> x(size);
    Tensor<type, 1> y(size);
    Tensor<Index, 1> z(0);

//    x.initialize_sequential();
    z.setRandom();

    type correlation;

    y = 2.0*x;

    correlation = rank_linear_correlation_missing_values(thread_pool_device,x,y);

    //@todo(assert_true(abs(correlation - static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)
    assert_true(abs(correlation) <= static_cast<type>(1.0), LOG);

    y = -1.0*x;

    correlation = rank_linear_correlation_missing_values(thread_pool_device, x, y);
    //@todo(assert_true(abs(correlation + static_cast<type>(1.0)) < numeric_limits<type>::min(), LOG);)
    assert_true(abs(correlation) <= static_cast<type>(1.0), LOG);

    //Test missing values
    Tensor<type, 1> vector;
    vector.resize(5);
    vector[0] = 1;
    vector[1] = 2;
    vector[2] = 3;
    vector[3] = 4;
    vector[4] = 5;

    Tensor<type, 1> target;
    target.resize(5);
    target[0] = 1;
    target[1] = 2;
    target[2] = 3;
    target[3] = static_cast<type>(NAN);
    target[4] = 5;

    type rank_linear_correlation = rank_linear_correlation_missing_values(thread_pool_device, vector, target);

    assert_true((rank_linear_correlation - static_cast<type>(1.0)) < static_cast<type>(1.0e-3), LOG );

    // Test with ties
    Tensor<type, 1> vector_ties;
    vector_ties.resize(5);
    vector_ties[0] = 1;
    vector_ties[1] = 1;
    vector_ties[2] = 3;
    vector_ties[3] = static_cast<type>(NAN);
    vector_ties[4] = 5;

    Tensor<type, 1> target_ties;
    target_ties.resize(5);
    target_ties[0] = 1;
    target_ties[1] = 1;
    target_ties[2] = 3;
    target_ties[3] = static_cast<type>(NAN);
    target_ties[4] = 5;

    type rank_linear_correlation_ties = rank_linear_correlation_missing_values(thread_pool_device, vector_ties, target_ties);

    //@todo(assert_true(abs(rank_linear_correlation_ties - static_cast<type>(1.0)) < static_cast<type>(1.0e-3), LOG );)
}


void CorrelationsTest::test_logistic_correlation_missing_values()
{
    cout << "test_logistic_correlation_missing_values\n";

    Index size;

    Tensor<type, 1> x;

    Tensor<type, 1> y;

    type correlation = 0.0;

    // Test

    size = 100;

    x.resize(size);
//    x.initialize_sequential();

    y.resize(size);

    for(Index i = 0; i < size/2; i++) y[i] = 0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

//    correlation = logistic_correlation_missing_values(x.to_column_matrix(),y);

    assert_true(correlation <= static_cast<type>(0.95), LOG);

    // Test

    for(Index i = 0; i < size/2; i++) y[i] = 1.0;

    for(Index i= size - (size/2); i < size; i++) y[i] = 0.0;

//    correlation = logistic_correlation_missing_values(x.to_column_matrix(),y);

    assert_true(correlation >= static_cast<type>(-0.95), LOG);

    // Test

    y.setConstant(0.0);

//    correlation = logistic_correlation_missing_values(x.to_column_matrix(),y);

    assert_true(abs(correlation) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

    // Test

//    y.randomize_binary();

//    correlation = logistic_correlation_missing_values(x.to_column_matrix(),y);

    assert_true(correlation < static_cast<type>(0.3), LOG);
}


void CorrelationsTest::test_autocorrelation()
{
    cout << "test_calculate_autocorrelation\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> correlations;

    correlations = autocorrelations(x, size/100);
    //@todo(assert_true(minimum(correlations) > static_cast<type>(0.9), LOG);)
}


void CorrelationsTest::test_cross_correlations()
{
    cout << "test_calculate_autocorrelation\n";

    Index size = 1000;
    Tensor<type, 1> x(size);
    Tensor<type, 1> y(size);

//    x.initialize_sequential();
//    y.initialize_sequential();

    Tensor<type, 1> cros_correlations;

    cros_correlations = cross_correlations(x, y, 10);
    //@todo(assert_true(cros_correlations(0) < 5.0, LOG);)
    //@todo(assert_true(cros_correlations(1) > 0.9, LOG);)
}


void CorrelationsTest::test_logarithmic_correlation()
{
    cout << "test_calculate_logarithmic_correlation\n";

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

//    // Perfect case

    const Tensor<type, 1> vector1;//({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    const Tensor<type, 1> vector2;//({0, 0.30103, 0.477121, 0.60206, 0.69897, 0.778151, 0.845098, 0.90309, 0.954243, 1});

    const type solution = 1;

    const type correl = logarithmic_correlation(thread_pool_device, vector1, vector2) - solution;

    assert_true(correl - solution <= static_cast<type>(0.0001), LOG);

    const Tensor<type, 1> vector_1;//({100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,});
    const Tensor<type, 1> vector_2;//({1002,1002.004321,1002.0086,1002.012837,1002.017033,1002.021189,1002.025306,1002.029384,1002.033424,1002.037426,1002.041393,1002.045323,1002.049218,1002.053078,1002.056905,1002.060698,1002.064458,1002.068186,});

    assert_true(logarithmic_correlation(thread_pool_device, vector_1, vector_2) - solution <= static_cast<type>(0.00001), LOG);
}


void CorrelationsTest::test_logarithmic_correlation_missing_values()
{
    cout << "test_calculate_logarithmic_correlation_missing_values\n";

       //Test missing values

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
    cout << "test_calculate_exponential_correlation\n";

    const Index size = 100;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> y(size);

    for(Index i = 0; i < size; i++)
    {
        y[i] = exp(static_cast<type>(2.5)*x[i] + static_cast<type>(1.4));
    }

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    type correlation = exponential_correlation(thread_pool_device,x,y);
    //@todo(assert_true(correlation > static_cast<type>(0.999999), LOG);)
}


void CorrelationsTest::test_exponential_correlation_missing_values()
{
    cout << "test_calculate_exponential_correlation_missing_values\n";

    //Test missing values

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
           target[i] = exp(static_cast<type>(2.5)*vector[i] + static_cast<type>(1.4));
       }

//       type exponential_correlation = exponential_correlation_missing_values(vector, target);
//       assert_true(abs(exponential_correlation - 1.0) < 1.0e-3, LOG );
}


void CorrelationsTest::test_linear_regression()
{
    cout << "test_linear_regression\n";

    // Device

//    const int n = omp_get_max_threads();
//    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
//    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

//    Tensor<type, 1> vector1(4);
//    vector1.setValues({10, 16, 25, 40, 60});
//    Tensor<type, 1> vector2(5);
//    vector2.setValues({94, 118, 147, 180, 230});

//    const type solution1 = static_cast<type>(74.067);
//    const type solution2 = static_cast<type>(2.6402);
//    const type solution3 = static_cast<type>(0.99564);

//    RegressionResults lr = linear_regression(thread_pool_device, vector1,vector2);
//    assert_true(lr.a - solution1 <= 0.01, LOG);
//    assert_true(lr.b - solution2 <= 0.01, LOG);
//    assert_true(lr.correlation - solution3 <= 0.01, LOG);
}


void CorrelationsTest::test_linear_regression_missing_values()
{
    cout << "test_linear_regression_missing_values\n";

    // Device
    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(6);
    vector1.setValues({10, 16, 25, 40, 60, NAN});
    Tensor<type, 1> vector2(6);
    vector2.setValues({94, 118, 147, 180, 230, 100});

    const type solution1 = static_cast<type>(74.067);
    const type solution2 = static_cast<type>(2.6402);
    const type solution3 = static_cast<type>(0.99564);


    RegressionResults lr = linear_regression(thread_pool_device,vector1,vector2);

    //@todo(assert_true(abs(lr.a - solution1) <= 0.01, LOG);)
    //@todo(assert_true(abs(lr.b - solution2) <= 0.01, LOG);)
    assert_true(abs(lr.correlation - solution3) <= 0.01, LOG);
}


void CorrelationsTest::test_exponential_regression()
{
    cout << "test_exponential_regression\n";

    // Device
    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(5);
    vector1.setValues({10, 16, 25, 40, 60});
    Tensor<type, 1> vector2(5);
    vector2.setValues({94, 118, 147, 180, 230});

    const type solution1 = static_cast<type>(87.805);
    const type solution2 = static_cast<type>(0.017);
    const type solution3 = static_cast<type>(0.9754);


    RegressionResults er = exponential_regression(thread_pool_device,vector1,vector2);

    assert_true(er.a - solution1 <= static_cast<type>(0.01), LOG);
    //@todo(assert_true(er.b - solution2 <= static_cast<type>(0.01), LOG);)
    assert_true(er.correlation - solution3 <= static_cast<type>(0.1), LOG);
}


void CorrelationsTest::test_exponential_regression_missing_values()
{
    cout << "test_exponential_regression_missing_values\n";

    // Device
    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(6);
    vector1.setValues({10, 16, 25, 40, 60, NAN});
    Tensor<type, 1> vector2(6);
    vector2.setValues({94, 118, 147, 180, 230, 100});

    const type solution1 = static_cast<type>(87.805);
    const type solution2 = static_cast<type>(0.017);
    const type solution3 = static_cast<type>(0.9754);

//    RegressionResults er = exponential_regression_missing_values(vector1,vector2);

//    assert_true(er.a - solution1 <= 0.01, LOG);
//    assert_true(er.b - solution2 <= 0.01, LOG);
//    assert_true(er.correlation - solution3 <= 0.1, LOG);
}


void CorrelationsTest::test_logarithmic_regression() // @todo
{
    cout << "test_logarithmic_regression\n";

    // Device

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(5);
    vector1.setValues({10, 16, 25, 40, 60});
    Tensor<type, 1> vector2(5);
    vector2.setValues({94, 118, 147, 180, 230});

    const type solution1 = static_cast<type>(-83.935);
    const type solution2 = static_cast<type>(73.935);
    const type solution3 = static_cast<type>(0.985799);

    RegressionResults lr = logarithmic_regression(thread_pool_device, vector1,vector2);

    assert_true(lr.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(lr.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(lr.correlation - solution3 <= static_cast<type>(0.01), LOG);
}


void CorrelationsTest::test_logarithmic_regression_missing_values() // @todo
{
    cout << "test_logarithmic_regression_missing_values\n";

    // Device

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(6);
    vector1.setValues({10, 16, 25, 40, 60, NAN});
    Tensor<type, 1> vector2(6);
    vector2.setValues({94, 118, 147, 180, 230, 100});

    const type solution1 = static_cast<type>(-83.935);
    const type solution2 = static_cast<type>(73.935);
    const type solution3 = static_cast<type>(0.985799);


    RegressionResults lr = logarithmic_regression(thread_pool_device,vector1,vector2);

    assert_true(lr.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(lr.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(lr.correlation - solution3 <= static_cast<type>(0.01), LOG);
}


void CorrelationsTest::test_power_regression() // @todo
{
    cout << "test_power_regression\n";

    // Device

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(5);
    vector1.setValues({10, 16, 25, 40, 60});
    Tensor<type, 1> vector2(5);
    vector2.setValues({94, 118, 147, 180, 230});

    const type solution1 = static_cast<type>(30.213);
    const type solution2 = static_cast<type>(0.491);
    const type solution3 = static_cast<type>(0.998849);


    RegressionResults pr = power_regression(thread_pool_device,vector1,vector2);

    assert_true(pr.a - solution1 <= static_cast<type>(0.01), LOG);
    assert_true(pr.b - solution2 <= static_cast<type>(0.01), LOG);
    assert_true(pr.correlation - solution3 <= static_cast<type>(0.01), LOG);

}


void CorrelationsTest::test_power_regression_missing_values()
{
    cout << "test_power_regression_missing_values\n";

    // Device

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    Tensor<type, 1> vector1(6);
    vector1.setValues({10, 16, 25, 40, 60, NAN});
    Tensor<type, 1> vector2(6);
    vector2.setValues({94, 118, 147, 180, 230, 100});

    const type solution1 = static_cast<type>(30.213);
    const type solution2 = static_cast<type>(0.491);
    const type solution3 = static_cast<type>(0.998849);

//    RegressionResults pr = power_regression_missing_values(vector1,vector2);

//    assert_true(pr.a - solution1 <= 0.01, LOG);
//    assert_true(pr.b - solution2 <= 0.01, LOG);
//    assert_true(pr.correlation - solution3 <= 0.01, LOG);
}


void CorrelationsTest::test_logistic_regression() // @todo
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

    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

//    RegressionResults log = logistic_regression(thread_pool_device,x,y);

//    assert_true(abs(log.correlation - static_cast<type>(0.95)) <= static_cast<type>(0.01), LOG);
}


void CorrelationsTest::test_covariance()
{
    cout << "test_calculate_covariance\n";

    Index size = 100;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> y(size);
//    y.initialize_sequential();

    type covariance = OpenNN::covariance(x,y);

    //@todo(assert_true(abs(covariance - static_cast<type>(841.6666666666666)) < static_cast<type>(1.0e-3), LOG);)
    //@todo(assert_true(covariance < 842, LOG);)
}


void CorrelationsTest::test_covariance_missing_values()
{
    cout << "test_covariance_missing_values\n";

    Index size = 100;
    Tensor<type, 1> x(size);
//    x.initialize_sequential();
    Tensor<type, 1> y(size);
//    y.initialize_sequential();

    type covariance = OpenNN::covariance(x,y);
    type covariance_missing_values = OpenNN::covariance_missing_values(x,y);

    //@todo(assert_true(abs(covariance_missing_values - static_cast<type>(841.6666666666666)) < static_cast<type>(1.0e-3), LOG);)
    //@todo(assert_true(abs(covariance_missing_values - covariance) < static_cast<type>(1.0e-3), LOG);)
}


void CorrelationsTest::test_covariance_matrix()
{
    cout << "test_calculate_covariance_matrix\n";

    Index size = 2;
    Tensor<type, 1> vector1(size);
    vector1.setConstant(1.0);
    Tensor<type, 1> vector2(size);
    vector2.setConstant( 1.0);

    Tensor<type, 2> matrix(size, 2);
//    matrix.set_column(0, vector1);
//    matrix.set_column(1, vector2);

    Tensor<type, 2> matrix_solution(2, 2);
    matrix_solution.setConstant( 0.0);
    Tensor<type, 2> covarianze_matrix;

    covarianze_matrix = covariance_matrix(matrix);

    assert_true(covarianze_matrix(0, 0) == matrix_solution(0, 0), LOG);
    assert_true(covarianze_matrix(0, 1) == matrix_solution(0, 1), LOG);
    assert_true(covarianze_matrix(1, 0) == matrix_solution(1, 0), LOG);
    assert_true(covarianze_matrix(1, 1) == matrix_solution(1, 1), LOG);
}


void CorrelationsTest::test_less_rank_with_ties() // @todo
{
    cout << "test_calculate_less_rank_with_ties\n";

    Tensor<type, 1> vector1(3);
    vector1.setValues({0.0, 1, 2.0});
    Tensor<type, 1> vector2(3);
    vector2.setValues({5, 2, 3});
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

    ranks_1 = less_rank_with_ties(vector1);
    ranks_2 = less_rank_with_ties(vector2);
    ranks_3 = less_rank_with_ties(vector3);

//    average_ranks = mean(ranks_3);
//    Tensor<type, 1> vector_solution_3(3, average_ranks);

//    assert_true(ranks_1 == vector_solution_1 + 1, LOG);
//    assert_true(ranks_2 == vector_solution_2 + 1, LOG);
//    assert_true(ranks_3 == vector_solution_3 , LOG);
}


void CorrelationsTest::test_remove_correlations()
{
    cout << "test_remove_correlations\n";
}


void CorrelationsTest::test_contingency_table() // @todo
{
    cout << "test_contingency_table\n";

    Tensor<string, 1> vector_1(4);
    vector_1.setValues({"a", "b", "b", "a" });
    Tensor<string, 1> vector_2(4);
    vector_2.setValues({"c", "c", "d", "d" });
//    Matrix<Index> matrix(2, 2, 1);

//    assert_true(contingency_table(vector_1, vector_2) == matrix, LOG);

}


void CorrelationsTest::test_chi_square_test() // @todo
{
    cout << "test_chi_square_test\n";

//    Tensor<string, 1> vector_1({"a", "b", "b", "a" });
//    Tensor<string, 1> vector_2({"c", "c", "d", "d" });

//    assert_true(abs(chi_square_test(contingency_table(vector_1, vector_2).to_type_matrix()) - 0.0) < 1.0e-3, LOG);

}


void CorrelationsTest::test_chi_square_critical_point()
{
    cout << "test_chi_square_critical_point\n";

    type crit_point;
    type solution = static_cast<type>(14.067);

    crit_point = chi_square_critical_point(static_cast<type>(0.05),7);

    assert_true(crit_point - solution <= static_cast<type>(0.01), LOG);
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
}


void CorrelationsTest::test_karl_pearson_correlation_missing_values() // @todo
{
    cout << "test_karl_pearson_correlation_missing_values\n";

    Tensor<type, 2> matrix1(5,2);
    Tensor<type, 2> matrix2(5,2);

//    matrix1.set_column(0, {1, 0, 1, 0, 0});
//    matrix1.set_column(1, {0, 1, NAN, 1, NAN});

//    matrix2.set_column(0, {1, 0, 1, 0, 0});
//    matrix2.set_column(1, {0, NAN, 0, 1, 1});

//    const type solution = 1;

//    const type correlation = karl_pearson_correlation_missing_values(matrix1, matrix2);

//    assert_true(abs(correlation - solution) <= 0.001, LOG);
}


void CorrelationsTest::run_test_case()
{
   cout << "Running correlation analysis test case...\n";

   // Linear correlation methods

   test_linear_correlation();
   test_spearman_linear_correlation();
   test_linear_correlation_missing_values();
   test_rank_linear_correlation();
   test_rank_linear_correlation_missing_values();


   // Logistic correlation methods

   test_logistic_correlation();
   test_rank_logistic_correlation();
   test_logistic_correlation_missing_values();
   test_logistic_error_gradient();
   test_logistic_correlation_missing_values();


   // Logarithmic correlation methods

   test_logarithmic_correlation();
   test_logarithmic_correlation_missing_values();


   // Exponential correlation methods

   test_exponential_correlation();
   test_exponential_correlation_missing_values();


   // Regression Methods

   test_linear_regression();
   test_linear_regression_missing_values();
   test_exponential_regression();
   test_exponential_correlation_missing_values();
   test_logarithmic_regression();
   test_logarithmic_regression_missing_values();
   test_power_regression();
   test_power_regression_missing_values();
   test_logistic_regression();


   // Time series correlation methods

   test_autocorrelation();
   test_cross_correlations();


   // Covariance

   test_covariance();
   test_covariance_missing_values();
   test_covariance_matrix();
   test_less_rank_with_ties();


   //Remove methods

   test_remove_correlations();


   //Contingency table

   test_contingency_table();
   test_chi_square_test();
   test_chi_square_critical_point();
   test_karl_pearson_correlation();
   test_karl_pearson_correlation_missing_values();

   cout << "End of correlation analysis test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C); 2005-2020 Artificial Intelligence Techniques, SL.
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
