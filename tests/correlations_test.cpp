#include "pch.h"

#include "../opennn/config.h"
#include "../opennn/correlations.h"
#include "../opennn/tensors.h"
#include "../opennn/statistics.h"


TEST(CorrelationsTest, SpearmanCorrelations)
{
/*
    Tensor<type, 1> x(10);
    x.setValues({ type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10) });
    Tensor<type, 1> y(10);
    y.setValues({ type(1), type(3), type(7), type(9), type(10), type(16), type(20), type(28), type(44), type(100) });

//    type solution = type(1);

//    EXPECT_EQ(linear_correlation_spearman(thread_pool_device, x, y).r - solution < type(NUMERIC_LIMITS_MIN));
*/
}


/*
namespace opennn
{

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

    EXPECT_EQ(linear_correlation(thread_pool_device, x, y).r - solution < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(linear_correlation(thread_pool_device, x, y).r - solution < type(NUMERIC_LIMITS_MIN));

    const Tensor<type, 1> x1 = calculate_rank_greater(x).cast<type>();
    const Tensor<type, 1> y1 = calculate_rank_greater(y).cast<type>();

    // Test

    y.setValues({type(10), type(9), type(8),type( 7),type( 6),type( 5),type( 4),type( 3),type( 2),type( 1)});

    EXPECT_EQ(linear_correlation(thread_pool_device, x, y).r + solution < type(NUMERIC_LIMITS_MIN));

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);
    y = type(2)*x;

    correlation = linear_correlation(thread_pool_device, x, y).r;

    EXPECT_EQ(abs(correlation - type(1)) < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(abs(correlation) - type(1) < type(NUMERIC_LIMITS_MIN));

    // Test

    y = type(-1.0)*x;

    correlation = linear_correlation(thread_pool_device, x, y).r;
    EXPECT_EQ(abs(correlation + type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(correlation) - type(1) < type(NUMERIC_LIMITS_MIN));
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

    EXPECT_EQ(abs(correlation.r) <= type(0.1));
    EXPECT_EQ((correlation.form == Correlation::Form::Logistic));

    // Test

    size = 10;

    x.resize(size);
    x.setValues({-5,-4,-3,-2,-1,1,2,3,4,5});

    y.resize(size);
    y.setValues({0,0,0,0,0,1,1,1,1,1});

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    EXPECT_EQ(correlation.r >= type(0.9));
    EXPECT_EQ((correlation.form == Correlation::Form::Logistic));

    y.setConstant(type(0));

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    EXPECT_EQ(correlation.r - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((correlation.form == Correlation::Form::Logistic));

    // Test

    size = 100;

    x.resize(size);
    initialize_sequential(x);

    y.resize(size);

    for(Index i = 0; i < size/2; i++) y[i] = 0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    EXPECT_EQ(correlation.r <= type(1));

    for(Index i = 0; i < size; i++)
        y[i] = exp(type(2.5)*x[i] + type(1.4));

    const unsigned int threads_number = thread::hardware_concurrency();

    ThreadPool* thread_pool = new ThreadPool(threads_number);

    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, threads_number);

    // Test

    for(Index i = 0; i < size/2; i++) y[i] = 1.0;

    for(Index i = size - (size/2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation_vector_vector(thread_pool_device, x, y);

    EXPECT_EQ(abs(correlation.r) >= type(-0.95));

    // Test

    y.setConstant(type(0));

    correlation = logistic_correlation_vector_vector(thread_pool_device, x,y);

    EXPECT_EQ(isnan(correlation.r));

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

    EXPECT_EQ(abs(correlation.r - solution) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(correlation.b - type(4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(correlation.a - type(0)) < type(NUMERIC_LIMITS_MIN));
}


void CorrelationsTest::test_exponential_correlation()
{
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

    EXPECT_EQ(abs(correlation.r - type(1))< type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(correlation.a - type(1))< type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(correlation.b - type(0.5)) < type(NUMERIC_LIMITS_MIN));

    // Test missing values

    size = 5;

    x.resize(size);
    x.setValues({ type(1),type(2),type(3),type(4),type(NAN)});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = type(1.4) * exp(type(2.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    EXPECT_EQ(abs(correlation.r - type(1)) < type(1.0e-3));
    EXPECT_EQ(correlation.b - type(2.5)< type(NUMERIC_LIMITS_MIN));
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

    EXPECT_EQ(correlation.r > type(0.999999));
    EXPECT_EQ(correlation.a - type(1)< type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(correlation.b - type(2)< type(NUMERIC_LIMITS_MIN));
}

void CorrelationsTest::test_autocorrelations()
{
    Index size = 1000;
    Tensor<type, 1> x(size);
    initialize_sequential(x);
    Tensor<type, 1> correlations;

    correlations = autocorrelations(thread_pool_device,x, size/100);
    EXPECT_EQ(minimum(correlations) > type(0.9));
}


void CorrelationsTest::test_cross_correlations()
{
    Index size = 1000;
    Tensor<type, 1> x(size);
    Tensor<type, 1> y(size);

    initialize_sequential(x);
    for(Index i = 0; i < size; i++) y(i) = type(i);

    Tensor<type, 1> cros_correlations;

    cros_correlations = cross_correlations(thread_pool_device,x, y, 10);
    EXPECT_EQ(cros_correlations(0) < 5.0);
    EXPECT_EQ(cros_correlations(1) > 0.9);

}

}
*/
