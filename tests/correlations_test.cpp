#include "pch.h"


#include "../opennn/correlations.h"
#include "../opennn/tensors.h"
#include "../opennn/statistics.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/scaling_layer_2d.h"
#include "../opennn/probabilistic_layer.h"
#include "../opennn/strings_utilities.h"

using namespace opennn;

class CorrelationsTest : public ::testing::Test 
{
protected:
    
    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    void SetUp() override {
        const unsigned int threads_number = thread::hardware_concurrency();
        thread_pool = make_unique<ThreadPool>(threads_number);
        thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);
    }

    void TearDown() override {
    }
};


TEST_F(CorrelationsTest, SpearmanCorrelation)
{

    Tensor<type, 1> x(10);
    x.setValues({ type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10) });

    Tensor<type, 1> y(10);
    y.setValues({ type(1), type(4), type(9), type(16), type(25), type(36), type(49), type(64), type(81), type(100) });
    
    Correlation result = linear_correlation_spearman(thread_pool_device.get(), x, y);

    EXPECT_NEAR(result.r, type(1), NUMERIC_LIMITS_MIN);

}


TEST_F(CorrelationsTest, LinearCorrelation)
{
    Tensor<type, 1> x(10);
    x.setValues({ type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10) });

    Tensor<type, 1> y(10);
    y.setValues({ type(10), type(20), type(30),type(40),type(50),type(60),type(70),type(80),type(90),type(100) });

    EXPECT_NEAR(linear_correlation(thread_pool_device.get(), x, y).r, type(1), NUMERIC_LIMITS_MIN);
    
    y.setValues({ type(10), type(9), type(8),type(7),type(6),type(5),type(4),type(3),type(2),type(1) });

    EXPECT_NEAR(linear_correlation(thread_pool_device.get(), x, y).r, type(- 1), NUMERIC_LIMITS_MIN);
    
    // Test

    x.setRandom();
    y.setRandom();

    EXPECT_NE(linear_correlation(thread_pool_device.get(), x, y).r, type(-1));
    EXPECT_NE(linear_correlation(thread_pool_device.get(), x, y).r, type( 0));
    EXPECT_NE(linear_correlation(thread_pool_device.get(), x, y).r, type( 1));
}


TEST_F(CorrelationsTest, LogisticCorrelation)
{
    Tensor<type, 1> x(20);
    x.setValues({ -10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9 });

    Tensor<type, 1> y(20);
    y.setValues({ 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1 });

    Correlation correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    correlation.print(); /*system("pause");*/

    EXPECT_LE(abs(correlation.r), type(0.1));
    EXPECT_EQ(correlation.form, Correlation::Form::Logistic);

    // Test

    Index size = 10;

    x.resize(size);
    x.setValues({ -5,-4,-3,-2,-1,1,2,3,4,5 });

    y.resize(size);
    y.setValues({ 0,0,0,0,0,1,1,1,1,1 });

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    correlation.print();

    EXPECT_GE(correlation.r, type(0.9));
    EXPECT_LE(correlation.r, type(1));
    EXPECT_EQ(correlation.form, Correlation::Form::Logistic);

    // EXPECT_NEAR(correlation.r, type(1), NUMERIC_LIMITS_MIN);

    // Test

    size = 100;

    vector<Index> x_vector(100);

    x.resize(size);

    iota(x_vector.begin(), x_vector.end(),0);
    for(Index i = 0; i < x.size(); i++)
        x(i) = type(x_vector[i]);

    y.resize(size);

    for (Index i = 0; i < size / 2; i++) y[i] = 0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    correlation.print();

    EXPECT_EQ(correlation.r <= type(1), true);

    for (Index i = 0; i < size; i++)
        y[i] = exp(type(2.5) * x[i] + type(1.4));

    // Test

    for (Index i = 0; i < size / 2; i++) y[i] = 1.0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    correlation.print();

    EXPECT_EQ(abs(correlation.r) >= type(0.95), true);
/*
    // Test

    y.setConstant(type(0));

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    EXPECT_EQ(isnan(correlation.r), true);

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    correlation.print();

    EXPECT_GE(correlation.r, type(0.9));
    EXPECT_EQ(correlation.form, Correlation::Form::Logistic);

    y.setConstant(type(0));

    for (Index i = size - (size / 2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    // EXPECT_NEAR(correlation.r, type(1), NUMERIC_LIMITS_MIN);
    EXPECT_EQ(correlation.form, Correlation::Form::Logistic);
/*
    // Test

    size = 100;

    x.resize(size);
    // initialize_sequential(x);

    y.resize(size);

    for (Index i = 0; i < size / 2; i++) y[i] = 0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    EXPECT_LE(correlation.r, type(1));

    for (Index i = 0; i < size; i++)
        y[i] = exp(type(2.5) * x[i] + type(1.4));

    const unsigned int threads_number = thread::hardware_concurrency();

    ThreadPool* thread_pool = new ThreadPool(threads_number);

    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, threads_number);

    // Test

    for (Index i = 0; i < size / 2; i++) y[i] = 1.0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    EXPECT_GE(abs(correlation.r), type(-0.95));

    // Test

    y.setConstant(type(0));

    correlation = logistic_correlation_vector_vector(thread_pool_device.get(), x, y);

    EXPECT_EQ(isnan(correlation.r), true);
*/
}

/*
void CorrelationsTest::test_logarithmic_correlation()
{
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

    EXPECT_NEAR(correlation.r, solution, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(correlation.b, type(4), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(correlation.a, type(0), NUMERIC_LIMITS_MIN);
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

    EXPECT_NEAR(correlation.r, type(1)), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(correlation.a, type(1)), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(correlation.b, type(0.5), NUMERIC_LIMITS_MIN);

    // Test missing values

    size = 5;

    x.resize(size);
    x.setValues({ type(1),type(2),type(3),type(4),type(NAN)});

    y.resize(size);
    for(Index i = 0; i < size; i++) y[i] = type(1.4) * exp(type(2.5)*x[i]);

    correlation = exponential_correlation(thread_pool_device, x, y);

    EXPECT_NEAR(correlation.r, type(1), type(1.0e-3));
    EXPECT_NEAR(correlation.b, type(2.5), NUMERIC_LIMITS_MIN);
}


void CorrelationsTest::test_power_correlation()
{
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

    EXPECT_NEAR(correlation.r > type(0.999999));
    EXPECT_NEAR(correlation.a, type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(correlation.b, type(2), NUMERIC_LIMITS_MIN);
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
