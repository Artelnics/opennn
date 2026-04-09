
#include "pch.h"


#include "../opennn/correlations.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/statistics.h"
#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/string_utilities.h"

using namespace opennn;

class CorrelationsTest : public ::testing::Test 
{
protected:
    
    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> device;

    void SetUp() override {
        const unsigned int threads_number = thread::hardware_concurrency();
        thread_pool = make_unique<ThreadPool>(threads_number);
        device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);
    }

    void TearDown() override {
    }
};

TEST_F(CorrelationsTest, SpearmanCorrelation)
{

    VectorR x(10);
    x << type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10);

    VectorR y(10);
    y << type(1), type(4), type(9), type(16), type(25), type(36), type(49), type(64), type(81), type(100);
    
    Correlation result = linear_correlation_spearman(x, y);

    EXPECT_NEAR(result.r, type(1), EPSILON);

}


TEST_F(CorrelationsTest, LinearCorrelation)
{
    VectorR x(10);
    x << type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10);

    VectorR y(10);
    y << type(10), type(20), type(30),type(40),type(50),type(60),type(70),type(80),type(90),type(100);

    EXPECT_NEAR(linear_correlation(x, y).r, type(1), EPSILON);
    
    y << type(10), type(9), type(8),type(7),type(6),type(5),type(4),type(3),type(2),type(1);

    EXPECT_NEAR(linear_correlation(x, y).r, type(- 1), EPSILON);
    
    // Test

    x.setRandom();
    y.setRandom();

    EXPECT_NE(linear_correlation(x, y).r, type(-1));
    EXPECT_NE(linear_correlation(x, y).r, type( 0));
    EXPECT_NE(linear_correlation(x, y).r, type( 1));
}


TEST_F(CorrelationsTest, LogisticCorrelation)
{
    VectorR x(20);
    x << -10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9;

    VectorR y(20);
    y << 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1;

    Correlation correlation = logistic_correlation(x, y);

    correlation.print();
    //correlation.print(); system("pause");

    EXPECT_LE(abs(correlation.r), type(0.1));
    EXPECT_EQ(correlation.form, Correlation::Form::Sigmoid);

    // Test

    Index size = 10;

    x.resize(size);
    x << -5,-4,-3,-2,-1,1,2,3,4,5;

    y.resize(size);
    y << 0,0,0,0,0,1,1,1,1,1;

    correlation = logistic_correlation(x, y);

    //correlation.print();

    EXPECT_GE(correlation.r, type(0.9));
    EXPECT_LE(correlation.r, type(1));
    EXPECT_EQ(correlation.form, Correlation::Form::Sigmoid);

    EXPECT_NEAR(correlation.r, type(1), EPSILON);

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

    correlation = logistic_correlation(x, y);

    //correlation.print();

    EXPECT_LE(correlation.r, type(1));

    for (Index i = 0; i < size; i++)
        y[i] = exp(type(2.5) * x[i] + type(1.4));

    // Test

    for (Index i = 0; i < size / 2; i++) y[i] = 1.0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation(x, y);

    //correlation.print();

    EXPECT_LE(abs(correlation.r), type(1));

    // Test

    y.setConstant(type(0));

    correlation = logistic_correlation(x, y);

    EXPECT_EQ(isnan(correlation.r), true);

    y.setConstant(type(0));

    for (Index i = size - (size / 2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation(x, y);

    //EXPECT_NEAR(correlation.r, type(1), EPSILON);
    //EXPECT_EQ(correlation.form, Correlation::Form::Sigmoid);

    // Test

    size = 100;

    x.resize(size);

    for (Index i = 0; i < size; i++)
    {
        x[i] = i;
    }

    y.resize(size);

    for (Index i = 0; i < size / 2; i++) y[i] = 0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 1;

    correlation = logistic_correlation(x, y);

    EXPECT_LE(correlation.r, type(1));

    for (Index i = 0; i < size; i++)
        y[i] = exp(type(2.5) * x[i] + type(1.4));

    //const unsigned int threads_number = thread::hardware_concurrency();

    //ThreadPool* thread_pool = new ThreadPool(threads_number);

    //ThreadPoolDevice* device = new ThreadPoolDevice(thread_pool, threads_number);

    // Test

    for (Index i = 0; i < size / 2; i++) y[i] = 1.0;

    for (Index i = size - (size / 2); i < size; i++) y[i] = 0.0;

    correlation = logistic_correlation(x, y);

    EXPECT_GE(abs(correlation.r), type(-0.95));

    // Test

    y.setConstant(type(0));

    correlation = logistic_correlation(x, y);

    EXPECT_EQ(isnan(correlation.r), true);

}

TEST_F(CorrelationsTest, LogarithmicCorrelation)
{
    VectorR x;
    VectorR y;
    Index size;
    Correlation correlation;
    type solution;

    //Perfect case

    size = 10;

    x.resize(size);
    y.resize(size);

    x << type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10);
    for (Index i = 0; i < size; i++)
    {
        y[i] = type(4) * log(x[i]);
    }

    correlation = logarithmic_correlation(x, y);

    solution = type(1);

    EXPECT_NEAR(correlation.r, solution, EPSILON);
    EXPECT_NEAR(correlation.b, type(4), EPSILON);
    EXPECT_NEAR(correlation.a, type(0), EPSILON);
}

TEST_F(CorrelationsTest, ExponentialCorrelation)
{
    VectorR x;
    VectorR y;
    Index size;
    Correlation correlation;

    //Test

    size = 10;

    x.resize(size);
    y.resize(size);

    for (Index i = 0; i < size; i++)
    {
        x[i] = i;
        y[i] = type(1) * exp(type(0.5) * x[i]);
    }

    correlation = exponential_correlation(x, y);

    EXPECT_NEAR(correlation.r, type(1), EPSILON);
    EXPECT_NEAR(correlation.a, type(1), EPSILON);
    EXPECT_NEAR(correlation.b, type(0.5), EPSILON);

    // Test missing values

    size = 5;

    x.resize(size);
    y.resize(size);

    x << type(1),type(2),type(3),type(4),type(NAN);
    for(Index i = 0; i < size; i++)
    {
        y[i] = type(1.4) * exp(type(2.5) * x[i]);
    }

    correlation = exponential_correlation(x, y);

    EXPECT_NEAR(correlation.r, type(1), type(1.0e-3));
    EXPECT_NEAR(correlation.b, type(2.5), EPSILON);
    EXPECT_NEAR(correlation.a, type(1.4), EPSILON);

}

TEST_F(CorrelationsTest, PowerCorrelation)
{

    VectorR x;
    VectorR y;
    Index size;
    Correlation correlation;

    // Test

    size = 10;

    x.resize(size);
    y.resize(size);

    for(Index i = 0; i < size; i++)
    {
        x[i] = type(i + 1);
        y[i] = type(1) * pow(x[i], type(2));
    }

    correlation = power_correlation(x, y);

    // Test

    EXPECT_NEAR(correlation.r, type(1), EPSILON);
    EXPECT_NEAR(correlation.a, type(1), EPSILON);
    EXPECT_NEAR(correlation.b, type(2), EPSILON);

}

TEST_F(CorrelationsTest, Autocorrelation)
{

    Index size = 1000;
    VectorR x(size);
    VectorR y;
    VectorR correlation;

    for(Index i = 0; i < size; i++)
    {
        x[i] = i;
    }

    correlation = autocorrelations(x, size/100);

    EXPECT_GT(minimum(correlation), type(0.9));

}

TEST_F(CorrelationsTest, CrossCorrelation)
{

    Index size = 1000;
    VectorR x(size);
    VectorR y(size);
    VectorR cross_correlation;

    for (Index i = 0; i < size; i++)
    {
        x[i] = i;
        y[i] = i;
    }

    cross_correlation = cross_correlations(x, y, 10);

    EXPECT_LT(cross_correlation(0), type(5));
    EXPECT_GT(cross_correlation(1), type(0.9));

}
