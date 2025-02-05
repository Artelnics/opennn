#include "pch.h"

#include "../opennn/statistics.h"
#include "../opennn/histogram.h"
#include "../opennn/tensors.h"
#include "../opennn/strings_utilities.h"

using namespace opennn;

TEST(StatisticsTest, CountEmptyBins)
{   
    
    Histogram histogram;

    EXPECT_EQ(histogram.count_empty_bins(), 0);

    // Test

    Tensor<type, 1> centers;
    Tensor<Index, 1> frecuencies;

    centers.resize(3);
    centers.setValues({ type(1),type(2),type(3) });

    frecuencies.resize(3);
    frecuencies.setValues({ 1,1,0 });

    Histogram histogram_1(centers, frecuencies);
    EXPECT_EQ(histogram_1.count_empty_bins(), 1);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers, frecuencies);
    EXPECT_EQ(histogram_2.count_empty_bins(), 3);

    // Test

    centers.resize(3);
    centers.setValues({ type(1),type(2),type(3) });

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_3(centers, frecuencies);
    EXPECT_EQ(histogram_3.count_empty_bins(), 3);
    
}

TEST(StatisticsTest, CalculateMinimumFrequency)
{
    
    Histogram histogram;
    Index minimum = histogram.calculate_minimum_frequency();
    string str_minimum = to_string(minimum);
    EXPECT_EQ(is_numeric_string(str_minimum), true);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,1,0});

    Histogram histogram_1(centers,frecuencies);
    EXPECT_EQ(histogram_1.calculate_minimum_frequency(), 0);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    EXPECT_EQ(histogram_2.calculate_minimum_frequency(), 0);

    // Test

    centers.resize(3);
    centers.setValues({type(1),type(2),type(3)});

    frecuencies.resize(3);
    frecuencies.setValues({5,4,10});

    Histogram histogram_3(centers,frecuencies);
    EXPECT_EQ(histogram_3.calculate_minimum_frequency(), 4);
    
}


TEST(StatisticsTest, CalculateMaximumFrequency)
{
    
    Histogram histogram;
    Index maximum = histogram.calculate_maximum_frequency();
    string str_maximum = to_string(maximum);
    EXPECT_EQ(is_numeric_string(str_maximum), true);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,0,1});

    Histogram histogram_1(centers,frecuencies);
    EXPECT_EQ(histogram_1.calculate_maximum_frequency(), 1);

    // Test

    centers.resize(3);
    centers.setValues({ type(1),type(3),type(5)});

    frecuencies.resize(3);
    frecuencies.setValues({5,21,8});

    Histogram histogram_2(centers,frecuencies);
    EXPECT_EQ(histogram_2.calculate_maximum_frequency(), 21);
    
}

TEST(StatisticsTest, CalculateMostPopulatedBin)
{
    
    Histogram histogram;
    EXPECT_EQ(histogram.calculate_most_populated_bin(), 0);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,1});

    Histogram histogram_1(centers,frecuencies);
    //EXPECT_EQ(histogram_1.calculate_most_populated_bin(),2);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    //EXPECT_EQ(histogram_2.calculate_most_populated_bin(), 0);

    // Test

    centers.resize(3);
    centers.setValues({ type(1),type(5),type(9)});

    frecuencies.resize(3);
    frecuencies.setValues({5,4,10});

    Histogram histogram_3(centers,frecuencies);
//    EXPECT_EQ(histogram_3.calculate_most_populated_bin(), 2);

}


TEST(StatisticsTest, CalculateMinimalCenters)
{
    /*
    Histogram histogram;

    // Test

    Tensor<type, 1> vector(14);
    vector.setValues(
                {type(1), type(1), type(12), type(1), type(1), type(1), type(2),
                 type(2), type(6), type(4), type(8), type(1), type(4), type(7)});
    
    histogram = opennn::histogram(vector);

    Tensor<type, 1> solution(4);
    solution.setValues({type(6), type(7), type(8), type(12)});

    EXPECT_EQ((Index(histogram.calculate_minimal_centers()[0] - solution[0])) < 1.0e-7, true);
    EXPECT_EQ((Index(histogram.calculate_minimal_centers()[1] - solution[1])) < 1.0e-7, true);
    EXPECT_EQ((Index(histogram.calculate_minimal_centers()[2] - solution[2])) < 1.0e-7, true);
    EXPECT_EQ((Index(histogram.calculate_minimal_centers()[3] - solution[3])) < 1.0e-7, true);

    // Test

    Histogram histogram_0;
    EXPECT_EQ(isnan(histogram_0.calculate_minimal_centers()(0)), true);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,1});

    Histogram histogram_1(centers,frecuencies);

    EXPECT_EQ(Index(histogram_1.calculate_minimal_centers()(0)), 1);
    EXPECT_EQ(Index(histogram_1.calculate_minimal_centers()(1)), 2);
    */
}


TEST(StatisticsTest, CalculateMaximalCenters)
{
    
    Histogram histogram;

    // Test

    Tensor<type, 1> vector(18);
    vector.setValues(
                {type(1), type(1), type(1),
                 type(1), type(7), type(2),
                 type(2), type(6), type(7),
                 type(4), type(8), type(8),
                 type(8), type(1), type(4),
                 type(7), type(7), type(7)});

    histogram = opennn::histogram(vector);

    Tensor<type, 1> solution(2);
    solution.setValues({ type(1), type(7)});

    //EXPECT_NEAR(histogram.calculate_maximal_centers()(4), solution[1], 1.0e-7);
    
    // Test

    Histogram histogram_0;
    EXPECT_EQ(isnan(histogram_0.calculate_maximal_centers()(0)),true);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,1,0});

    Histogram histogram_1(centers,frecuencies);

    EXPECT_EQ(Index(histogram_1.calculate_maximal_centers()(0)), 1);
    EXPECT_EQ(Index(histogram_1.calculate_maximal_centers()(1)), 2);
    
}


TEST(StatisticsTest, CalculateBin)
{
 /*
    Histogram histogram;
    EXPECT_EQ(histogram.calculate_bin(type(0)), 0);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({ type(2),type(4),type(6)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,0});

    Histogram histogram_1(centers,frecuencies);
    EXPECT_EQ(histogram_1.calculate_bin(type(6)) == 2,true);
    
    // Test

    Tensor<type, 1> vector(3);
    Index bin;

    vector.setValues({ type(1), type(1), type(11.0)});
    histogram = opennn::histogram(vector, 10);
    
    bin = histogram.calculate_bin(vector[0]);
    EXPECT_EQ(bin, 0);
    bin = histogram.calculate_bin(vector[1]);
    EXPECT_EQ(bin, 0);
    //bin = histogram.calculate_bin(vector[2]);
    //EXPECT_EQ(bin, 1);
*/
}


TEST(StatisticsTest, CalculateFrequency)
{

    Histogram histogram;
    EXPECT_EQ(histogram.calculate_frequency(type(0)), 0);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,1,2});

    Histogram histogram_1(centers,frecuencies);
    EXPECT_EQ(histogram_1.calculate_frequency(type(2)) == 1, true);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    EXPECT_EQ(histogram_2.calculate_frequency(type(0)), 0);

    // Test

    Tensor<type, 1> vector(3);
    Index frequency_3;
    Histogram histogram_3;

    vector.setValues({type(0), type(1), type(9) });
    //histogram_3 = opennn::histogram(vector, 10);
    //frequency_3 = histogram_3.calculate_frequency(vector[9]);

    //EXPECT_EQ(frequency_3,1);

}


TEST(StatisticsTest, Minimum)
{
    Tensor<type, 1> vector;

    // Test

    EXPECT_EQ(isnan(type(minimum(vector))),true);

    // Test

    vector.resize(3);
    vector.setValues({type(0), type(1), type(9)});

    EXPECT_NEAR(minimum(vector), type(0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(3);
    vector.setValues({type(1),type(2),type(3)});

    EXPECT_NEAR(minimum(vector), type(1), NUMERIC_LIMITS_MIN);

    vector.resize(3);
    vector.setValues({ type(-1),type(2),type(3)});

    EXPECT_NEAR(minimum(vector), type(-1), NUMERIC_LIMITS_MIN);
}


TEST(StatisticsTest, Maximum)
{
    Tensor<type, 1> vector;

    // Test

    EXPECT_EQ(isnan(maximum(vector)), true);

    // Test

    vector.resize(3);
    vector.setValues({ type(0), type(1), type(9)});

    EXPECT_NEAR(maximum(vector), type(9), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(3);
    vector.setValues({type(1),type(2),type(3)});

    EXPECT_NEAR(maximum(vector), type(3), NUMERIC_LIMITS_MIN);

    vector.resize(3);
    vector.setValues({ type(-1),type(-2),type(-3)});

    EXPECT_NEAR(maximum(vector), type(-1), NUMERIC_LIMITS_MIN);
}


TEST(StatisticsTest, Mean)
{
    Tensor<type, 2> matrix(3,3);
    matrix.setZero();

    EXPECT_NEAR(mean(matrix)(0), 0, NUMERIC_LIMITS_MIN);

    matrix.setValues({{type(0),type(1),type(-2)},
                      {type(0),type(1),type(8)},
                      {type(0),type(1),type(6)}});

    EXPECT_NEAR(mean(matrix)(0), 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(mean(matrix)(1), type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(mean(matrix)(2), type(4), NUMERIC_LIMITS_MIN);

    Tensor<type, 1> vector(2);
    vector.setValues({ type(1), type(1)});

    EXPECT_NEAR(mean(vector), type(1), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(2);
    vector.setValues({ type(-1), type(1) });

    EXPECT_NEAR(mean(vector), type(0), NUMERIC_LIMITS_MIN);

    // Test missing values

    vector.resize(5);
    vector.setValues({ type(1), type(NAN), type(2.0), type(3.0), type(4.0)});

    EXPECT_NEAR(mean(vector), type(2.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({ type(1), type(1), type(NAN), type(1) });

    EXPECT_NEAR(mean(vector), type(1), NUMERIC_LIMITS_MIN);

    // Test empty matrix

    matrix.resize(0, 0);

    EXPECT_EQ(isnan(mean(matrix,2)), true);
}


TEST(StatisticsTest, StandardDeviation)
{
    Tensor<type, 1> vector(1);
    vector.setZero();

    type standard_deviation;

    // Test

    EXPECT_NEAR(opennn::standard_deviation(vector), type(0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({ type(2),type(4),type(8),type(10)});

    EXPECT_NEAR(opennn::standard_deviation(vector), sqrt(type(40)/type(3)), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setConstant(type(-11));

    EXPECT_NEAR(opennn::standard_deviation(vector), type(0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(3);
    vector.setZero();

    EXPECT_NEAR(opennn::standard_deviation(vector), 0, NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(2);
    vector.setValues({ type(1), type(1)});

    standard_deviation = opennn::standard_deviation(vector);

    EXPECT_NEAR(standard_deviation, 0, NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(2);
    vector.setValues({ type(-1.0), type(1) });

    standard_deviation = opennn::standard_deviation(vector);

    EXPECT_NEAR(standard_deviation, sqrt(type(2)), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(1);
    vector[0] = type(NAN);

    standard_deviation = opennn::standard_deviation(vector);

    EXPECT_NEAR(standard_deviation, 0, NUMERIC_LIMITS_MIN);
}


TEST(StatisticsTest, Median)
{/*
    Tensor<type, 1> vector;
    vector.setZero();
    Tensor<type, 2> matrix;

    type median;

    // Test

    vector.resize(2);

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({type(2),type(4),type(8),type(10)});

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(6), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({type(-11),type(-11),type(-11),type(-11)});

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(-11), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({ type(1),type(2),type(3),type(4)});

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(2.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(5);
    vector.setValues({ type(1),type(2),type(3),type(4),type(5)});

    median = opennn::median(vector);

    EXPECT_NEAR(abs(median), type(3), NUMERIC_LIMITS_MIN);

    // Test

    matrix.resize(3,2);
    matrix.setValues({{type(1),type(1)},
                      {type(2),type(3)},
                      {type(3),type(4)}});

    EXPECT_NEAR(abs(opennn::median(matrix, 0)), type(2), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(opennn::median(matrix, 1)), type(3), NUMERIC_LIMITS_MIN);

    // Test

    matrix.resize(3,2);
    matrix.setValues({{type(1),type(NAN)},
                      {type(NAN),type(NAN)},
                      {type(3),type(3.5)}});

    EXPECT_NEAR(abs(opennn::median(matrix, 0)), type(2), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(opennn::median(matrix, 1)), type(3.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({type(3),type(NAN),type(1),type(NAN)});

    median = opennn::median(vector);

    EXPECT_NEAR(abs(median), type(2), NUMERIC_LIMITS_MIN);
    */
}


TEST(StatisticsTest, Variance)
{
    Tensor<type, 1> vector;

    // Test

    vector.resize(3);
    vector.setZero();

    EXPECT_EQ(Index(variance(vector)), 0);

    // Test , 2

    vector.resize(4);
    vector.setValues({ type(2),type(4),type(8),type(10)});

    EXPECT_NEAR(variance(vector), type(40)/type(3), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({ type(-11),type(-11),type(-11),type(-11)});

    EXPECT_NEAR(variance(vector), type(0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(1);
    vector.setConstant(type(1));

    EXPECT_NEAR(abs(variance(vector)), type(0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(3);
    vector.setValues({type(2),type(1),type(2)});

    EXPECT_NEAR(abs(variance(vector)), type(1)/type(3), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(3);
    vector.setValues({type(1),type(NAN),type(2)});

    EXPECT_NEAR(abs(variance(vector)), type(0.5), NUMERIC_LIMITS_MIN);
}


TEST(StatisticsTest, Quartiles)
{
    Tensor<type, 1> vector;
    Tensor<type, 1> quartiles;
    
    /*
    // Test
    
    vector.resize(1);
    vector.setZero();

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(Index(quartiles(0)),type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(Index(quartiles(1)), type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(Index(quartiles(2)), type(0), NUMERIC_LIMITS_MIN);
        
    // Test

    vector.resize(2);
    vector.setValues({type(0), type(1)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)) , type(0.25), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(0.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(3);
    vector.setValues({ type(0),type(1),type(2)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(0.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(1.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(4);
    vector.setValues({ type(0),type(1),type(2),type(3)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(0.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(2.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(5);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(0.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(2), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(3.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(6);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(2.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(4), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(7);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5),type(6)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(3.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(5.0), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(8);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(3.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(5.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(9);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7),type(8)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(4.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(6.5), NUMERIC_LIMITS_MIN);

   // Test

    vector.resize(9);
    vector.setValues({ type(1),type(4),type(6),type(2),type(0),type(3),type(4),type(7),type(10)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(4.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(6.5), NUMERIC_LIMITS_MIN);

    // Test

    vector.resize(20);
    vector.setValues({type(12),type(14),type(50),type(76),type(12),type(34),type(56),type(74),type(89),type(60),type(96),type(24),type(53),type(25),type(67),type(84),type(92),type(45),type(62),type(86)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(29.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(58.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(80.0), NUMERIC_LIMITS_MIN);
    
  
    // Test missing values:

    // Test

    vector.resize(5);
    vector.setValues({type(1), type(2), type(3), type(NAN), type(4)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(2.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(3.5), NUMERIC_LIMITS_MIN);
    
    // Test

    vector.resize(6);
    vector.setValues({type(1), type(2), type(3), type(NAN), type(4), type(5)});

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(1)), type(3.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(quartiles(2)), type(4.5), NUMERIC_LIMITS_MIN);
    */
}


TEST(StatisticsTest, Histogram)
{
    /*
    Tensor<type, 1> vector;

    Tensor<type, 1> centers;
    Tensor<Index, 1> frequencies;

    // Test

    vector.resize(11);
    vector.setValues({type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7),type(8),type(9),type(10)});
    Histogram histogram(vector, 10);
    //EXPECT_EQ(histogram.get_bins_number() == 10);

    centers = histogram.centers;
    frequencies = histogram.frequencies;

    //EXPECT_EQ(abs(centers[0] - type(0.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[1] - type(1.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[2] - type(2.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[3] - type(3.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[4] - type(4.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[5] - type(5.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[6] - type(6.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[7] - type(7.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[8] - type(8.5)) < type(1.0e-3));
    //EXPECT_EQ(abs(centers[9] - type(9.5)) < type(1.0e-3));

    //EXPECT_EQ(frequencies[0] == 1);
    //EXPECT_EQ(frequencies[1] == 1);
    //EXPECT_EQ(frequencies[2] == 1);
    //EXPECT_EQ(frequencies[3] == 1);
    //EXPECT_EQ(frequencies[4] == 1);
    //EXPECT_EQ(frequencies[5] == 1);
    //EXPECT_EQ(frequencies[6] == 1);
    //EXPECT_EQ(frequencies[7] == 1);
    //EXPECT_EQ(frequencies[8] == 1);
    //EXPECT_EQ(frequencies[9] == 2);

    Tensor<Index, 0> sum_frec_1 = frequencies.sum();

    //EXPECT_EQ(sum_frec_1(0) == 11);

    // Test

    vector.resize(20);
    vector.setRandom();

    Histogram histogram_2(vector, 10);

    centers = histogram_2.centers;
    frequencies = histogram_2.frequencies;

    Tensor<Index, 0> sum_frec_2;
    sum_frec_2 = frequencies.sum();

    //EXPECT_EQ(sum_frec_2(0) == 20);
*/
}


TEST(StatisticsTest, Histograms)
{
    /*
    Tensor<Histogram, 1> histograms;
    Tensor<type, 2> matrix(3,3);
    matrix.setValues({
                         {type(1),type(1),type(1)},
                         {type(2),type(2),type(2)},
                         {type(3),type(3),type(3)}
                     });

    histograms = opennn::histograms(matrix, 3);

    //EXPECT_EQ(histograms(0).frequencies(0), 1);
    //EXPECT_EQ(histograms(1).frequencies(0), 1);
    //EXPECT_EQ(histograms(2).frequencies(0), 1);
    */
}


TEST(StatisticsTest, TotalFrequencies)
{
    Tensor<Histogram, 1> histograms(3);
    /*
    // Test

    Tensor<type, 1> vector1_1(16);
    vector1_1.setValues({type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(0),type(1),type(1),type(1),type(2),type(2),type(2),type(2),type(2)});

    Tensor<type, 1> vector2_1(16);
    vector2_1.setValues({type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(0),type(1),type(1),type(1),type(2),type(2),type(2),type(2),type(2)});

    Tensor<type, 1> vector3_1(16);
    vector3_1.setValues({type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(0),type(1),type(1),type(1),type(2),type(2),type(2),type(2),type(2)});

    histograms(0) = histogram(vector1_1, 7);
    histograms(1) = histogram(vector2_1, 7);
    histograms(2) = histogram(vector3_1, 7);

    Tensor<Index, 1> total_frequencies = opennn::total_frequencies(histograms);

    //EXPECT_EQ(total_frequencies(0), 2);
    //EXPECT_EQ(total_frequencies(1), 4);
    //EXPECT_EQ(total_frequencies(2), 6);

    // Test

    Tensor<type, 2> matrix(3,3);
    matrix.setValues({
                         {type(1),type(1),type(NAN)},
                         {type(2),type(2),type(1)},
                         {type(3),type(3),type(2)},
                     });

    histograms = opennn::histograms(matrix, 3);

    //EXPECT_EQ(histograms(0).frequencies(0), 1 );
    //EXPECT_EQ(histograms(1).frequencies(0), 1);
    //EXPECT_EQ(histograms(2).frequencies(0), 1);
    */
}


TEST(StatisticsTest, MinimalIndex)
{
    Tensor<type, 1> vector;

    // Test

    EXPECT_EQ(minimal_index(vector), 0);

    // Test

    vector.resize(3);
    vector.setValues({ type(1),type(0),type(-1)});

    EXPECT_EQ(minimal_index(vector), 2);
}


TEST(StatisticsTest, MaximalIndex)
{
    // Test

    Tensor<type, 1> vector(0);

    EXPECT_EQ(maximal_index(vector), 0);

    // Test

    vector.resize(3);
    vector.setValues({ type(1),type(0),type(-1)});

    EXPECT_EQ(maximal_index(vector), 0);
}


TEST(StatisticsTest, MinimalIndices)
{
    /*
    Tensor<type, 1> vector;

    // Test

    EXPECT_EQ(minimal_indices(vector,0).dimension(0), 0);

    // Test

    vector.resize(3);
    vector.setValues({ type(-1),type(0),type(1)});

    EXPECT_EQ(minimal_indices(vector, 1)[0], 0);

    EXPECT_EQ(minimal_indices(vector, 3)[0], 0);
    EXPECT_EQ(minimal_indices(vector, 3)[1], 1);
    EXPECT_EQ(minimal_indices(vector, 3)[2], 2);

    // Test

    vector.resize(4);
    vector.setValues({ type(0),type(0),type(0),type(1)});

    EXPECT_EQ(minimal_indices(vector, 4)[0], 0);
    EXPECT_EQ(minimal_indices(vector, 4)[1], 1);
    EXPECT_EQ(minimal_indices(vector, 4)[3], 3);

    // Test

    vector.resize(5);
    vector.setValues({type(0),type(1),type(0),type(2),type(0)});

    EXPECT_EQ(minimal_indices(vector, 5)[0] == 0 || minimal_indices(vector, 5)[0] == 2 || minimal_indices(vector, 5)[0] == 4,true);
    EXPECT_EQ(minimal_indices(vector, 5)[1] == 0 || minimal_indices(vector, 5)[1] == 2 || minimal_indices(vector, 5)[1] == 4,true);
    EXPECT_EQ(minimal_indices(vector, 5)[2] == 0 || minimal_indices(vector, 5)[2] == 2 || minimal_indices(vector, 5)[2] == 4,true);
    EXPECT_EQ(minimal_indices(vector, 5)[3], 1);
    EXPECT_EQ(minimal_indices(vector, 5)[4], 3);

    // Test

    vector.resize(4);
    vector.setValues({type(-1),type(2),type(-3),type(4)});

    EXPECT_EQ(minimal_indices(vector, 2)[0], 2);
    EXPECT_EQ(minimal_indices(vector, 2)[1], 0);
    */
}


TEST(StatisticsTest, MaximalIndices)
{
    /*
    Tensor<type, 1> vector;

    // Test

    EXPECT_EQ(maximal_indices(vector,0).dimension(0), 0);

    // Test

    vector.resize(3);
    vector.setValues({ type(-1),type(0),type(1) });

    EXPECT_EQ(maximal_indices(vector, 1)[0], 2);

    // Test

    vector.resize(4);
    vector.setValues({ type(1),type(1),type(1),type(1) });

    EXPECT_EQ(maximal_indices(vector, 4)[0], 0);
    EXPECT_EQ(maximal_indices(vector, 4)[1], 1);
    EXPECT_EQ(maximal_indices(vector, 4)[3], 3);

    // Test

    vector.resize(5);
    vector.setValues({ type(1),type(5),type(6),type(7),type(2) });

    EXPECT_EQ(maximal_indices(vector, 5)[0], 3);
    EXPECT_EQ(maximal_indices(vector, 5)[1], 2);
    EXPECT_EQ(maximal_indices(vector, 5)[3], 4);
    */
}


TEST(StatisticsTest, BoxPlot)
{
    /*
    const Index size = get_random_index(1, 10);

    Tensor<type, 1> vector(size);

    BoxPlot box_plot;
    BoxPlot solution;
    
    // Test

    vector.resize(4);
    vector.setZero();

    box_plot = opennn::box_plot(vector);

    EXPECT_NEAR(box_plot.minimum, type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.first_quartile, type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.median, type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.third_quartile, type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.maximum, type(0), NUMERIC_LIMITS_MIN);
    
    // Test

    vector.resize(8);
    vector.setValues({ type(2.0), type(2.0), type(3.0), type(5.0), type(6.0), type(7.0), type(8.0), type(9.0) });

    box_plot = opennn::box_plot(vector);

    solution.set(type(2.0), type(2.5), type(5.5), type(7.5), type(9.0));

    EXPECT_NEAR(box_plot.minimum, solution.minimum, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.first_quartile, solution.first_quartile, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.median, solution.median, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.third_quartile, solution.third_quartile, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.maximum, solution.maximum, NUMERIC_LIMITS_MIN);
    
    // Test missing values

    vector.resize(9);
    vector.setValues({ type(2.0), type(2.0), type(3.0), type(5.0), type(6.0), type(7.0), type(NAN), type(8.0), type(9.0)});

    box_plot = opennn::box_plot(vector);

    solution.set(type(2.0), type(2.5), type(5.5), type(7.5), type(9.0));

    EXPECT_NEAR(box_plot.minimum, solution.minimum, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.first_quartile, solution.first_quartile, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.median, solution.median, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.third_quartile, solution.third_quartile, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(box_plot.maximum, solution.maximum, NUMERIC_LIMITS_MIN);
    */
}


TEST(StatisticsTest, Percentiles)
{
    /*
    Tensor<type, 1> vector;

    // Test

    Tensor<type, 1> empty_vector(10);
    empty_vector.setConstant(NAN);
    Tensor<type, 1> percentiles_empty = opennn::percentiles(empty_vector);

    EXPECT_EQ(isnan(percentiles_empty(0)),true);

    // Test

    vector.resize(10);
    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9) });

    
    Tensor<type, 1> percentiles = opennn::percentiles(vector);
    
    Tensor<type, 1> solution(10);
    solution.setValues({ type(0.5), type(1.5), type(2.5), type(3.5), type(4.5), type(5.5), type(6.5), type(7.5), type(8.5), type(9) });
    
    EXPECT_EQ(abs(percentiles(0)), solution(0), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(1)), solution(1), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(2)), solution(2), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(3)), solution(3), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(4)), solution(4), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(5)), solution(5), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(6)), solution(6), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(7)), solution(7), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(8)), solution(8), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(9)), solution(9), type(1.0e-7));
    
    
    // Test

    vector.resize(21);
    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10), type(11), type(12), type(13), type(14), type(15), type(16), type(17), type(18), type(19), type(20) });

    percentiles = opennn::percentiles(vector);

    solution.resize(10);
    solution.setValues({ type(2), type(4), type(6), type(8), type(10), type(12), type(14), type(16), type(18), type(20) });
    
    EXPECT_EQ(abs(percentiles(0)), solution(0), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(1)), solution(1), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(2)), solution(2), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(3)), solution(3), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(4)), solution(4), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(5)), solution(5), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(6)), solution(6), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(7)), solution(7), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(8)), solution(8), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(9)), solution(9), type(1.0e-7));
    
    
    // Test

    vector.resize(14);

    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(11), type(15), type(19), type(32) });

    percentiles = opennn::percentiles(vector);

    solution.resize(10);
    solution.setValues({ type(1), type(2), type(4), type(5), type(6.5), type(8), type(9), type(15), type(19), type(32) });
    
    EXPECT_EQ(abs(percentiles(0)), solution(0), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(1)), solution(1), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(2)), solution(2), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(3)), solution(3), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(4)), solution(4), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(5)), solution(5), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(6)), solution(6), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(7)), solution(7), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(8)), solution(8), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(9)), solution(9), type(1.0e-7));
    
    // Test
    vector.resize(21);
    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10), type(11), type(12), type(13), type(14), type(15), type(16), type(17), type(18), type(19), type(20) });

    vector(20) = type(NAN);

    percentiles = opennn::percentiles(vector);

    solution.resize(10);
    solution.setValues({ type(1.5), type(3.5), type(5.5), type(7.5), type(9.5), type(11.5), type(13.5), type(15.5), type(17.5), type(19) });

    EXPECT_EQ(abs(percentiles(0)), solution(0), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(1)), solution(1), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(2)), solution(2), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(3)), solution(3), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(4)), solution(4), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(5)), solution(5), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(6)), solution(6), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(7)), solution(7), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(8)), solution(8), type(1.0e-7));
    EXPECT_EQ(abs(percentiles(9)), solution(9), type(1.0e-7));
    */
}


