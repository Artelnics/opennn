#include "pch.h"

#include "opennn/statistics.h"
#include "opennn/string_utilities.h"
#include "opennn/random_utilities.h"
#include "opennn/io_utilities.h"

using namespace opennn;

TEST(StatisticsTest, Minimum)
{
    VectorR vector;

    // Test

    EXPECT_EQ(isnan(type(minimum(vector))),true);

    // Test

    vector.resize(3);
    vector << type(0), type(1), type(9);

    EXPECT_NEAR(minimum(vector), type(0), EPSILON);

    // Test

    vector.resize(3);
    vector << type(1),type(2),type(3);

    EXPECT_NEAR(minimum(vector), type(1), EPSILON);

    vector.resize(3);
    vector <<  type(-1),type(2),type(3);

    EXPECT_NEAR(minimum(vector), type(-1), EPSILON);
}


TEST(StatisticsTest, Maximum)
{
    VectorR vector;

    // Test

    EXPECT_EQ(isnan(maximum(vector)), true);

    // Test

    vector.resize(3);
    vector <<  type(0), type(1), type(9);

    EXPECT_NEAR(maximum(vector), type(9), EPSILON);

    // Test

    vector.resize(3);
    vector << type(1),type(2),type(3);

    EXPECT_NEAR(maximum(vector), type(3), EPSILON);

    vector.resize(3);
    vector <<  type(-1),type(-2),type(-3);

    EXPECT_NEAR(maximum(vector), type(-1), EPSILON);
}


TEST(StatisticsTest, Mean)
{
    MatrixR matrix(3,3);
    matrix.setZero();

    EXPECT_NEAR(mean(matrix)(0), 0, EPSILON);

    matrix << type(0),type(1),type(-2),
              type(0),type(1),type(8),
              type(0),type(1),type(6);

    EXPECT_NEAR(mean(matrix)(0), 0, EPSILON);
    EXPECT_NEAR(mean(matrix)(1), type(1), EPSILON);
    EXPECT_NEAR(mean(matrix)(2), type(4), EPSILON);

    VectorR vector(2);
    vector <<  type(1), type(1);

    EXPECT_NEAR(mean(vector), type(1), EPSILON);

    // Test

    vector.resize(2);
    vector <<  type(-1), type(1) ;

    EXPECT_NEAR(mean(vector), type(0), EPSILON);

    // Test missing values

    vector.resize(5);
    vector <<  type(1), type(NAN), type(2.0), type(3.0), type(4.0);

    EXPECT_NEAR(mean(vector), type(2.5), EPSILON);

    // Test

    vector.resize(4);
    vector <<  type(1), type(1), type(NAN), type(1) ;

    EXPECT_NEAR(mean(vector), type(1), EPSILON);

    // Test empty matrix

    matrix.resize(0, 0);

    EXPECT_EQ(isnan(mean(matrix,2)), true);
}


TEST(StatisticsTest, StandardDeviation)
{
    VectorR vector(1);
    vector.setZero();

    type standard_deviation;

    // Test

    EXPECT_NEAR(opennn::standard_deviation(vector), type(0), EPSILON);

    // Test

    vector.resize(4);
    vector <<  type(2),type(4),type(8),type(10);

    EXPECT_NEAR(opennn::standard_deviation(vector), sqrt(type(40)/type(3)), EPSILON);

    // Test

    vector.resize(4);
    vector.setConstant(type(-11));

    EXPECT_NEAR(opennn::standard_deviation(vector), type(0), EPSILON);

    // Test

    vector.resize(3);
    vector.setZero();

    EXPECT_NEAR(opennn::standard_deviation(vector), 0, EPSILON);

    // Test

    vector.resize(2);
    vector <<  type(1), type(1);

    standard_deviation = opennn::standard_deviation(vector);

    EXPECT_NEAR(standard_deviation, 0, EPSILON);

    // Test

    vector.resize(2);
    vector <<  type(-1.0), type(1) ;

    standard_deviation = opennn::standard_deviation(vector);

    EXPECT_NEAR(standard_deviation, sqrt(type(2)), EPSILON);

    // Test

    vector.resize(1);
    vector[0] = type(NAN);

    standard_deviation = opennn::standard_deviation(vector);

    EXPECT_NEAR(standard_deviation, 0, EPSILON);
}


TEST(StatisticsTest, Median)
{
    
    VectorR vector(2);
    vector.setZero();
    
    type median;
    
    // Test

    vector.resize(2);

    median = opennn::median(vector);
    
    EXPECT_NEAR(median, type(0), EPSILON);

    // Test

    vector.resize(4);
    vector << type(2),type(4),type(8),type(10);

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(6), EPSILON);

    // Test

    vector.resize(4);
    vector << type(-11),type(-11),type(-11),type(-11);

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(-11), EPSILON);

    // Test

    vector.resize(4);
    vector <<  type(1),type(2),type(3),type(4);

    median = opennn::median(vector);

    EXPECT_NEAR(median, type(2.5), EPSILON);

    // Test

    vector.resize(5);
    vector <<  type(1),type(2),type(3),type(4),type(5);

    median = opennn::median(vector);

    EXPECT_NEAR(abs(median), type(3), EPSILON);

    // Test

    vector.resize(4);
    vector << type(3),type(NAN),type(1),type(NAN);

    median = opennn::median(vector);

    EXPECT_NEAR(abs(median), type(2), EPSILON);
    
    // Test
    MatrixR matrix(3,2);

    matrix << type(1),type(1),
              type(2),type(3),
              type(3),type(4);

    EXPECT_NEAR(abs(opennn::median(matrix, 0)), type(2), EPSILON);
    EXPECT_NEAR(abs(opennn::median(matrix, 1)), type(3), EPSILON);
    
    // Test

    matrix.resize(3,2);
    matrix << type(1),type(NAN),
              type(NAN),type(NAN),
              type(3),type(3.5);

    EXPECT_NEAR(abs(opennn::median(matrix, 0)), type(2), EPSILON);
    EXPECT_NEAR(abs(opennn::median(matrix, 1)), type(3.5), EPSILON);

}


TEST(StatisticsTest, Variance)
{
    VectorR vector;

    // Test

    vector.resize(3);
    vector.setZero();

    EXPECT_EQ(Index(variance(vector)), 0);

    // Test , 2

    vector.resize(4);
    vector <<  type(2),type(4),type(8),type(10);

    EXPECT_NEAR(variance(vector), type(40)/type(3), EPSILON);

    // Test

    vector.resize(4);
    vector <<  type(-11),type(-11),type(-11),type(-11);

    EXPECT_NEAR(variance(vector), type(0), EPSILON);

    // Test

    vector.resize(1);
    vector.setConstant(type(1));

    EXPECT_NEAR(abs(variance(vector)), type(0), EPSILON);

    // Test

    vector.resize(3);
    vector << type(2),type(1),type(2);

    EXPECT_NEAR(abs(variance(vector)), type(1)/type(3), EPSILON);

    // Test

    vector.resize(3);
    vector << type(1),type(NAN),type(2);

    EXPECT_NEAR(abs(variance(vector)), type(0.5), EPSILON);
}


TEST(StatisticsTest, Quartiles)
{
    VectorR vector;
    VectorR quartiles;
    
    
    // Test
    
    vector.resize(1);
    vector.setZero();

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(Index(quartiles(0)),type(0), EPSILON);
    EXPECT_NEAR(Index(quartiles(1)), type(0), EPSILON);
    EXPECT_NEAR(Index(quartiles(2)), type(0), EPSILON);
        
    // Test

    vector.resize(2);
    vector << type(0), type(1);


    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)) , type(0.25), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(0.5), EPSILON);

    // Test

    vector.resize(3);
    vector <<  type(0),type(1),type(2);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(0.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(1), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(1.5), EPSILON);

    // Test

    vector.resize(4);
    vector <<  type(0),type(1),type(2),type(3);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(0.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(1.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(2.5), EPSILON);    
    
    // Test

    vector.resize(5);
    vector <<  type(0),type(1),type(2),type(3),type(4);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(0.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(2), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(3.5), EPSILON);

    // Test

    vector.resize(6);
    vector <<  type(0),type(1),type(2),type(3),type(4),type(5);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(2.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(4), EPSILON);

    
    // Test

    vector.resize(7);
    vector <<  type(0),type(1),type(2),type(3),type(4),type(5),type(6);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(3.0), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(5.0), EPSILON);

    // Test

    vector.resize(8);
    vector <<  type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(3.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(5.5), EPSILON);

    // Test

    vector.resize(9);
    vector <<  type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7),type(8);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(4.0), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(6.5), EPSILON);

   // Test

    vector.resize(9);
    vector <<  type(1),type(4),type(6),type(2),type(0),type(3),type(4),type(7),type(10);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(4.0), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(6.5), EPSILON);

    // Test

    vector.resize(20);
    vector << type(12),type(14),type(50),type(76),type(12),type(34),type(56),type(74),type(89),type(60),type(96),type(24),type(53),type(25),type(67),type(84),type(92),type(45),type(62),type(86);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(29.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(58.0), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(80.0), EPSILON);
    
    
    // Test missing values:

    // Test

    vector.resize(5);
    vector << type(1), type(2), type(3), type(NAN), type(4);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(2.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(3.5), EPSILON);
    
    // Test

    vector.resize(6);
    vector << type(1), type(2), type(3), type(NAN), type(4), type(5);

    quartiles = opennn::quartiles(vector);

    EXPECT_NEAR(abs(quartiles(0)), type(1.5), EPSILON);
    EXPECT_NEAR(abs(quartiles(1)), type(3.0), EPSILON);
    EXPECT_NEAR(abs(quartiles(2)), type(4.5), EPSILON);
    
}


TEST(StatisticsTest, Histogram)
{
    VectorR vector;

    VectorR centers;
    VectorR frequencies;
    
    // Test

    vector.resize(11);
    vector << type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7),type(8),type(9),type(10);
    Histogram histogram(vector, 10);
    EXPECT_EQ(histogram.get_bins_number(), 10);

    centers = histogram.centers;
    frequencies = histogram.frequencies;

    EXPECT_NEAR(abs(centers[0]), type(0.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[1]), type(1.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[2]), type(2.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[3]), type(3.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[4]), type(4.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[5]), type(5.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[6]), type(6.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[7]), type(7.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[8]), type(8.5), type(1.0e-3));
    EXPECT_NEAR(abs(centers[9]), type(9.5), type(1.0e-3));

    EXPECT_EQ(frequencies[0], 1);
    EXPECT_EQ(frequencies[1], 1);
    EXPECT_EQ(frequencies[2], 1);
    EXPECT_EQ(frequencies[3], 1);
    EXPECT_EQ(frequencies[4], 1);
    EXPECT_EQ(frequencies[5], 1);
    EXPECT_EQ(frequencies[6], 1);
    EXPECT_EQ(frequencies[7], 1);
    EXPECT_EQ(frequencies[8], 1);
    EXPECT_EQ(frequencies[9], 2);

    type sum_frec_1 = frequencies.sum();

    EXPECT_EQ(sum_frec_1, 11);

    // Test

    vector.resize(20);
    vector.setRandom();

    Histogram histogram_2(vector, 10);

    centers = histogram_2.centers;
    frequencies = histogram_2.frequencies;

    type sum_frec_2 = frequencies.sum();

    EXPECT_EQ(sum_frec_2, 20);
}


TEST(StatisticsTest, Histograms)
{
    MatrixR matrix(3, 3);
    matrix << type(1), type(1), type(1),
              type(2), type(2), type(2),
              type(3), type(3), type(3);

    const vector<Histogram> result = histograms(matrix, 3);

    ASSERT_EQ(ssize(result), 3);

    for (const Histogram& column_histogram : result)
    {
        ASSERT_EQ(column_histogram.frequencies.size(), 3);
        EXPECT_EQ(column_histogram.frequencies.sum(), 3);
        EXPECT_EQ(column_histogram.frequencies(0), 1);
        EXPECT_EQ(column_histogram.frequencies(1), 1);
        EXPECT_EQ(column_histogram.frequencies(2), 1);
    }
}


TEST(StatisticsTest, MinimalIndex)
{
    VectorR vector;

    // Test

    EXPECT_EQ(minimal_index(vector), 0);

    // Test

    vector.resize(3);
    vector <<  type(1),type(0),type(-1);

    EXPECT_EQ(minimal_index(vector), 2);
}


TEST(StatisticsTest, MaximalIndex)
{
    // Test

    VectorR vector(0);

    EXPECT_EQ(maximal_index(vector), 0);

    // Test

    vector.resize(3);
    vector <<  type(1),type(0),type(-1);

    EXPECT_EQ(maximal_index(vector), 0);
}


TEST(StatisticsTest, MinimalIndices)
{
    
    VectorR vector;

    // Test
   
    EXPECT_EQ(minimal_indices(vector, 0).rows(), 0);

    // Test
    
    vector.resize(3);
    vector <<  type(-1),type(0),type(1);

    EXPECT_EQ(minimal_indices(vector, 1)(0), 0);

    EXPECT_EQ(minimal_indices(vector, 3)(0), 0);
    EXPECT_EQ(minimal_indices(vector, 3)(1), 1);
    EXPECT_EQ(minimal_indices(vector, 3)(2), 2);

    // Test

    vector.resize(4);
    vector <<  type(0),type(0),type(0),type(1);

    EXPECT_EQ(minimal_indices(vector, 4)(0), 0);
    EXPECT_EQ(minimal_indices(vector, 4)(1), 1);
    EXPECT_EQ(minimal_indices(vector, 4)(3), 3);

    // Test
  
    vector.resize(5);
    vector << type(0),type(1),type(0),type(2),type(0);

    EXPECT_EQ(minimal_indices(vector, 5)(0) == 0 || minimal_indices(vector, 5)(0) == 2 || minimal_indices(vector, 5)(0) == 4, true);
    EXPECT_EQ(minimal_indices(vector, 5)(1) == 0 || minimal_indices(vector, 5)(1) == 2 || minimal_indices(vector, 5)(1) == 4, true);
    EXPECT_EQ(minimal_indices(vector, 5)(2) == 0 || minimal_indices(vector, 5)(2) == 2 || minimal_indices(vector, 5)(2) == 4, true);
    EXPECT_EQ(minimal_indices(vector, 5)(3), 1);
    EXPECT_EQ(minimal_indices(vector, 5)(4), 3);
    

    // Test

    vector.resize(4);
    vector << type(-1),type(2),type(-3),type(4);

    EXPECT_EQ(minimal_indices(vector, 2)(0), 2);
    EXPECT_EQ(minimal_indices(vector, 2)(1), 0);
    
}


TEST(StatisticsTest, MaximalIndices)
{
    
    VectorR vector;

    // Test

    EXPECT_EQ(maximal_indices(vector,0).rows(), 0);

    // Test

    vector.resize(3);
    vector <<  type(-1),type(0),type(1) ;

    EXPECT_EQ(maximal_indices(vector, 1)[0], 2);

    // Test

    vector.resize(4);
    vector <<  type(1),type(1),type(1),type(1) ;

    EXPECT_EQ(maximal_indices(vector, 4)[0], 0);
    EXPECT_EQ(maximal_indices(vector, 4)[1], 1);
    EXPECT_EQ(maximal_indices(vector, 4)[3], 3);

    // Test

    vector.resize(5);
    vector <<  type(1),type(5),type(6),type(7),type(2) ;

    EXPECT_EQ(maximal_indices(vector, 5)[0], 3);
    EXPECT_EQ(maximal_indices(vector, 5)[1], 2);
    EXPECT_EQ(maximal_indices(vector, 5)[3], 4);

}


TEST(StatisticsTest, BoxPlot)
{
    
    const Index size = random_integer(1, 10);

    VectorR vector(size);

    BoxPlot box_plot;
    BoxPlot solution;
    
    // Test

    vector.resize(4);
    vector.setZero();
    
    box_plot = opennn::box_plot(vector);

    EXPECT_NEAR(box_plot.minimum, type(0), EPSILON);
    EXPECT_NEAR(box_plot.first_quartile, type(0), EPSILON);
    EXPECT_NEAR(box_plot.median, type(0), EPSILON);
    EXPECT_NEAR(box_plot.third_quartile, type(0), EPSILON);
    EXPECT_NEAR(box_plot.maximum, type(0), EPSILON);
    
    // Test

    vector.resize(8);
    vector <<  type(2.0), type(2.0), type(3.0), type(5.0), type(6.0), type(7.0), type(8.0), type(9.0) ;

    box_plot = opennn::box_plot(vector);

    solution.set(type(2.0), type(2.5), type(5.5), type(7.5), type(9.0));

    EXPECT_NEAR(box_plot.minimum, solution.minimum, EPSILON);

    EXPECT_NEAR(box_plot.first_quartile, solution.first_quartile, EPSILON);
    EXPECT_NEAR(box_plot.median, solution.median, EPSILON);
    EXPECT_NEAR(box_plot.third_quartile, solution.third_quartile, EPSILON);
    EXPECT_NEAR(box_plot.maximum, solution.maximum, EPSILON);

    // Test missing values

    vector.resize(9);
    vector <<  type(2.0), type(2.0), type(3.0), type(5.0), type(6.0), type(7.0), type(NAN), type(8.0), type(9.0);

    box_plot = opennn::box_plot(vector);

    solution.set(type(2.0), type(2.5), type(5.5), type(7.5), type(9.0));

    EXPECT_NEAR(box_plot.minimum, solution.minimum, EPSILON);

    EXPECT_NEAR(box_plot.first_quartile, solution.first_quartile, EPSILON);
    EXPECT_NEAR(box_plot.median, solution.median, EPSILON);
    EXPECT_NEAR(box_plot.third_quartile, solution.third_quartile, EPSILON);
    EXPECT_NEAR(box_plot.maximum, solution.maximum, EPSILON);
    
}




