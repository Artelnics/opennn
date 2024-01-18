//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   T E S T   C L A S S                             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "statistics_test.h"

StatisticsTest::StatisticsTest() : UnitTesting()
{   
}


StatisticsTest::~StatisticsTest()
{
}


void StatisticsTest::test_count_empty_bins()
{
    cout << "test_count_empty_bins\n";

    Tensor<type, 1> centers;
    Tensor<Index, 1> frecuencies;

    // Test

    Histogram histogram;
    assert_true(histogram.count_empty_bins() == 0, LOG);

    // Test

    centers.resize(3);
    centers.setValues({type(1),type(2),type(3)});

    frecuencies.resize(3);
    frecuencies.setValues({1,1,0});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.count_empty_bins() == 1, LOG);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.count_empty_bins() == 3, LOG);

    // Test

    centers.resize(3);
    centers.setValues({type(1),type(2),type(3)});

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_3(centers,frecuencies);
    assert_true(histogram_3.count_empty_bins() == 3, LOG);
}


void StatisticsTest::test_calculate_minimum_frequency()
{
    cout << "test_calculate_minimun_frecuency\n";

    // Test

    Histogram histogram;
    assert_true(is_not_numeric(histogram.calculate_minimum_frequency()) , LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,1,0});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.calculate_minimum_frequency() == 0, LOG);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.calculate_minimum_frequency() == 0, LOG);

    // Test

    centers.resize(3);
    centers.setValues({type(1),type(2),type(3)});

    frecuencies.resize(3);
    frecuencies.setValues({5,4,10});

    Histogram histogram_3(centers,frecuencies);
    assert_true(histogram_3.calculate_minimum_frequency() == 4, LOG);
}


void StatisticsTest::test_calculate_maximum_frequency()
{
    cout << "test_calculate_maximum_frequency\n";

    // Test

    Histogram histogram;
    assert_true(is_not_numeric(histogram.calculate_maximum_frequency()), LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,0,1});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.calculate_maximum_frequency() == 1, LOG);

    // Test

    centers.resize(3);
    centers.setValues({ type(1),type(3),type(5)});

    frecuencies.resize(3);
    frecuencies.setValues({5,21,8});

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.calculate_maximum_frequency() == 21, LOG);
}


void StatisticsTest::test_calculate_most_populated_bin()
{
    cout << "test_calculate_most_populated_bin\n";

    // Test

    Histogram histogram;
    assert_true(histogram.calculate_most_populated_bin() == 0, LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,1});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.calculate_most_populated_bin() == 2, LOG);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.calculate_most_populated_bin() == 0, LOG);

    // Test

    centers.resize(3);
    centers.setValues({ type(1),type(5),type(9)});

    frecuencies.resize(3);
    frecuencies.setValues({5,4,10});

    Histogram histogram_3(centers,frecuencies);
    assert_true(histogram_3.calculate_most_populated_bin() == 2, LOG);
}


void StatisticsTest::test_calculate_minimal_centers()
{
    cout << "test_calculate_minimal_centers\n";

    Histogram histogram;

    // Test

    Tensor<type, 1> vector(14);
    vector.setValues(
                {type(1), type(1), type(12), type(1), type(1), type(1), type(2),
                 type(2), type(6), type(4), type(8), type(1), type(4), type(7)});

    histogram = opennn::histogram(vector);

    Tensor<type, 1> solution(4);
    solution.setValues({type(6), type(7), type(8), type(12)});

    assert_true((Index(histogram.calculate_minimal_centers()[0] - solution[0])) < 1.0e-7, LOG);
    assert_true((Index(histogram.calculate_minimal_centers()[1] - solution[1])) < 1.0e-7, LOG);
    assert_true((Index(histogram.calculate_minimal_centers()[2] - solution[2])) < 1.0e-7, LOG);
    assert_true((Index(histogram.calculate_minimal_centers()[3] - solution[3])) < 1.0e-7, LOG);

    // Test

    Histogram histogram_0;
    assert_true(isnan(histogram_0.calculate_minimal_centers()(0)), LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,1});

    Histogram histogram_1(centers,frecuencies);

    assert_true(Index(histogram_1.calculate_minimal_centers()(0)) == 1, LOG);
    assert_true(Index(histogram_1.calculate_minimal_centers()(1)) == 2, LOG);
}


void StatisticsTest::test_calculate_maximal_centers()
{
    cout << "test_calculate_maximal_centers\n";

    Histogram histogram;

    // Test

    Tensor<type, 1> vector(18);
    vector.setValues(
                {type(1), type(1), type(1),
                 type(1), type(7), type(2),
                 type(2), type(6), type(7),
                 type(4), type(8), type(8),
                 type(8), type(1), type(4),
                 type(7), type(7), type(7) });

    histogram = opennn::histogram(vector);

    Tensor<type, 1> solution(2);
    solution.setValues({ type(1), type(7)});

    assert_true(Index(histogram.calculate_maximal_centers()[0] - solution[0]) < 1.0e-7, LOG);
    assert_true(Index(histogram.calculate_maximal_centers()[1] - solution[1]) < 1.0e-7, LOG);

    // Test

    Histogram histogram_0;
    assert_true(isnan(histogram_0.calculate_maximal_centers()(0)), LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,1,0});

    Histogram histogram_1(centers,frecuencies);

    assert_true(Index(histogram_1.calculate_maximal_centers()(0)) == 1, LOG);
    assert_true(Index(histogram_1.calculate_maximal_centers()(1)) == 2, LOG);
}


void StatisticsTest::test_calculate_bin()
{
    cout << "test_calculate_bin\n";

    // Test

    Histogram histogram;
    assert_true(histogram.calculate_bin(type(0)) == 0, LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({ type(2),type(4),type(6)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,0});

    Histogram histogram_1(centers,frecuencies);

    assert_true(histogram_1.calculate_bin(type(6)) == 2, LOG);

    // Test

    Tensor<type, 1> vector(3);
    Index bin;

    vector.setValues({ type(1), type(1), type(11.0)});
    histogram = opennn::histogram(vector, 10);

    bin = histogram.calculate_bin(vector[0]);
    assert_true(bin == 0, LOG);

    bin = histogram.calculate_bin(vector[1]);
    assert_true(bin == 0, LOG);

    bin = histogram.calculate_bin(vector[2]);
    assert_true(bin == 1, LOG);
}


void StatisticsTest::test_calculate_frequency()
{
    cout << "test_calculate_frequency\n";

    // Test

    Histogram histogram;
    assert_true(histogram.calculate_frequency(type(0)) == 0, LOG);

    // Test

    Tensor<type, 1> centers(3);
    centers.setValues({type(1),type(2),type(3)});

    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,1,2});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.calculate_frequency(type(2)) == 1, LOG);

    // Test

    centers.resize(3);
    centers.setZero();

    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.calculate_frequency(type(0)) == 0, LOG);

    // Test

    Tensor<type, 1> vector(3);
    Index frequency_3;
    Histogram histogram_3;

    vector.setValues({type(0), type(1), type(9) });
    histogram_3 = opennn::histogram(vector, 10);
    frequency_3 = histogram_3.calculate_frequency(vector[9]);

    assert_true(frequency_3 == 1, LOG);
}


void StatisticsTest::test_minimum()
{
    cout << "test_calculate_minimum\n";

    Tensor<type, 1> vector;

    // Test

    assert_true(isnan(type(minimum(vector))), LOG);

    // Test

    vector.resize(3);
    vector.setValues({type(0), type(1), type(9)});

    assert_true(minimum(vector) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(3);
    vector.setValues({type(1),type(2),type(3)});

    vector.resize(3);
    vector.setValues({ type(-1),type(2),type(3)});

    assert_true(minimum(vector) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(minimum(vector) - type(-1) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_maximum()
{
    cout << "test_calculate_maximum\n";

    Tensor<type, 1> vector;

    // Test

    assert_true(isnan(maximum(vector)), LOG);

    // Test

    vector.resize(3);
    vector.setValues({ type(0), type(1), type(9)});

    assert_true(maximum(vector) - type(9) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(3);
    vector.setValues({type(1),type(2),type(3)});

    vector.resize(3);
    vector.setValues({ type(-1),type(-2),type(-3)});

    assert_true(maximum(vector) - type(3) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(maximum(vector) - type(-1) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_mean()
{
    cout << "test_mean\n";

    Tensor<type, 1> vector;
    Tensor<type, 2> matrix;

    // Test

    matrix.resize(3,3);
    matrix.setZero();
    assert_true(mean(matrix)(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    matrix.resize(3,3);
    matrix.setValues({
                         {type(0),type(1),type(-2)},
                         {type(0),type(1),type(8)},
                         {type(0),type(1),type(6)}});

    assert_true(mean(matrix)(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(mean(matrix)(1) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(mean(matrix)(2) - type(4) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(2);
    vector.setValues({ type(1), type(1)});

    assert_true(mean(vector) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(2);
    vector[0] = type(-1);
    vector[1] = type(1);
    assert_true(mean(vector) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test missing values

    vector.resize(5);

    vector.setValues({ type(1), type(NAN), type(2.0), type(3.0), type(4.0)});

    assert_true(abs(mean(vector)) - type(2.5) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector[0] = type(1);
    vector[1] = type(1);
    vector[2] = type(NAN);
    vector[3] = type(1);

    assert_true(mean(vector) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test empty matrix

    matrix.resize(0, 0);

    assert_true(isnan(mean(matrix,2)), LOG);
}


void StatisticsTest::test_standard_deviation()
{
    cout << "test_standard_deviation\n";

    Tensor<type, 1> vector(1);
    vector.setZero();

    type standard_deviation;

    // Test

    assert_true(opennn::standard_deviation(vector) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(2),type(4),type(8),type(10)});

    assert_true(opennn::standard_deviation(vector) - sqrt(type(40)/type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setConstant(type(-11));

    assert_true(opennn::standard_deviation(vector) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(3);
    vector.setZero();

    assert_true(Index(opennn::standard_deviation(vector)) < NUMERIC_LIMITS_MIN, LOG);

    // Test

    vector.resize(2);
    vector.setValues({ type(1), type(1)});

    standard_deviation = opennn::standard_deviation(vector);

    assert_true(abs(Index(standard_deviation)) < NUMERIC_LIMITS_MIN, LOG);

    // Test

    vector.resize(2);
    vector[0] = type(-1.0);
    vector[1] = type(1);

    standard_deviation = opennn::standard_deviation(vector);

    assert_true(abs(standard_deviation- sqrt(type(2))) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(1);
    vector[0] = type(NAN);

    standard_deviation = opennn::standard_deviation(vector);

    assert_true(standard_deviation < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_median()
{
    cout << "test_calculate_median\n";

    Tensor<type, 1> vector;
    Tensor<type, 2> matrix;

    // Test

    vector.resize(2);
    vector.setZero();

    assert_true(median(vector) == 0, LOG);

    // Test

    vector.resize(4);
    vector.setValues({type(2),type(4),type(8),type(10)});

    assert_true(median(vector) - type(6) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setValues({type(-11),type(-11),type(-11),type(-11)});

    assert_true(median(vector) - type(-11) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(1),type(2),type(3),type(4)});

    assert_true(abs(median(vector) - type(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(5);
    vector.setValues({ type(1),type(2),type(3),type(4),type(5)});

    assert_true(abs(median(vector) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    matrix.resize(3,2);
    matrix.setValues({
                         {type(1),type(1)},
                         {type(2),type(3)},
                         {type(3),type(4)}
                     });

    assert_true(abs(median(matrix)(0) - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(median(matrix)(1) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    matrix.resize(3,2);
    matrix.setValues({
                         {type(1),type(1)},
                         {type(NAN),type(NAN)},
                         {type(3),type(3)}
                     });

    assert_true(isnan(median(matrix)(0)), LOG);
    assert_true(isnan(median(matrix)(1)), LOG);

    // Test

    vector.resize(4);
    vector.setValues({type(3),type(NAN),type(1),type(NAN)});

    assert_true(median(vector) - type(2) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_variance()
{
    cout << "test_variance\n";

    Tensor<type, 1> vector;

    // Test

    vector.resize(3);
    vector.setZero();

    assert_true(Index(variance(vector)) == 0, LOG);

    // Test , 2

    vector.resize(4);
    vector.setValues({ type(2),type(4),type(8),type(10)});

    assert_true(variance(vector) - type(40)/type(3) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(-11),type(-11),type(-11),type(-11)});

    assert_true(variance(vector) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(1);
    vector.resize(1);
    vector.setConstant(type(1));

    assert_true(abs(variance(vector) - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(3);
    vector.setValues({type(2),type(1),type(2)});

    assert_true(abs(variance(vector) - type(1)/type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(3);
    vector.setValues({type(1),type(NAN),type(2)});

    assert_true(abs(variance(vector) - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_asymmetry()
{
    cout << "test_calculate_asymmetry\n";

    Tensor<type, 1> vector;

    // Test

    vector.resize(3);
    vector.setZero();

    cout << asymmetry(vector) << endl;
    assert_true(asymmetry(vector) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test
    vector.resize(4);
    vector.setValues({type(1), type(5), type(3), type(9)});

    type asymmetry_value = opennn::asymmetry(vector);

    assert_true(asymmetry_value - type(0.75) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(1),type(5),type(3),type(9)});

    assert_true(asymmetry(vector) - type(0.75) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_kurtosis()
{
    cout << "test_calculate_kurtosis\n";

    Tensor<type, 1> vector;

    // Test

    vector.resize(4);
    vector.setValues({ type(1),type(5),type(3),type(9)});

    assert_true(abs(kurtosis(vector) - type(-1.9617)) < type(1e-3), LOG);

    // Test

    vector.resize(5);
    vector.setValues({type(1), type(5), type(NAN), type(3), type(9)});

    type kurtosis = opennn::kurtosis(vector);
}


void StatisticsTest::test_quartiles()
{
    cout << "test_quartiles\n";

    Tensor<type, 1> vector;
    Tensor<type, 1> quartiles;

    // Test

    vector.resize(1);
    vector.setZero();

    quartiles = opennn::quartiles(vector);

    assert_true(Index(quartiles(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(Index(quartiles(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(Index(quartiles(2)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(2);
    vector.setValues({type(0), type(1)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(0.25)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(3);
    vector.setValues({ type(0),type(1),type(2)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(0),type(1),type(2),type(3)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(5);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(2.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(3.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(6);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(4.0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(7);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5),type(6)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(3.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(5.0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(8);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(3.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(5.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(9);
    vector.setValues({ type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7),type(8)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(4.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(6.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(9);
    vector.setValues({ type(1),type(4),type(6),type(2),type(0),type(3),type(4),type(7),type(10)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(4.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(6.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(20);
    vector.setValues({type(12),type(14),type(50),type(76),type(12),type(34),type(56),type(74),type(89),type(60),type(96),type(24),type(53),type(25),type(67),type(84),type(92),type(45),type(62),type(86)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(29.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(58.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(80.0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test missing values:

    // Test

    vector.resize(5);
    vector.setValues({type(1), type(2), type(3), type(NAN), type(4)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(3.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(6);
    vector.setValues({type(1), type(2), type(3), type(NAN), type(4), type(5)});

    quartiles = opennn::quartiles(vector);

    assert_true(abs(quartiles(0) - type(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(1) - type(3.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(quartiles(2) - type(4.5)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_histogram()
{
    cout << "test_histogram\n";

    Tensor<type, 1> vector;

    Tensor<type, 1> centers;
    Tensor<Index, 1> frequencies;

    // Test

    vector.resize(11);
    vector.setValues({type(0),type(1),type(2),type(3),type(4),type(5),type(6),type(7),type(8),type(9),type(10)});

    Histogram histogram(vector, 10);
    assert_true(histogram.get_bins_number() == 10, LOG);

    centers = histogram.centers;
    frequencies = histogram.frequencies;

    assert_true(abs(centers[0] - type(0.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[1] - type(1.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[2] - type(2.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[3] - type(3.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[4] - type(4.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[5] - type(5.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[6] - type(6.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[7] - type(7.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[8] - type(8.5)) < type(1.0e-3), LOG);
    assert_true(abs(centers[9] - type(9.5)) < type(1.0e-3), LOG);

    assert_true(frequencies[0] == 1, LOG);
    assert_true(frequencies[1] == 1, LOG);
    assert_true(frequencies[2] == 1, LOG);
    assert_true(frequencies[3] == 1, LOG);
    assert_true(frequencies[4] == 1, LOG);
    assert_true(frequencies[5] == 1, LOG);
    assert_true(frequencies[6] == 1, LOG);
    assert_true(frequencies[7] == 1, LOG);
    assert_true(frequencies[8] == 1, LOG);
    assert_true(frequencies[9] == 2, LOG);

    Tensor<Index, 0> sum_frec_1 = frequencies.sum();

    assert_true(sum_frec_1(0) == 11, LOG);

    // Test

    vector.resize(20);
    vector.setRandom();

    Histogram histogram_2(vector, 10);

    centers = histogram_2.centers;
    frequencies = histogram_2.frequencies;

    Tensor<Index, 0> sum_frec_2;
    sum_frec_2 = frequencies.sum();

    assert_true(sum_frec_2(0) == 20, LOG);
}


void StatisticsTest::test_histograms()
{
    cout << "test_histograms\n";

    Tensor<Histogram, 1> histograms;

    Tensor<type, 2> matrix(3,3);
    matrix.setValues({
                         {type(1),type(1),type(1)},
                         {type(2),type(2),type(2)},
                         {type(3),type(3),type(3)}
                     });

    histograms = opennn::histograms(matrix, 3);

    assert_true(histograms(0).frequencies(0) == 1, LOG);
    assert_true(histograms(1).frequencies(0) == 1, LOG);
    assert_true(histograms(2).frequencies(0) == 1, LOG);
}


void StatisticsTest::test_total_frequencies()   //<--- Check
{
    cout << "test_total_frequencies\n";

    Tensor<Histogram, 1> histograms(3);

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

    assert_true(total_frequencies(0) == 2, LOG);
    assert_true(total_frequencies(1) == 4, LOG);
    assert_true(total_frequencies(2) == 6, LOG);

    // Test

    Tensor<type, 2> matrix(3,3);
    matrix.setValues({
                         {type(1),type(1),type(NAN)},
                         {type(2),type(2),type(1)},
                         {type(3),type(3),type(2)},
                     });

    histograms = opennn::histograms(matrix, 3);

    assert_true(histograms(0).frequencies(0) == 1 , LOG);
    assert_true(histograms(1).frequencies(0) == 1, LOG);
    assert_true(histograms(2).frequencies(0) == 0, LOG);
}


void StatisticsTest::test_calculate_minimal_index()
{
    cout << "test_calculate_minimal_index\n";

    Tensor<type, 1> vector;

    // Test

    assert_true(minimal_index(vector) == 0, LOG);

    // Test

    vector.resize(3);
    vector.setValues({ type(1),type(0),type(-1)});

    assert_true(minimal_index(vector) == 2, LOG);
}


void StatisticsTest::test_calculate_maximal_index()
{
    cout << "test_calculate_maximal_index\n";

    // Test

    Tensor<type, 1> vector(0);

    assert_true(maximal_index(vector) == 0, LOG);

    // Test

    vector.resize(3);
    vector.setValues({ type(1),type(0),type(-1)});

    assert_true(maximal_index(vector) == 0, LOG);
}


void StatisticsTest::test_calculate_minimal_indices()
{
    cout << "test_calculate_minimal_indices\n";

    Tensor<type, 1> vector;

    // Test

    assert_true(minimal_indices(vector,0).dimension(0) == 0, LOG);

    // Test

    vector.resize(3);
    vector.setValues({ type(-1),type(0),type(1)});

    assert_true(minimal_indices(vector, 1)[0] == 0, LOG);

    assert_true(minimal_indices(vector, 3)[0] == 0, LOG);
    assert_true(minimal_indices(vector, 3)[1] == 1, LOG);
    assert_true(minimal_indices(vector, 3)[2] == 2, LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(0),type(0),type(0),type(1)});

    assert_true(minimal_indices(vector, 4)[0] == 0, LOG);
    assert_true(minimal_indices(vector, 4)[1] == 1, LOG);
    assert_true(minimal_indices(vector, 4)[3] == 3, LOG);

    // Test

    vector.resize(5);
    vector.setValues({type(0),type(1),type(0),type(2),type(0)});

    assert_true(minimal_indices(vector, 5)[0] == 0 || minimal_indices(vector, 5)[0] == 2 || minimal_indices(vector, 5)[0] == 4, LOG);
    assert_true(minimal_indices(vector, 5)[1] == 0 || minimal_indices(vector, 5)[1] == 2 || minimal_indices(vector, 5)[1] == 4, LOG);
    assert_true(minimal_indices(vector, 5)[2] == 0 || minimal_indices(vector, 5)[2] == 2 || minimal_indices(vector, 5)[2] == 4, LOG);
    assert_true(minimal_indices(vector, 5)[3] == 1, LOG);
    assert_true(minimal_indices(vector, 5)[4] == 3, LOG);

    // Test

    vector.resize(4);
    vector.setValues({type(-1),type(2),type(-3),type(4)});

    assert_true(minimal_indices(vector, 2)[0] == 2, LOG);
    assert_true(minimal_indices(vector, 2)[1] == 0, LOG);
}


void StatisticsTest::test_calculate_maximal_indices()
{
    cout << "test_calculate_maximal_indices\n";

    Tensor<type, 1> vector;

    // Test

    assert_true(maximal_indices(vector,0).dimension(0) == 0, LOG);

    // Test

    vector.resize(3);
    vector.setValues({ type(-1),type(0),type(1) });

    assert_true(maximal_indices(vector, 1)[0] == 2, LOG);

    // Test

    vector.resize(4);
    vector.setValues({ type(1),type(1),type(1),type(1) });

    assert_true(maximal_indices(vector, 4)[0] == 0, LOG);
    assert_true(maximal_indices(vector, 4)[1] == 1, LOG);
    assert_true(maximal_indices(vector, 4)[3] == 3, LOG);

    // Test

    vector.resize(5);
    vector.setValues({ type(1),type(5),type(6),type(7),type(2) });

    assert_true(maximal_indices(vector, 5)[0] == 3, LOG);
    assert_true(maximal_indices(vector, 5)[1] == 2, LOG);
    assert_true(maximal_indices(vector, 5)[3] == 4, LOG);
}


void StatisticsTest::test_box_plot()
{
    cout << "test_box_plot\n";

    Tensor<type, 1> vector;

    BoxPlot box_plot;
    BoxPlot solution;

    // Test

    vector.resize(4);
    vector.setZero();

    box_plot = opennn::box_plot(vector);

    assert_true(box_plot.minimum - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.first_quartile - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.median - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.third_quartile - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.maximum - type(0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    vector.resize(8);
    vector.setValues({ type(2.0), type(2.0), type(3.0), type(5.0), type(6.0), type(7.0), type(8.0), type(9.0) });

    box_plot = opennn::box_plot(vector);

    solution.set(type(2.0), type(2.5), type(5.5), type(7.5), type(9.0));

    assert_true(box_plot.minimum - solution.minimum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.first_quartile - solution.first_quartile < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.median - solution.median < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.third_quartile - solution.third_quartile < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(box_plot.maximum - solution.maximum < type(NUMERIC_LIMITS_MIN), LOG);

    // Test missing values

    vector.resize(9);
    vector.setValues({ type(2.0), type(2.0), type(3.0), type(5.0), type(6.0), type(7.0), type(NAN), type(8.0), type(9.0)});

    box_plot = opennn::box_plot(vector);

    solution.set(type(2.0), type(2.5), type(5.5), type(7.5), type(9.0));

    assert_true((box_plot.minimum - solution.minimum) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((box_plot.first_quartile - solution.first_quartile) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((box_plot.median - solution.median) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((box_plot.third_quartile - solution.third_quartile) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((box_plot.maximum - solution.maximum) < type(NUMERIC_LIMITS_MIN), LOG);
}


void StatisticsTest::test_percentiles()
{
    cout << "test_percentiles\n";

    Tensor<type, 1> vector;

    // Test

    Tensor<type, 1> empty_vector(10);
    empty_vector.setConstant(NAN);
    Tensor<type, 1> percentiles_empty = opennn::percentiles(empty_vector);

    assert_true(isnan(percentiles_empty(0)), LOG);

    // Test

    vector.resize(10);
    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9) });

    Tensor<type, 1> percentiles = opennn::percentiles(vector);

    Tensor<type, 1> solution(10);
    solution.setValues({ type(0.5), type(1.5), type(2.5), type(3.5), type(4.5), type(5.5), type(6.5), type(7.5), type(8.5), type(9) });

    assert_true(abs(percentiles(0) - solution(0)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(1) - solution(1)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(2) - solution(2)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(3) - solution(3)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(4) - solution(4)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(5) - solution(5)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(6) - solution(6)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(7) - solution(7)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(8) - solution(8)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(9) - solution(9)) < type(1.0e-7), LOG);

    // Test

    vector.resize(21);
    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10), type(11), type(12), type(13), type(14), type(15), type(16), type(17), type(18), type(19), type(20) });

    percentiles = opennn::percentiles(vector);

    solution.resize(10);
    solution.setValues({ type(2), type(4), type(6), type(8), type(10), type(12), type(14), type(16), type(18), type(20) });

    assert_true(abs(percentiles(0) - solution(0)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(1) - solution(1)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(2) - solution(2)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(3) - solution(3)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(4) - solution(4)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(5) - solution(5)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(6) - solution(6)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(7) - solution(7)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(8) - solution(8)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(9) - solution(9)) < type(1.0e-7), LOG);

    // Test

    vector.resize(14);

    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(11), type(15), type(19), type(32) });

    percentiles = opennn::percentiles(vector);

    solution.resize(10);
    solution.setValues({ type(1), type(2), type(4), type(5), type(6.5), type(8), type(9), type(15), type(19), type(32) });

    assert_true(abs(percentiles(0) - solution(0)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(1) - solution(1)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(2) - solution(2)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(3) - solution(3)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(4) - solution(4)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(5) - solution(5)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(6) - solution(6)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(7) - solution(7)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(8) - solution(8)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(9) - solution(9)) < type(1.0e-7), LOG);


    // Test
    vector.resize(21);
    vector.setValues({ type(0), type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8), type(9), type(10), type(11), type(12), type(13), type(14), type(15), type(16), type(17), type(18), type(19), type(20) });

    vector(20) = type(NAN);

    percentiles = opennn::percentiles(vector);

    solution.resize(10);
    solution.setValues({ type(1.5), type(3.5), type(5.5), type(7.5), type(9.5), type(11.5), type(13.5), type(15.5), type(17.5), type(19) });

    assert_true(abs(percentiles(0) - solution(0)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(1) - solution(1)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(2) - solution(2)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(3) - solution(3)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(4) - solution(4)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(5) - solution(5)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(6) - solution(6)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(7) - solution(7)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(8) - solution(8)) < type(1.0e-7), LOG);
    assert_true(abs(percentiles(9) - solution(9)) < type(1.0e-7), LOG);
}


void StatisticsTest::run_test_case()
{
    cout << "Running statistics test case...\n";

    // Minimum

    test_minimum();

    // Maximun

    test_maximum();

    // Mean

    test_mean();

    // Median

    test_median();

    // Variance

    test_variance();

    // Assymetry

    test_asymmetry();

    // Kurtosis

    test_kurtosis();

    // Standard deviation

    test_standard_deviation();

    // Quartiles

    test_quartiles();

    // Box plot

    test_box_plot();

    // Histogram

    test_count_empty_bins();
    test_calculate_minimum_frequency();
    test_calculate_maximum_frequency();
    test_calculate_most_populated_bin();
    test_calculate_minimal_centers();
    test_calculate_maximal_centers();
    test_calculate_bin();
    test_calculate_frequency();
    test_histogram();
    test_total_frequencies();
    test_histograms();

    // Minimal indices

    test_calculate_minimal_index();
    test_calculate_minimal_indices();

    // Maximal indices

    test_calculate_maximal_index();
    test_calculate_maximal_indices();

    // Percentiles

    test_percentiles();

    cout << "End of descriptives test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
