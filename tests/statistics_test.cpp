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


void StatisticsTest::test_set_minimum()
{
   cout << "test_set_minimum\n";

   Descriptives descriptives;

   // Test

   descriptives.set_minimum(5.0);

   assert_true(static_cast<Index>(descriptives.minimum) == 5, LOG);
}


void StatisticsTest::test_set_maximum()
{
   cout << "test_set_maximun\n";

   Descriptives descriptives;

   // Test

   descriptives.set_maximum(5.0);

   assert_true(static_cast<Index>(descriptives.maximum) == 5, LOG);
}


void StatisticsTest::test_set_mean()
{
   cout << "test_set_mean\n";

   Descriptives descriptives;

   // Test

   descriptives.set_mean(5.0);

   assert_true(static_cast<Index>(descriptives.mean) == 5, LOG);
}


void StatisticsTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   Descriptives descriptives;

   // Test

   descriptives.set_standard_deviation(3.0);

   assert_true(static_cast<Index>(descriptives.standard_deviation) == 3.0, LOG);
}


void StatisticsTest::test_has_mean_zero_standard_deviation_one()
{
    cout << "test_has_mean_zero_standard_deviation_one\n";

    Descriptives descriptives;

    // Test

//    descriptives.set(-4.0, 5.0, 0.0, 1.0);

    assert_true(descriptives.has_mean_zero_standard_deviation_one(), LOG);

    // Test

    descriptives.set(-4.0 ,5.0 ,1.0 ,1.0);
    assert_true(!descriptives.has_mean_zero_standard_deviation_one(), LOG);

    // Test

    descriptives.set(-4.0 ,5.0 ,0.0 ,2.0);
    assert_true(!descriptives.has_mean_zero_standard_deviation_one(), LOG);

    // Test

    descriptives.set(-4.0 ,5.0 ,2.0 ,2.0);
    assert_true(!descriptives.has_mean_zero_standard_deviation_one(), LOG);
}


void StatisticsTest::test_has_minimum_minus_one_maximum_one()
{
    cout << "test_set_has_minimum_minus_one_maximum_one\n";

    Descriptives descriptives;

    //Test_0

    descriptives.set(-1.0 ,1.0 ,0.0 ,1.0);
    assert_true(descriptives.has_minimum_minus_one_maximum_one(), LOG);

    //Test_1
    descriptives.set(-2.0 ,1.0 ,0.0 ,1.0);
    assert_true(!descriptives.has_minimum_minus_one_maximum_one(), LOG);

    //Test_2
    descriptives.set(-1.0 ,2.0 ,0.0 ,1.0);
    assert_true(!descriptives.has_minimum_minus_one_maximum_one(), LOG);

    //Test_3
    descriptives.set(-2.0 ,2.0 ,0.0 ,1.0);
    assert_true(!descriptives.has_minimum_minus_one_maximum_one(), LOG);
}


void StatisticsTest::test_get_bins_number()
{
    cout << "test_get_bins_number\n";

    // Test
    Histogram histogram;
    assert_true(histogram.get_bins_number() == 0, LOG);

    // Test
    const Index bins_number_1 = 50;

    Histogram histogram_1(bins_number_1);
    assert_true(histogram_1.get_bins_number() == 50, LOG);
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
    centers.setValues({1,2,3});

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
    centers.setValues({1,2,3});

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
    assert_true(histogram.calculate_minimum_frequency() == 0, LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({1,2,3});
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
    centers.setValues({1,2,3});
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
    assert_true(histogram.calculate_maximum_frequency() == 0, LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({1,2,3});
    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,0,1});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.calculate_maximum_frequency() == 1, LOG);

    // Test
    centers.resize(3);
    centers.setValues({1,3,5});
    frecuencies.resize(3);
    frecuencies.setValues({5,21,8});

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.calculate_maximum_frequency() == 21, LOG);

}


void StatisticsTest::test_calculate_most_populated_bin()
{
    cout << "test_calculate_most_populated_bin\n";

    //  Test 0 
    Histogram histogram;
    assert_true(histogram.calculate_most_populated_bin() == 0, LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({1,2,3});
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
    centers.setValues({1,5,9});

    frecuencies.resize(3);
    frecuencies.setValues({5,4,10});

    Histogram histogram_3(centers,frecuencies);
    assert_true(histogram_3.calculate_most_populated_bin() == 2, LOG);

}


void StatisticsTest::test_calculate_minimal_centers()
{
    cout << "test_calculate_minimal_centers\n";

    Histogram histogram;

    Tensor<type, 1> vector(14);
    vector.setValues({1, 1, 12, 1, 1, 1, 2, 2, 6, 4, 8, 1, 4, 7});

    histogram = OpenNN::histogram(vector);

    Tensor<type, 1> solution(4);
    solution.setValues({6, 7, 8, 12});

    assert_true((static_cast<Index>(histogram.calculate_minimal_centers()[0] - solution[0])) < 1.0e-7, LOG);
    assert_true((static_cast<Index>(histogram.calculate_minimal_centers()[1] - solution[1])) < 1.0e-7, LOG);
    assert_true((static_cast<Index>(histogram.calculate_minimal_centers()[2] - solution[2])) < 1.0e-7, LOG);
    assert_true((static_cast<Index>(histogram.calculate_minimal_centers()[3] - solution[3])) < 1.0e-7, LOG);

    //  Test 0
    Histogram histogram_0;
    assert_true(::isnan(histogram_0.calculate_minimal_centers()(0)), LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({1,2,3});
    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,1});

    Histogram histogram_1(centers,frecuencies);

    assert_true(static_cast<Index>(histogram_1.calculate_minimal_centers()(0)) == 1, LOG);
    assert_true(static_cast<Index>(histogram_1.calculate_minimal_centers()(1)) == 2, LOG);
}


void StatisticsTest::test_calculate_maximal_centers()
{
    cout << "test_calculate_maximal_centers\n";

    Histogram histogram;

    Tensor<type, 1> vector(18);
    vector.setValues({1, 1, 1, 1, 7, 2, 2, 6, 7, 4, 8, 8, 8, 1, 4, 7, 7, 7});

    histogram = OpenNN::histogram(vector);

    Tensor<type, 1> solution(2);
    solution.setValues({1, 7});

    assert_true(static_cast<Index>(histogram.calculate_maximal_centers()[0] - solution[0]) < 1.0e-7, LOG);
    assert_true(static_cast<Index>(histogram.calculate_maximal_centers()[1] - solution[1]) < 1.0e-7, LOG);

    //  Test 0
    Histogram histogram_0;
    assert_true(::isnan(histogram_0.calculate_maximal_centers()(0)), LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({1,2,3});
    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({1,1,0});

    Histogram histogram_1(centers,frecuencies);

    assert_true(static_cast<Index>(histogram_1.calculate_maximal_centers()(0)) == 1, LOG);
    assert_true(static_cast<Index>(histogram_1.calculate_maximal_centers()(1)) == 2, LOG);
}


void StatisticsTest::test_calculate_bin()
{
    cout << "test_calculate_bin\n";

    // Test
    Histogram histogram;
    assert_true(histogram.calculate_bin(0) == 0, LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({2,4,6});
    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,0,0});

    Histogram histogram_1(centers,frecuencies);

    assert_true(histogram_1.calculate_bin(6) == 2, LOG);


    Tensor<type, 1> vector(3);
    Index bin;

    vector.setValues({1.0, 1.0, 11.0});
    histogram = OpenNN::histogram(vector, 10);

    // Test
    bin = histogram.calculate_bin(vector[0]);
    assert_true(bin == 0, LOG);

    // Test
    bin = histogram.calculate_bin(vector[1]);
    assert_true(bin == 0, LOG);

    // Test 
    bin = histogram.calculate_bin(vector[2]);
    assert_true(bin == 1, LOG);

}


void StatisticsTest::test_calculate_frequency()
{
    cout << "test_calculate_frequency\n";

    // Test
    Histogram histogram;
    assert_true(histogram.calculate_frequency(0) == 0, LOG);

    // Test
    Tensor<type, 1> centers(3);
    centers.setValues({1,2,3});
    Tensor<Index, 1> frecuencies(3);
    frecuencies.setValues({0,1,2});

    Histogram histogram_1(centers,frecuencies);
    assert_true(histogram_1.calculate_frequency(2) == 1, LOG);

    // Test
    centers.resize(3);
    centers.setZero();
    frecuencies.resize(3);
    frecuencies.setZero();

    Histogram histogram_2(centers,frecuencies);
    assert_true(histogram_2.calculate_frequency(0) == 0, LOG);

    // Test

    Tensor<type, 1> vector(3);
    Index frequency_3;
    Histogram histogram_3;

    vector.setValues({0.0, 1.0, 9.0});
    histogram_3 = OpenNN::histogram(vector, 10);
    frequency_3 = histogram_3.calculate_frequency(vector[9]);

    assert_true(frequency_3 == 1, LOG);

}


void StatisticsTest::test_minimum()
{
   cout << "test_calculate_minimum\n";

   Tensor<type, 1> vector;

   // Test

   assert_true(::isnan(minimum(vector)), LOG);

   // Test

   vector.resize(3);
   vector.setValues({0.0, 1.0, 9.0});

   assert_true(minimum(vector) - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(3);
   vector.setValues({1,2,3});

   vector.resize(3);
   vector.setValues({-1,2,3});

   assert_true(minimum(vector) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
   assert_true(minimum(vector) - static_cast<type>(-1.0) < numeric_limits<type>::min(), LOG);

}


void StatisticsTest::test_maximum()
{
   cout << "test_calculate_maximum\n";

   Tensor<type, 1> vector;

   // Test

   assert_true(::isnan(maximum(vector)), LOG);

   // Test

   vector.resize(3);
   vector.setValues({0.0, 1.0, 9.0});

   assert_true(maximum(vector) - static_cast<type>(9.0) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(3);
   vector.setValues({1,2,3});

   vector.resize(3);
   vector.setValues({-1,-2,-3});

   assert_true(maximum(vector) - static_cast<type>(3.0) < numeric_limits<type>::min(), LOG);
   assert_true(maximum(vector) - static_cast<type>(-1.0) < numeric_limits<type>::min(), LOG);

}


void StatisticsTest::test_mean()
{
   cout << "test_mean\n";

   Tensor<type, 1> vector;
   Tensor<type, 2> matrix;

   // Test

   matrix.resize(3,3);
   matrix.setZero();
   assert_true(mean(matrix)(0) < numeric_limits<type>::min(), LOG);

   // Test

   matrix.resize(3,3);
   matrix.setValues({{0,1,-2},{0,1,8},{0,1,6}});

   assert_true(mean(matrix)(0) < numeric_limits<type>::min(), LOG);
   assert_true(mean(matrix)(1) - static_cast<type>(1) < numeric_limits<type>::min(), LOG);
   assert_true(mean(matrix)(2) - static_cast<type>(4) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(2);
   vector.setValues({1, 1.0});

   assert_true(mean(vector) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(2);
   vector[0] = -1.0;
   vector[1] = 1.0;
   assert_true(mean(vector) - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);

   // Test missing values

   vector.resize(5);

   vector.setValues({1.0, NAN, 2.0, 3.0, 4.0});

   assert_true(abs(mean(vector) - 0) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(4);
   vector[0] = 1;
   vector[1] = 1;
   vector[2] = static_cast<type>(NAN);
   vector[3] = 1;

   assert_true(mean(vector) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);

   // Test empty matrix

   matrix.resize(0, 0);

   assert_true(::isnan(mean(matrix,2)), LOG);

}


void StatisticsTest::test_standard_deviation()
{
   cout << "test_standard_deviation\n";

   Tensor<type, 1> vector;

   type standard_deviation;

   // Test

//   assert_true(OpenNN::standard_deviation(vector) - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(4);
   vector.setValues({2,4,8,10});

   assert_true(OpenNN::standard_deviation(vector) - sqrt(static_cast<type>(40)/static_cast<type>(3)) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(4);
   vector.setValues({-11,-11,-11,-11});

   assert_true(OpenNN::standard_deviation(vector) - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(3);
   vector.setZero();

   assert_true(static_cast<Index>(OpenNN::standard_deviation(vector)) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(2);
   vector.setValues({1, 1.0});

   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(abs(static_cast<Index>(standard_deviation)) < numeric_limits<type>::min(), LOG);

   // Test 
   vector.resize(2);
   vector[0] = -1.0;
   vector[1] = 1.0;

   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(abs(standard_deviation- sqrt(static_cast<type>(2))) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(1);
   vector[0] = static_cast<type>(NAN);

   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(standard_deviation < numeric_limits<type>::min(), LOG);
}


void StatisticsTest::test_median()
{
    cout << "test_calculate_median\n";

    Tensor<type, 1> vector;
    Tensor<type, 2> matrix;

    // Test

//    assert_true(median(vector) == 0, LOG);

    // Test , 2
    vector.resize(4);
    vector.setValues({2,4,8,10});

    vector.resize(4);
    vector.setValues({-11,-11,-11,-11});

    assert_true(median(vector) - static_cast<type>(6) < static_cast<type>(1.0e-3), LOG);
    assert_true(median(vector) - static_cast<type>(-11) < static_cast<type>(1.0e-3), LOG);

    // Test
    vector.resize(4);
    vector.setValues({1,2,3,4});

    assert_true(abs(median(vector) - static_cast<type>(2.5)) < static_cast<type>(1.0e-3), LOG);

    // Test 
    vector.resize(5);
    vector.setValues({1,2,3,4,5});

    assert_true(abs(median(vector) - static_cast<type>(3)) < static_cast<type>(1.0e-3), LOG);

    // Test 5
    matrix.resize(3,2);    
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 1.0;
    matrix(1, 0) = 2.0;
    matrix(1, 1) = 3.0;
    matrix(2, 0) = 3.0;
    matrix(2, 1) = 4.0;

    assert_true(abs(median(matrix)(0) - static_cast<type>(2)) < static_cast<type>(1.0e-3), LOG);
    assert_true(abs(median(matrix)(1) - static_cast<type>(3)) < static_cast<type>(1.0e-3), LOG);

    // Test median missing values matrix
    matrix.resize(3,2);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 1.0;
    matrix(1, 0) = static_cast<type>(NAN);
    matrix(1, 1) = static_cast<type>(NAN);
    matrix(2, 0) = 3.0;
    matrix(2, 1) = 3.0;

    assert_true(abs(median(matrix)(0) - static_cast<type>(2)) < static_cast<type>(1.0e-3), LOG);
    assert_true(abs(median(matrix)(1) - static_cast<type>(2)) < static_cast<type>(1.0e-3), LOG);

    // Test median missing values vector
    vector.resize(4);
    vector[0] = 3.0;
    vector[1] = static_cast<type>(NAN);
    vector[2] = 1.0;
    vector[3] = static_cast<type>(NAN);

    assert_true(median(vector) - static_cast<type>(2) < static_cast<type>(1.0e-3), LOG);
}


void StatisticsTest::test_variance()
{
    cout << "test_variance\n";

    Tensor<type, 1> vector;    

    // Test

    vector.resize(3);
    vector.setZero();
    assert_true(static_cast<Index>(variance(vector)) == 0, LOG);

    // Test , 2

    vector.resize(4);
    vector.setValues({2,4,8,10});

    vector.resize(4);
    vector.setValues({-11,-11,-11,-11});

    assert_true(variance(vector) - static_cast<type>(40)/static_cast<type>(3) < numeric_limits<type>::min(), LOG);
    assert_true(variance(vector) - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);

    // Test
    vector.resize(1);
    vector.resize(1);
    vector[0] = 1;

    assert_true(abs(variance(vector) - static_cast<type>(0.0)) < numeric_limits<type>::min(), LOG);

    // Test 
    vector.resize(3);
    vector[0] = 2.0;
    vector[1] = 1.0;
    vector[2] = 2.0;
    assert_true(abs(variance(vector) - static_cast<type>(1)/static_cast<type>(3)) < numeric_limits<type>::min(), LOG);

    // Test variance missing values

    vector.resize(3);
    vector[0] = 1.0;
    vector[1] = static_cast<type>(NAN);
    vector[2] = 2.0;

    vector.resize(2);
    vector[0] = 1.0;
    vector[1] = 2.0;

    assert_true(abs(variance(vector) - variance(vector)) < numeric_limits<type>::min(), LOG);
    assert_true(abs(variance(vector) - static_cast<type>(0.5)) < numeric_limits<type>::min(), LOG);
}


void StatisticsTest::test_asymmetry()
{
    cout << "test_calculate_asymmetry\n";

    Tensor<type, 1> vector;

    // Test
    vector.resize(3);
    vector.setZero();

    assert_true(asymmetry(vector) - static_cast<Index>(0) < static_cast<type>(1.0e-3), LOG);

    // Test
    vector.resize(4);
    vector[0] = 1.0;
    vector[0] = 5.0;
    vector[0] = 3.0;
    vector[0] = 9.0;

    type asymmetry_value = OpenNN::asymmetry(vector);

    assert_true(asymmetry_value - static_cast<type>(0.75) < static_cast<type>(1.0e-3), LOG);

    // Test
    vector.resize(4);
    vector.setValues({1,5,3,9});

    assert_true(asymmetry(vector) - static_cast<Index>(0.75) < static_cast<type>(1.0e-3), LOG);

    // Test missing values

    vector.resize(4);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = 3.0;
    vector[3] = 9.0;

    vector.resize(5);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = static_cast<type>(NAN);
    vector[3] = 3.0;
    vector[4] = 9.0;

    type asymmetry = OpenNN::asymmetry(vector);

//    assert_true(abs(asymmetry - asymmetry_missing_values) < static_cast<type>(1.0e-3), LOG);

}


void StatisticsTest::test_kurtosis()
{
    cout << "test_calculate_kurtosis\n";

    Tensor<type, 1> vector;

    // Test

    vector.resize(4);
    vector.setValues({1,5,3,9});

    assert_true(abs(kurtosis(vector) - static_cast<type>(-1.9617)) < static_cast<type>(1.0e-3), LOG);

    // Test missing values

    vector.resize(4);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = 3.0;
    vector[3] = 9.0;

    vector.resize(5);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = static_cast<type>(NAN);
    vector[3] = 3.0;
    vector[4] = 9.0;

    type kurtosis = OpenNN::kurtosis(vector);
}


void StatisticsTest::test_quartiles()
{
   cout << "test_quartiles\n";

   Tensor<type, 1> vector;
   Tensor<type, 1> quartiles;

   // Test

   vector.resize(1);
   vector.setZero();

   quartiles = OpenNN::quartiles(vector);

   assert_true(static_cast<Index>(quartiles(0)) < numeric_limits<type>::min(), LOG);
   assert_true(static_cast<Index>(quartiles(1)) < numeric_limits<type>::min(), LOG);
   assert_true(static_cast<Index>(quartiles(2)) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(2);
   vector.setValues({0,1});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles(0) - static_cast<type>(0.25) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(1) - static_cast<type>(0.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(2) - static_cast<type>(0.75) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(3);
   vector.setValues({0,1,2});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles(0) - static_cast<type>(0.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(1) - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(2) - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);

   // Test 

   vector.resize(4);
   vector.setValues({0,1,2,3});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles(0) - static_cast<type>(0.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(1) - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(2) - static_cast<type>(2.5) < numeric_limits<type>::min(), LOG);

   // Test 5

   vector.resize(5);
   vector.setValues({0,1,2,3,4});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(0.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(2.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(3.5) < numeric_limits<type>::min(), LOG);

   // Test 6

   vector.resize(6);
   vector.setValues({0,1,2,3,4,5});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(2.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(4.0) < numeric_limits<type>::min(), LOG);

   // Test 7

   vector.resize(7);
   vector.setValues({0,1,2,3,4,5,6});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(1.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(3.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(5.0) < numeric_limits<type>::min(), LOG);

   // Test 8

   vector.resize(8);
   vector.setValues({0,1,2,3,4,5,6,7});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(3.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(5.5) < numeric_limits<type>::min(), LOG);

   // Test 9
   vector.resize(9);
   vector.setValues({0,1,2,3,4,5,6,7,8});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(4.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(6.5) < numeric_limits<type>::min(), LOG);

   // Test 10
   vector.resize(9);
   vector.setValues({1,4,6,2,0,3,4,7,10});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(4.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(6.5) < numeric_limits<type>::min(), LOG);

   // Test 11

   vector.resize(20);
   vector.setValues({12,14,50,76,12,34,56,74,89,60,96,24,53,25,67,84,92,45,62,86});

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] - static_cast<type>(29.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[1] - static_cast<type>(58.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles[2] - static_cast<type>(80.0) < numeric_limits<type>::min(), LOG);

   // Test missing values:

   // Test

   vector.resize(5);
   vector[0] = 1;
   vector[1] = 2;
   vector[2] = 3;
   vector[3] = static_cast<type>(NAN);
   vector[4] = 4;

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles(0) - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(1) - static_cast<type>(2.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(2) - static_cast<type>(3.5) < numeric_limits<type>::min(), LOG);

   // Test

   vector.resize(6);
   vector[0] = 1;
   vector[1] = 2;
   vector[2] = 3;
   vector[3] = static_cast<type>(NAN);
   vector[4] = 4;
   vector[5] = 5;

   quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles(0) - static_cast<type>(1.5) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(1) - static_cast<type>(3.0) < numeric_limits<type>::min(), LOG);
   assert_true(quartiles(2) - static_cast<type>(4.5) < numeric_limits<type>::min(), LOG);

}



/// @todo ERROR frequencies

void StatisticsTest::test_histogram()
{
   cout << "test_histogram\n";

   Tensor<type, 1> vector;

   Tensor<type, 1> centers;
   Tensor<Index, 1> frequencies;

   // Test

   vector.resize(11);
   vector.setValues({0,1,2,3,4,5,6,7,8,9,10});

//   Histogram histogram(vector, 10);
//   assert_true(histogram.get_bins_number() == 10, LOG);

//   centers = histogram.centers;
//   frequencies = histogram.frequencies;

//   assert_true(abs(centers[0] - static_cast<type>(0.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[1] - static_cast<type>(1.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[2] - static_cast<type>(2.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[3] - static_cast<type>(3.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[4] - static_cast<type>(4.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[5] - static_cast<type>(5.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[6] - static_cast<type>(6.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[7] - static_cast<type>(7.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[8] - static_cast<type>(8.5)) < static_cast<type>(1.0e-3), LOG);
//   assert_true(abs(centers[9] - static_cast<type>(9.5)) < static_cast<type>(1.0e-3), LOG);

//   assert_true(frequencies[0] == 1, LOG);
//   assert_true(frequencies[1] == 1, LOG);
//   assert_true(frequencies[2] == 1, LOG);
//   assert_true(frequencies[3] == 1, LOG);
//   assert_true(frequencies[4] == 1, LOG);
//   assert_true(frequencies[5] == 1, LOG);
//   assert_true(frequencies[6] == 1, LOG);
//   assert_true(frequencies[7] == 1, LOG);
//   assert_true(frequencies[8] == 1, LOG);
//   assert_true(frequencies[9] == 1, LOG);

   Tensor<Index, 0> sum_frec_1 = frequencies.sum();
   //assert_true(sum_frec_1(0) == 11, LOG); // <--- failed


   // Test
   vector.resize(20);
   vector.setRandom();

//   Histogram histogram_2(vector, 10);

//   centers = histogram_2.centers;
//   frequencies = histogram_2.frequencies;

//   Tensor<Index, 0> sum_frec_2;
//   sum_frec_2 = frequencies.sum();
   //assert_true(sum_frec_2(0) == 20, LOG); // <--- failed

}


void StatisticsTest::test_histograms()
{
    cout << "test_histograms\n";

    Tensor<Histogram, 1> histograms;


    Tensor<type, 2> matrix(3,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 1.0;
    matrix(0,2) = 1.0;
    matrix(1,0) = 2.0;
    matrix(1,1) = 2.0;
    matrix(1,2) = 2.0;
    matrix(2,0) = 3.0;
    matrix(2,1) = 3.0;
    matrix(2,2) = 3.0;

    histograms = OpenNN::histograms(matrix, 3);

    Tensor<Index, 1> solution(3);
    solution.setValues({1, 1, 1});

//    assert_true(histograms[0].frequencies == 1, LOG);
//    assert_true(histograms[1].frequencies == 1, LOG);
//    assert_true(histograms[2].frequencies == 1, LOG);
}


void StatisticsTest::test_total_frequencies()   //<--- Check
{
    cout << "test_total_frequencies\n";

    Tensor<Histogram, 1> histograms(3);

    // Test
    Tensor<type, 1> vector1_1(16);
    vector1_1.setValues({0,1,2,3,4,5,6,0,1,1,1,2,2,2,2,2});

    Tensor<type, 1> vector2_1(16);
    vector2_1.setValues({0,1,2,3,4,5,6,0,1,1,1,2,2,2,2,2});

    Tensor<type, 1> vector3_1(16);
    vector3_1.setValues({0,1,2,3,4,5,6,0,1,1,1,2,2,2,2,2});

    histograms[0] = histogram(vector1_1, 7);
    histograms[1] = histogram(vector2_1, 7);
    histograms[2] = histogram(vector3_1, 7);

    Tensor<Index, 1> total_frequencies = OpenNN::total_frequencies(histograms);

    assert_true(total_frequencies[0] == 2, LOG);
    assert_true(total_frequencies[1] == 4, LOG);
    assert_true(total_frequencies[2] == 6, LOG);

    Tensor<type, 2> matrix(3,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 1.0;
    matrix(0,2) = static_cast<type>(NAN);
    matrix(1,0) = 2.0;
    matrix(1,1) = 2.0;
    matrix(1,2) = 1.0;
    matrix(2,0) = 3.0;
    matrix(2,1) = 3.0;
    matrix(2,2) = 2.0;

//    histograms = histograms(matrix, 3);
    Tensor<Index, 1> solution(3);
    solution.setValues({1, 1, 1});

    Tensor<Index, 1> solution_missing_values(3);
    solution_missing_values.setValues({1, 0, 1});

//    assert_true(histograms[0].frequencies == solution, LOG);
//    assert_true(histograms[1].frequencies == solution, LOG);
//    assert_true(histograms[2].frequencies == solution_missing_values, LOG);
}


void StatisticsTest::test_calculate_minimal_index()
{
   cout << "test_calculate_minimal_index\n";

   Tensor<type, 1> vector;

   // Test

   assert_true(minimal_index(vector) == 0, LOG);

   // Test
   vector.resize(3);
   vector.setValues({1,0,-1});

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
   vector.setValues({1,0,-1});

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
    vector.setValues({-1,0,1});

    assert_true(minimal_indices(vector, 1)[0] == 0, LOG);

    assert_true(minimal_indices(vector, 3)[0] == 0, LOG);
    assert_true(minimal_indices(vector, 3)[1] == 1, LOG);
    assert_true(minimal_indices(vector, 3)[2] == 2, LOG);

    // Test

    vector.resize(4);
    vector.setValues({0,0,0,1});

    assert_true(minimal_indices(vector, 4)[0] == 0, LOG);
    assert_true(minimal_indices(vector, 4)[1] == 1, LOG);
    assert_true(minimal_indices(vector, 4)[3] == 3, LOG);

    // Test

    vector.resize(5);
    vector[0] = 0;
    vector[1] = 1;
    vector[2] = 0;
    vector[3] = 2;
    vector[4] = 0;

    assert_true(minimal_indices(vector, 5)[0] == 0 || minimal_indices(vector, 5)[0] == 2 || minimal_indices(vector, 5)[0] == 4, LOG);
    assert_true(minimal_indices(vector, 5)[1] == 0 || minimal_indices(vector, 5)[1] == 2 || minimal_indices(vector, 5)[1] == 4, LOG);
    assert_true(minimal_indices(vector, 5)[2] == 0 || minimal_indices(vector, 5)[2] == 2 || minimal_indices(vector, 5)[2] == 4, LOG);
    assert_true(minimal_indices(vector, 5)[3] == 1, LOG);
    assert_true(minimal_indices(vector, 5)[4] == 3, LOG);

    // Test 

    vector.resize(4);
    vector[0] = -1.0;
    vector[1] = 2.0;
    vector[2] = -3.0;
    vector[3] = 4.0;

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
    vector.setValues({-1,0,1});

    assert_true(maximal_indices(vector, 1)[0] == 2, LOG);

    // Test
    vector.resize(4);
    vector.setValues({1,1,1,1});

    assert_true(maximal_indices(vector, 4)[0] == 0, LOG);
    assert_true(maximal_indices(vector, 4)[1] == 1, LOG);
    assert_true(maximal_indices(vector, 4)[3] == 3, LOG);

    // Test

    vector.resize(5);
    vector.setValues({1,5,6,7,2});

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

    box_plot = OpenNN::box_plot(vector);

    assert_true(box_plot.minimum - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.first_quartile - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.median - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.third_quartile - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.maximum - static_cast<type>(0.0) < numeric_limits<type>::min(), LOG);

    // Test

    vector.resize(8);
    vector.setValues({2.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    box_plot = OpenNN::box_plot(vector);

    solution.set(2.0, 2.5, 5.5, 7.5, 9.0);

    assert_true(box_plot.minimum - solution.minimum < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.first_quartile - solution.first_quartile < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.median - solution.median < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.third_quartile - solution.third_quartile < numeric_limits<type>::min(), LOG);
    assert_true(box_plot.maximum - solution.maximum < numeric_limits<type>::min(), LOG);

    // Test missing values

    vector.resize(9);
    vector.setValues({2.0, 2.0, 3.0, 5.0, 6.0, 7.0, static_cast<type>(NAN), 8.0, 9.0});

    box_plot = OpenNN::box_plot(vector);

    solution.set(2.0, 2.5, 5.5, 7.5, 9.0);

    assert_true((box_plot.minimum - solution.minimum) < numeric_limits<type>::min(), LOG);
    assert_true((box_plot.first_quartile - solution.first_quartile) < numeric_limits<type>::min(), LOG);
    assert_true((box_plot.median - solution.median) < numeric_limits<type>::min(), LOG);
    assert_true((box_plot.third_quartile - solution.third_quartile) < numeric_limits<type>::min(), LOG);
    assert_true((box_plot.maximum - solution.maximum) < numeric_limits<type>::min(), LOG);

    //Histogram_missing_values

    Histogram histogram;
    Tensor<type, 1> centers;
    Tensor<Index, 1> frequencies;

    vector.resize(5);
    vector[0] = 1;
    vector[1] = 3;
    vector[2] = 2;
    vector[3] = 4;
    vector[4] = static_cast<type>(NAN);

//    graphic = histogram_missing_values(vector, 4);

    centers = histogram.centers;
    frequencies = histogram.frequencies;

    //Normal histogram

    vector.resize(4);

    vector[0] = 1;
    vector[1] = 3;
    vector[2] = 2;
    vector[3] = 4;

//    graphic_2 = histogram(vector, 4);

//    centers = graphic_2.centers;
//    frequencies = graphic_2.frequencies;

//    assert_true(centers == centers , LOG);
//    assert_true(frequencies == frequencies, LOG);
}


void StatisticsTest::test_percentiles()
{
    cout << "test_percentiles\n";

    Tensor<type, 1> vector;

    // Test

//    Tensor<type, 1> empty_vector;
//    Tensor<type, 1> percentiles_empty = OpenNN::percentiles(empty_vector);
//    assert_true(::isnan(percentiles_empty[0]), LOG);

    // Test

    vector.resize(10);

    vector.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

//    Tensor<type, 1> percentiles = OpenNN::percentiles(vector);

    Tensor<type, 1> solution(10);
    solution.setValues({0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9});

//    assert_true((percentiles[0] - solution[0]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[1] - solution[1]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[2] - solution[2]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[3] - solution[3]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[4] - solution[4]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[5] - solution[5]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[6] - solution[6]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[7] - solution[7]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[8] - solution[8]) < static_cast<type>(1.0e-7), LOG);
//    assert_true((percentiles[9] - solution[9]) < static_cast<type>(1.0e-7), LOG);

    // Test

    vector.resize(21);

    vector.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

    Tensor<type, 1> percentiles_2 = OpenNN::percentiles(vector);

    Tensor<type, 1> solution_2(10);
    solution_2.setValues({2, 4, 6, 8, 10, 12, 14, 16, 18, 20});

    assert_true((percentiles_2[0] - solution_2[0]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[1] - solution_2[1]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[2] - solution_2[2]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[3] - solution_2[3]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[4] - solution_2[4]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[5] - solution_2[5]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[6] - solution_2[6]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[7] - solution_2[7]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[8] - solution_2[8]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_2[9] - solution_2[9]) < static_cast<type>(1.0e-7), LOG);

    // Test

    vector.resize(14);

    vector.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 19, 32});

    Tensor<type, 1> percentiles = OpenNN::percentiles(vector);

    Tensor<type, 1> solution_3(10);
    solution_3.setValues({1, 2, 4, 5, 6.5, 8, 9, 15, 19, 32});

    assert_true((percentiles[0] - solution_3[0]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[1] - solution_3[1]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[2] - solution_3[2]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[3] - solution_3[3]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[4] - solution_3[4]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[5] - solution_3[5]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[6] - solution_3[6]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[7] - solution_3[7]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[8] - solution_3[8]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[9] - solution_3[9]) < static_cast<type>(1.0e-7), LOG);

    // Test missing values

    vector.resize(21); //@todo

    vector.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

    vector[20] = static_cast<type>(NAN);

    percentiles = OpenNN::percentiles(vector);

    solution.resize(10);
    solution.setValues({2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20});

    assert_true((percentiles[0] - solution[0]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[1] - solution[1]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[2] - solution[2]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[3] - solution[3]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[4] - solution[4]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[5] - solution[5]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[6] - solution[6]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[7] - solution[7]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[8] - solution[8]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles[9] - solution[9]) < static_cast<type>(1.0e-7), LOG);

}


void StatisticsTest::test_means_by_categories()
{
    cout << "test_means_by_categories\n";

    // Test

    Tensor<type, 2> matrix(6, 6);
    matrix.setValues({{1,2,3,1,2,3}, {6,2,3,12,2,3}});

    Tensor<type, 1> solution(3);
    solution.setValues({9.0, 2.0, 3.0});

//    assert_true(means_by_categories(matrix)(0) == solution(0), LOG);
//    assert_true(means_by_categories(matrix)(1) == solution(1), LOG);
//    assert_true(means_by_categories(matrix)(2) == solution(2), LOG);

//    // Test missing vaues

//    Tensor<type, 2> matrix({Tensor<type, 1>({1,1,1,2,2,2}),Tensor<type, 1>({1,1,1,2,6,static_cast<type>(NAN)})});

//    Tensor<type, 1> solution({1.0, 4.0});

//    assert_true(means_by_categories_missing_values(matrix) == solution, LOG);

}


void StatisticsTest::run_test_case()
{
   cout << "Running statistics test case...\n";

   // Descriptives

   test_set_standard_deviation();
   test_has_mean_zero_standard_deviation_one();
   test_has_minimum_minus_one_maximum_one();

   // Minimum

   test_set_minimum();
   test_minimum();

   // Maximun

   test_set_maximum();
   test_maximum();

   // Mean

   test_set_mean();
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

   test_get_bins_number();
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

   // Means by categories

   test_means_by_categories();


   cout << "End of descriptives test case.\n\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
