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


void StatisticsTest::test_constructor()
{
   cout << "test_constructor\n";
}


void StatisticsTest::test_destructor()
{
   cout << "test_destructor\n";
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

    // Test 0

//    descriptives.set(-4.0, 5.0, 0.0, 1.0);

    assert_true(descriptives.has_mean_zero_standard_deviation_one(), LOG);

    // Test 1
    Descriptives descriptives_1(-4.0 ,5.0 ,1.0 ,1.0);
    assert_true(!descriptives_1.has_mean_zero_standard_deviation_one(), LOG);

    // Test 2
    Descriptives descriptives_2(-4.0 ,5.0 ,0.0 ,2.0);
    assert_true(!descriptives_2.has_mean_zero_standard_deviation_one(), LOG);

    // Test 3
    Descriptives descriptives_3(-4.0 ,5.0 ,2.0 ,2.0);
    assert_true(!descriptives_3.has_mean_zero_standard_deviation_one(), LOG);
}


void StatisticsTest::test_has_minimum_minus_one_maximum_one()
{
    cout << "test_set_has_minimum_minus_one_maximum_one\n";

    //Test_0
    Descriptives descriptives(-1.0 ,1.0 ,0.0 ,1.0);
    assert_true(descriptives.has_minimum_minus_one_maximum_one(), LOG);

    //Test_1
    Descriptives descriptives_1(-2.0 ,1.0 ,0.0 ,1.0);
    assert_true(!descriptives_1.has_minimum_minus_one_maximum_one(), LOG);

    //Test_2
    Descriptives descriptives_2(-1.0 ,2.0 ,0.0 ,1.0);
    assert_true(!descriptives_2.has_minimum_minus_one_maximum_one(), LOG);

    //Test_3
    Descriptives descriptives_3(-2.0 ,2.0 ,0.0 ,1.0);
    assert_true(!descriptives_3.has_minimum_minus_one_maximum_one(), LOG);
}


void StatisticsTest::test_get_bins_number()
{
    cout << "test_get_bins_number\n";

    // Test 0
    Histogram histogram;
    assert_true(histogram.get_bins_number() == 0, LOG);

    // Test 1
    const Index bins_number_1 = 50;

    Histogram histogram_1(bins_number_1);
    assert_true(histogram_1.get_bins_number() == 50, LOG);
}


void StatisticsTest::test_count_empty_bins()
{
    cout << "test_count_empty_bins\n";

    // Test 0
    Histogram histogram;
    assert_true(histogram.count_empty_bins() == 0, LOG);

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({1,1,0});

    Histogram histogram_1(centers_1,frecuencies_1);
    assert_true(histogram_1.count_empty_bins() == 1, LOG);

    // Test 2
    Tensor<type, 1> centers_2(3);
    centers_2.setZero();
    Tensor<Index, 1> frecuencies_2(3);
    frecuencies_2.setZero();

    Histogram histogram_2(centers_2,frecuencies_2);
    assert_true(histogram_2.count_empty_bins() == 3, LOG);

    // Test 3
    Tensor<type, 1> centers_3(3);
    centers_3.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_3(3);
    frecuencies_3.setZero();

    Histogram histogram_3(centers_3,frecuencies_3);
    assert_true(histogram_3.count_empty_bins() == 3, LOG);
}


void StatisticsTest::test_calculate_minimum_frequency()
{
    cout << "test_calculate_minimun_frecuency\n";

    // Test 0
    Histogram histogram;
    assert_true(histogram.calculate_minimum_frequency() == 0, LOG);

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({1,1,0});

    Histogram histogram_1(centers_1,frecuencies_1);
    assert_true(histogram_1.calculate_minimum_frequency() == 0, LOG);

    // Test 2
    Tensor<type, 1> centers_2(3);
    centers_2.setZero();
    Tensor<Index, 1> frecuencies_2(3);
    frecuencies_2.setZero();

    Histogram histogram_2(centers_2,frecuencies_2);
    assert_true(histogram_2.calculate_minimum_frequency() == 0, LOG);

    // Test 3
    Tensor<type, 1> centers_3(3);
    centers_3.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_3(3);
    frecuencies_3.setValues({5,4,10});

    Histogram histogram_3(centers_3,frecuencies_3);
    assert_true(histogram_3.calculate_minimum_frequency() == 4, LOG);
}


void StatisticsTest::test_calculate_maximum_frequency()
{
    cout << "test_calculate_maximum_frequency\n";

    // Test 0
    Histogram histogram;
    assert_true(histogram.calculate_maximum_frequency() == 0, LOG);

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({1,0,1});

    Histogram histogram_1(centers_1,frecuencies_1);
    assert_true(histogram_1.calculate_maximum_frequency() == 1, LOG);

    // Test 2
    Tensor<type, 1> centers_2(3);
    centers_2.setValues({1,3,5});
    Tensor<Index, 1> frecuencies_2(3);
    frecuencies_2.setValues({5,21,8});

    Histogram histogram_2(centers_2,frecuencies_2);
    assert_true(histogram_2.calculate_maximum_frequency() == 21, LOG);

}


void StatisticsTest::test_calculate_most_populated_bin()
{
    cout << "test_calculate_most_populated_bin\n";

    //  Test 0 
    Histogram histogram;
    assert_true(histogram.calculate_most_populated_bin() == 0, LOG);

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({0,0,1});

    Histogram histogram_1(centers_1,frecuencies_1);
    assert_true(histogram_1.calculate_most_populated_bin() == 2, LOG);

    // Test 2
    Tensor<type, 1> centers_2(3);
    centers_2.setZero();
    Tensor<Index, 1> frecuencies_2(3);
    frecuencies_2.setZero();

    Histogram histogram_2(centers_2,frecuencies_2);
    assert_true(histogram_2.calculate_most_populated_bin() == 0, LOG);

    // Test 3
    Tensor<type, 1> centers_3(3);
    centers_3.setValues({1,5,9});
    Tensor<Index, 1> frecuencies_3(3);
    frecuencies_3.setValues({5,4,10});

    Histogram histogram_3(centers_3,frecuencies_3);
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

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({0,0,1});

    Histogram histogram_1(centers_1,frecuencies_1);

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

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({1,1,0});

    Histogram histogram_1(centers_1,frecuencies_1);

    assert_true(static_cast<Index>(histogram_1.calculate_maximal_centers()(0)) == 1, LOG);
    assert_true(static_cast<Index>(histogram_1.calculate_maximal_centers()(1)) == 2, LOG);
}


void StatisticsTest::test_calculate_bin()
{
    cout << "test_calculate_bin\n";

    // Test 0
    Histogram histogram;
    assert_true(histogram.calculate_bin(0) == 0, LOG);

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({2,4,6});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({0,0,0});

    Histogram histogram_1(centers_1,frecuencies_1);

    assert_true(histogram_1.calculate_bin(6) == 2, LOG);


    Tensor<type, 1> vector(3);
    Index bin;

    vector.setValues({1.0, 1.0, 11.0});
    histogram = OpenNN::histogram(vector, 10);

    // Test 2
    bin = histogram.calculate_bin(vector[0]);
    assert_true(bin == 0, LOG);

    // Test 3
    bin = histogram.calculate_bin(vector[1]);
    assert_true(bin == 0, LOG);

    // Test 4
    bin = histogram.calculate_bin(vector[2]);
    assert_true(bin == 1, LOG);

}


void StatisticsTest::test_calculate_frequency()
{
    cout << "test_calculate_frequency\n";

    // Test 0
    Histogram histogram;
    assert_true(histogram.calculate_frequency(0) == 0, LOG);

    // Test 1
    Tensor<type, 1> centers_1(3);
    centers_1.setValues({1,2,3});
    Tensor<Index, 1> frecuencies_1(3);
    frecuencies_1.setValues({0,1,2});

    Histogram histogram_1(centers_1,frecuencies_1);
    assert_true(histogram_1.calculate_frequency(2) == 1, LOG);

    // Test 2
    Tensor<type, 1> centers_2(3);
    centers_2.setZero();
    Tensor<Index, 1> frecuencies_2(3);
    frecuencies_2.setZero();

    Histogram histogram_2(centers_2,frecuencies_2);
    assert_true(histogram_2.calculate_frequency(0) == 0, LOG);

    // Test 3

    Tensor<type, 1> vector_3(3);
    Index frequency_3;
    Histogram histogram_3;

    vector_3.setValues({0.0, 1.0, 9.0});
    histogram_3 = OpenNN::histogram(vector_3, 10);
    frequency_3 = histogram_3.calculate_frequency(vector_3[9]);

    assert_true(frequency_3 == 1, LOG);

}


void StatisticsTest::test_minimum()
{
   cout << "test_calculate_minimum\n";

   // Test 0
   Tensor<type, 1> vector_0;

   assert_true(::isnan(minimum(vector_0)), LOG);

   // Test 1
   Tensor<type, 1> vector(3);
   vector.setValues({0.0, 1.0, 9.0});

   assert_true(minimum(vector) - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);

   // Test 2
   Tensor<type, 1> vector_1(3);
   vector_1.setValues({1,2,3});

   Tensor<type, 1> vector_2(3);
   vector_2.setValues({-1,2,3});

   assert_true(minimum(vector_1) - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(minimum(vector_2) - static_cast<type>(-1.0) < static_cast<type>(1.0e-6), LOG);

}


void StatisticsTest::test_maximum()
{
   cout << "test_calculate_maximum\n";

   // Test 0
   Tensor<type, 1> vector_0;

   assert_true(::isnan(maximum(vector_0)), LOG);

   // Test 1
   Tensor<type, 1> vector(3);
   vector.setValues({0.0, 1.0, 9.0});

   assert_true(maximum(vector) - static_cast<type>(9.0) < static_cast<type>(1.0e-6), LOG);

   // Test 2
   Tensor<type, 1> vector_1(3);
   vector_1.setValues({1,2,3});

   Tensor<type, 1> vector_2(3);
   vector_2.setValues({-1,-2,-3});

   assert_true(maximum(vector_1) - static_cast<type>(3.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(maximum(vector_2) - static_cast<type>(-1.0) < static_cast<type>(1.0e-6), LOG);

}


void StatisticsTest::test_mean()
{
   cout << "test_mean\n";

   // Test 0
   Tensor<type, 2> matrix(3,3);
   matrix.setZero();
   assert_true(mean(matrix)(0) - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);


   // Test 1
   Tensor<type, 2> matrix_1(3,3);
   matrix_1.setValues({{0,1,-2},{0,1,8},{0,1,6}});

   assert_true(mean(matrix_1)(0) - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);
   assert_true(mean(matrix_1)(1) - static_cast<type>(1) < static_cast<type>(1.0e-6), LOG);
   assert_true(mean(matrix_1)(2) - static_cast<type>(4) < static_cast<type>(1.0e-6), LOG);

   // Test 2
   Tensor<type, 1> vector(2);
   vector.setValues({1, 1.0});

   assert_true(mean(vector) - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);

   vector.resize(2);
   vector[0] = -1.0;
   vector[1] = 1.0;
   assert_true(mean(vector) - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);

   // Test missing values
   Tensor<type, 1> vector1(5);
   Tensor<type, 1> vector2(4);

   vector1.setValues({1.0, NAN, 2.0, 3.0, 4.0});
   vector2.setValues({1.0, 2.0, 3.0, 4.0});

   assert_true(abs(mean(vector2) - mean(vector1)) < static_cast<type>(1.0e-6), LOG);

   Tensor<type, 1> vector3;
   vector3.resize(4);
   vector3[0] = 1;
   vector3[1] = 1;
   vector3[2] = static_cast<type>(NAN);
   vector3[3] = 1;

   assert_true(mean(vector3) - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);

   // Test empty matrix
   Tensor<type, 2> matrix2;

   assert_true(::isnan(mean(matrix2,2)), LOG);

}


void StatisticsTest::test_standard_deviation()
{
   cout << "test_standard_deviation\n";

   // Test 0
//   Tensor<type, 1> vector_0;
//   assert_true(standard_deviation(vector_0) - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);


   // Test 1
   Tensor<type, 1> vector_1(4);
   vector_1.setValues({2,4,8,10});

   Tensor<type, 1> vector_2(4);
   vector_2.setValues({-11,-11,-11,-11});

   assert_true(standard_deviation(vector_1) - sqrt(static_cast<type>(40)/static_cast<type>(3)) < static_cast<type>(1.0e-6), LOG);
   assert_true(standard_deviation(vector_2) - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);


   // Test 2
   Tensor<type, 1> vector_3(3);
   vector_3.setZero();

   assert_true(static_cast<Index>(standard_deviation(vector_3)) - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);


   // Test 3
   Tensor<type, 1> vector(2);
   type standard_deviation;
   vector.setValues({1, 1.0});

   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(abs(static_cast<Index>(standard_deviation) - static_cast<type>(0)) < static_cast<type>(1.0e-6), LOG);

   // Test 4
   vector.resize(2);
   vector[0] = -1.0;
   vector[1] = 1.0;

   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(abs(standard_deviation- sqrt(static_cast<type>(2))) < static_cast<type>(1.0e-6), LOG);

   // Test missing values

   Tensor<type, 1> vector_4;
   Tensor<type, 1> vector_5;
   type standard_deviation_missing_values;

   // Test 5

   vector_4.resize(1);
   vector_4[0] = static_cast<type>(NAN);

   standard_deviation_missing_values = OpenNN::standard_deviation(vector_4);

   assert_true(standard_deviation_missing_values - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);

   // Test 6
   vector_4.resize(5);
   vector_4[0] = 1.0;
   vector_4[1] = static_cast<type>(NAN);
   vector_4[2] = 2.0;
   vector_4[3] = 3.0;
   vector_4[4] = 4.0;

   vector_5.resize(4);
   vector_5[0] = 1.0;
   vector_5[1] = 2.0;
   vector_5[2] = 3.0;
   vector_5[3] = 4.0;

   assert_true(abs(OpenNN::standard_deviation(vector_4) - OpenNN::standard_deviation(vector_5)) < static_cast<type>(1.0e-6) , LOG);

}


/// @todo <--- Zero  //<--- Missing values

void StatisticsTest::test_median()
{
    cout << "test_calculate_median\n";

    // Test 0

//    Tensor<type, 1> vector_0;
//    assert_true(median(vector_0) == 0, LOG);

    // Test 1 , 2
    Tensor<type, 1> vector_1(4);
    vector_1.setValues({2,4,8,10});

    Tensor<type, 1> vector_2(4);
    vector_2.setValues({-11,-11,-11,-11});

    assert_true(median(vector_1) - static_cast<type>(6) < static_cast<type>(1.0e-3), LOG);
    assert_true(median(vector_2) - static_cast<type>(-11) < static_cast<type>(1.0e-3), LOG);

    // Test 3
    Tensor<type, 1> vector(4);
    vector.resize(4);
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;

    assert_true(abs(median(vector) - static_cast<type>(2.5)) < static_cast<type>(1.0e-3), LOG);

    // Test 4
    vector.resize(5);
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    vector[4] = 5.0;

    assert_true(abs(median(vector) - static_cast<type>(3)) < static_cast<type>(1.0e-3), LOG);

    // Test 5
    Tensor<type, 2> matrix(3,2);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 1.0;
    matrix(1, 0) = 2.0;
    matrix(1, 1) = 3.0;
    matrix(2, 0) = 3.0;
    matrix(2, 1) = 4.0;

    assert_true(abs(median(matrix)(0) - static_cast<type>(2)) < static_cast<type>(1.0e-3), LOG);
    assert_true(abs(median(matrix)(1) - static_cast<type>(3)) < static_cast<type>(1.0e-3), LOG);

    // Test median missing values matrix
    Tensor<type, 2> matrix_2(3,2);
    matrix_2(0, 0) = 1.0;
    matrix_2(0, 1) = 1.0;
    matrix_2(1, 0) = static_cast<type>(NAN);
    matrix_2(1, 1) = static_cast<type>(NAN);
    matrix_2(2, 0) = 3.0;
    matrix_2(2, 1) = 3.0;

    assert_true(abs(median(matrix_2)(0) - static_cast<type>(2)) < static_cast<type>(1.0e-3), LOG);
    assert_true(abs(median(matrix_2)(1) - static_cast<type>(2)) < static_cast<type>(1.0e-3), LOG);

    // Test median missing values vector
    Tensor<type, 1> vector_3;
    vector_3.resize(4);
    vector_3[0] = 3.0;
    vector_3[1] = static_cast<type>(NAN);
    vector_3[2] = 1.0;
    vector_3[3] = static_cast<type>(NAN);

    assert_true(median(vector_3) - static_cast<type>(2) < static_cast<type>(1.0e-3), LOG);

}


void StatisticsTest::test_variance()
{
    cout << "test_variance\n";

    // Test 0

    Tensor<type, 1> vector_0(3);
    vector_0.setZero();
    assert_true(static_cast<Index>(variance(vector_0)) == 0, LOG);


    // Test 1 , 2
    Tensor<type, 1> vector_1(4);
    vector_1.setValues({2,4,8,10});

    Tensor<type, 1> vector_2(4);
    vector_2.setValues({-11,-11,-11,-11});

    assert_true(variance(vector_1) - static_cast<type>(40)/static_cast<type>(3) < static_cast<type>(1.0e-6), LOG);
    assert_true(variance(vector_2) - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);

    // Test 3
    Tensor<type, 1> vector(1);
    vector.resize(1);
    vector[0] = 1;

    assert_true(abs(variance(vector) - static_cast<type>(0.0)) < static_cast<type>(1.0e-6), LOG);

    // Test 4
    vector.resize(3);
    vector[0] = 2.0;
    vector[1] = 1.0;
    vector[2] = 2.0;
    assert_true(abs(variance(vector) - static_cast<type>(1)/static_cast<type>(3)) < static_cast<type>(1.0e-6), LOG);

    // Test variance missing values
    Tensor<type, 1> vector_3;
    Tensor<type, 1> vector_4;

    vector_3.resize(3);
    vector_3[0] = 1.0;
    vector_3[1] = static_cast<type>(NAN);
    vector_3[2] = 2.0;

    vector_4.resize(2);
    vector_4[0] = 1.0;
    vector_4[1] = 2.0;

    assert_true(abs(variance(vector_3) - variance(vector_4)) < static_cast<type>(1.0e-6), LOG);
    assert_true(abs(variance(vector_3) - static_cast<type>(0.5)) < static_cast<type>(1.0e-6), LOG);
}


void StatisticsTest::test_calculate_asymmetry()
{
    cout << "test_calculate_asymmetry\n";

    // Test 0
    Tensor<type, 1> vector_0(3);
    vector_0.setZero();

    assert_true(asymmetry(vector_0) - static_cast<Index>(0) < static_cast<type>(1.0e-3), LOG);

    // Test 1
    Tensor<type, 1> vector;
    vector.resize(4);
    vector[0] = 1.0;
    vector[0] = 5.0;
    vector[0] = 3.0;
    vector[0] = 9.0;

    type asymmetry_value = OpenNN::asymmetry(vector);

    assert_true(asymmetry_value - static_cast<type>(0.75) < static_cast<type>(1.0e-3), LOG);

    // Test 2
    Tensor<type, 1> vector_1(4);
    vector_1.setValues({1,5,3,9});

    assert_true(asymmetry(vector_1) - static_cast<Index>(0.75) < static_cast<type>(1.0e-3), LOG);

    // Test missing values
    Tensor<type, 1> vector_2;
    Tensor<type, 1> vector_missing_values;

    vector_2.resize(4);
    vector_2[0] = 1.0;
    vector_2[1] = 5.0;
    vector_2[2] = 3.0;
    vector_2[3] = 9.0;

    vector_missing_values.resize(5);
    vector_missing_values[0] = 1.0;
    vector_missing_values[1] = 5.0;
    vector_missing_values[2] = static_cast<type>(NAN);
    vector_missing_values[3] = 3.0;
    vector_missing_values[4] = 9.0;

    type asymmetry = OpenNN::asymmetry(vector_2);
    type asymmetry_missing_values = OpenNN::asymmetry(vector_missing_values);

    assert_true(abs(asymmetry - asymmetry_missing_values) < static_cast<type>(1.0e-3), LOG);

}


void StatisticsTest::test_calculate_kurtosis()
{
    cout << "test_calculate_kurtosis\n";

    // Test 1
    Tensor<type, 1> vector(4);
    vector.setValues({1,5,3,9});

    assert_true(abs(kurtosis(vector) - static_cast<type>(-1.9617)) < static_cast<type>(1.0e-3), LOG);

    // Test missing values

    Tensor<type, 1> vector_0;
    Tensor<type, 1> vector_missing_values;

    vector_0.resize(4);
    vector_0[0] = 1.0;
    vector_0[1] = 5.0;
    vector_0[2] = 3.0;
    vector_0[3] = 9.0;

    vector_missing_values.resize(5);
    vector_missing_values[0] = 1.0;
    vector_missing_values[1] = 5.0;
    vector_missing_values[2] = static_cast<type>(NAN);
    vector_missing_values[3] = 3.0;
    vector_missing_values[4] = 9.0;

    type kurtosis = OpenNN::kurtosis(vector_0);
    type kurtosis_missing_values = OpenNN::kurtosis(vector_missing_values);

    assert_true(abs(kurtosis - kurtosis_missing_values) < static_cast<type>(1.0e-3), LOG);
}


void StatisticsTest::test_quartiles()
{
   cout << "test_quartiles\n";

   // Test 1
   Tensor<type, 1> vector(1);
   vector.setZero();

   Tensor<type, 1> quartiles = OpenNN::quartiles(vector);

   assert_true(static_cast<Index>(quartiles(0)) - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);
   assert_true(static_cast<Index>(quartiles(1)) - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);
   assert_true(static_cast<Index>(quartiles(2)) - static_cast<type>(0) < static_cast<type>(1.0e-6), LOG);

   // Test 2
   vector.resize(2);
   vector.setValues({0,1});

   Tensor<type, 1> quartiles1 = OpenNN::quartiles(vector);

   assert_true(quartiles1(0) - static_cast<type>(0.25) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles1(1) - static_cast<type>(0.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles1(2) - static_cast<type>(0.75) < static_cast<type>(1.0e-6), LOG);

   // Test 3
   vector.resize(3);
   vector.setValues({0,1,2});

   Tensor<type, 1> quartiles2 = OpenNN::quartiles(vector);

   assert_true(quartiles2(0) - static_cast<type>(0.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles2(1) - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles2(2) - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);

   // Test 4
   vector.resize(4);
   vector.setValues({0,1,2,3});

   Tensor<type, 1> quartiles3 = OpenNN::quartiles(vector);

   assert_true(quartiles3(0) - static_cast<type>(0.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles3(1) - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles3(2) - static_cast<type>(2.5) < static_cast<type>(1.0e-6), LOG);

   // Test 5
   vector.resize(5);
   vector.setValues({0,1,2,3,4});

   Tensor<type, 1> quartiles4 = OpenNN::quartiles(vector);

   assert_true(quartiles4[0] - static_cast<type>(0.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles4[1] - static_cast<type>(2.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles4[2] - static_cast<type>(3.5) < static_cast<type>(1.0e-6), LOG);

   // Test 6
   vector.resize(6);
   vector.setValues({0,1,2,3,4,5});

   Tensor<type, 1> quartiles5 = OpenNN::quartiles(vector);

   assert_true(quartiles5[0] - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles5[1] - static_cast<type>(2.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles5[2] - static_cast<type>(4.0) < static_cast<type>(1.0e-6), LOG);

   // Test 7
   vector.resize(7);
   vector.setValues({0,1,2,3,4,5,6});

   Tensor<type, 1> quartiles6 = OpenNN::quartiles(vector);

   assert_true(quartiles6[0] - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles6[1] - static_cast<type>(3.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles6[2] - static_cast<type>(5.0) < static_cast<type>(1.0e-6), LOG);

   // Test 8
   vector.resize(8);
   vector.setValues({0,1,2,3,4,5,6,7});

   Tensor<type, 1> quartiles7 = OpenNN::quartiles(vector);

   assert_true(quartiles7[0] - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles7[1] - static_cast<type>(3.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles7[2] - static_cast<type>(5.5) < static_cast<type>(1.0e-6), LOG);

   // Test 9
   vector.resize(9);
   vector.setValues({0,1,2,3,4,5,6,7,8});

   Tensor<type, 1> quartiles8 = OpenNN::quartiles(vector);

   assert_true(quartiles8[0] - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles8[1] - static_cast<type>(4.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles8[2] - static_cast<type>(6.5) < static_cast<type>(1.0e-6), LOG);

   // Test 10
   vector.resize(9);
   vector.setValues({1,4,6,2,0,3,4,7,10});

   Tensor<type, 1> quartiles9 = OpenNN::quartiles(vector);

   assert_true(quartiles9[0] - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles9[1] - static_cast<type>(4.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles9[2] - static_cast<type>(6.5) < static_cast<type>(1.0e-6), LOG);

   // Test 11
   vector.resize(20);
   vector.setValues({12,14,50,76,12,34,56,74,89,60,96,24,53,25,67,84,92,45,62,86});

   Tensor<type, 1> quartiles10 = OpenNN::quartiles(vector);

   assert_true(quartiles10[0] - static_cast<type>(29.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles10[1] - static_cast<type>(58.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles10[2] - static_cast<type>(80.0) < static_cast<type>(1.0e-6), LOG);

   // Test missing values:

   // Test 1
   Tensor<type, 1> vector_m;
   vector_m.resize(5);
   vector_m[0] = 1;
   vector_m[1] = 2;
   vector_m[2] = 3;
   vector_m[3] = static_cast<type>(NAN);
   vector_m[4] = 4;

   Tensor<type, 1> quartiles_missing_values = OpenNN::quartiles(vector_m);

   assert_true(quartiles_missing_values(0) - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles_missing_values(1) - static_cast<type>(2.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles_missing_values(2) - static_cast<type>(3.5) < static_cast<type>(1.0e-6), LOG);

   // Test 2
   vector_m.resize(6);
   vector_m[0] = 1;
   vector_m[1] = 2;
   vector_m[2] = 3;
   vector_m[3] = static_cast<type>(NAN);
   vector_m[4] = 4;
   vector_m[5] = 5;

   Tensor<type, 1> quartiles_missing_values2 = OpenNN::quartiles(vector_m);

   assert_true(quartiles_missing_values2(0) - static_cast<type>(1.5) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles_missing_values2(1) - static_cast<type>(3.0) < static_cast<type>(1.0e-6), LOG);
   assert_true(quartiles_missing_values2(2) - static_cast<type>(4.5) < static_cast<type>(1.0e-6), LOG);

}



/// @todo ERROR frequencies

void StatisticsTest::test_histogram()
{
   cout << "test_histogram\n";

   Tensor<type, 1> vector;

   Tensor<type, 1> centers;
   Tensor<Index, 1> frequencies;

   // Test 1

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


   // Test 2
   vector.resize(20);
   vector.setRandom();

//   Histogram histogram_2(vector, 10);

//   centers_2 = histogram_2.centers;
//   frequencies_2 = histogram_2.frequencies;

//   Tensor<Index, 0> sum_frec_2;
//   sum_frec_2 = frequencies_2.sum();
   //assert_true(sum_frec_2(0) == 20, LOG); // <--- failed

}


void StatisticsTest::test_histograms()
{
    cout << "test_histograms\n";

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

    Tensor<Histogram, 1> histogram(matrix.dimension(1));

    histogram = histograms(matrix, 3);
    Tensor<Index, 1> solution(3);
    solution.setValues({1, 1, 1});

    //assert_true(histogram[0].frequencies == solution, LOG);
    //assert_true(histogram[1].frequencies == solution, LOG);
    //assert_true(histogram[2].frequencies == solution, LOG);
}


void StatisticsTest::test_total_frequencies()   //<--- Check
{
    cout << "test_total_frequencies\n";

    Tensor<Histogram, 1> histograms(3);

    // Test 1
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

   // Test 0
   Tensor<type, 1> vector(0);

   assert_true(minimal_index(vector) == 0, LOG);

   // Test 1
   Tensor<type, 1> vector_1(3);
   vector_1.setValues({1,0,-1});

   assert_true(minimal_index(vector_1) == 2, LOG);
}


void StatisticsTest::test_calculate_maximal_index()
{
   cout << "test_calculate_maximal_index\n";

   // Test 0
   Tensor<type, 1> vector(0);

   assert_true(maximal_index(vector) == 0, LOG);

   // Test 1
   Tensor<type, 1> vector_1(3);
   vector_1.setValues({1,0,-1});

   assert_true(maximal_index(vector_1) == 0, LOG);
}


void StatisticsTest::test_calculate_minimal_indices()
{
    cout << "test_calculate_minimal_indices\n";

    // Test 0
    Tensor<type, 1> vector(0);

    assert_true(minimal_indices(vector,0).dimension(0) == 0, LOG);

    // Test 1
    Tensor<type, 1> vector_1(3);
    vector_1.setValues({-1,0,1});

    assert_true(minimal_indices(vector_1, 1)[0] == 0, LOG);

    assert_true(minimal_indices(vector_1, 3)[0] == 0, LOG);
    assert_true(minimal_indices(vector_1, 3)[1] == 1, LOG);
    assert_true(minimal_indices(vector_1, 3)[2] == 2, LOG);

    // Test 2
    Tensor<type, 1> vector_2(4);
    vector_2.setValues({0,0,0,1});

    assert_true(minimal_indices(vector_2, 4)[0] == 0, LOG);
    assert_true(minimal_indices(vector_2, 4)[1] == 1, LOG);
    assert_true(minimal_indices(vector_2, 4)[3] == 3, LOG);


    // Test 3

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

    // Test 4

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

    // Test 0
    Tensor<type, 1> vector(0);

    assert_true(maximal_indices(vector,0).dimension(0) == 0, LOG);

    // Test 1
    Tensor<type, 1> vector_1(3);
    vector_1.setValues({-1,0,1});

    assert_true(maximal_indices(vector_1, 1)[0] == 2, LOG);

    // Test 2
    Tensor<type, 1> vector_2(4);
    vector_2.setValues({1,1,1,1});

    assert_true(maximal_indices(vector_2, 4)[0] == 0, LOG);
    assert_true(maximal_indices(vector_2, 4)[1] == 1, LOG);
    assert_true(maximal_indices(vector_2, 4)[3] == 3, LOG);

    // Test 3
    Tensor<type, 1> vector_3(5);
    vector_3.setValues({1,5,6,7,2});

    assert_true(maximal_indices(vector_3, 5)[0] == 3, LOG);
    assert_true(maximal_indices(vector_3, 5)[1] == 2, LOG);
    assert_true(maximal_indices(vector_3, 5)[3] == 4, LOG);
}


void StatisticsTest::test_l2_norm()
{
   cout << "test_l2_norm\n";

   // Test 0
   Tensor<type, 1> vector;

//   assert_true(l2_norm(vector) - sqrt(static_cast<type>(0.0)) < static_cast<type>(1.0e-6), LOG);

   // Test 1
   vector.resize(2);
   vector.setConstant(1);

   assert_true(abs(l2_norm(vector) - sqrt(static_cast<type>(2.0))) < static_cast<type>(1.0e-6), LOG);

   // Test 2
   Tensor<type, 1> vector_1(4);
   vector_1.setValues({1,2,3,4});

   assert_true(abs(l2_norm(vector_1) - sqrt(static_cast<type>(30.0))) < static_cast<type>(1.0e-6), LOG);
}


void StatisticsTest::test_box_plot()
{
    cout << "test_box_plot\n";

    // Test 0
    Tensor<type, 1> vector_0;

    BoxPlot boxplot_0 = box_plot(vector_0);

    assert_true(boxplot_0.minimum - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot_0.first_quartile - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot_0.median - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot_0.third_quartile - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot_0.maximum - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);

    // Test 1
    Tensor<type, 1> vector(8);
    vector.setValues({2.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    BoxPlot boxplot = box_plot(vector);

    BoxPlot solution(2.0, 2.5, 5.5, 7.5, 9.0);

    assert_true(boxplot.minimum - solution.minimum < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot.first_quartile - solution.first_quartile < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot.median - solution.median < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot.third_quartile - solution.third_quartile < static_cast<type>(1.0e-6), LOG);
    assert_true(boxplot.maximum - solution.maximum < static_cast<type>(1.0e-6), LOG);

    // Test missing values

    Tensor<type, 1> vector_m(9);
    vector_m.setValues({2.0, 2.0, 3.0, 5.0, 6.0, 7.0, static_cast<type>(NAN), 8.0, 9.0});

    BoxPlot boxplot_m = box_plot(vector_m);

    BoxPlot solution_m(2.0, 2.5, 5.5, 7.5, 9.0);

    assert_true((boxplot_m.minimum - solution_m.minimum) < static_cast<type>(1.0e-6), LOG);
    assert_true((boxplot_m.first_quartile - solution_m.first_quartile) < static_cast<type>(1.0e-6), LOG);
    assert_true((boxplot_m.median - solution_m.median) < static_cast<type>(1.0e-6), LOG);
    assert_true((boxplot_m.third_quartile - solution_m.third_quartile) < static_cast<type>(1.0e-6), LOG);
    assert_true((boxplot_m.maximum - solution_m.maximum) < static_cast<type>(1.0e-6), LOG);

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

    Tensor<type, 1> centers_2;
    Tensor<Index, 1> frequencies_2;

    Tensor<type, 1> vector_2;
    vector_2.resize(4);

    vector_2[0] = 1;
    vector_2[1] = 3;
    vector_2[2] = 2;
    vector_2[3] = 4;

//    graphic_2 = histogram(vector, 4);

//    centers_2 = graphic_2.centers;
//    frequencies_2 = graphic_2.frequencies;

//    assert_true(centers_2 == centers , LOG);
//    assert_true(frequencies_2 == frequencies, LOG );
}


/// @todo <--- EXC(Negative numb)

void StatisticsTest::test_means_binary_columns() //<--- EXC(Negative numb)
{
    cout << "test_means_binary_columns\n";
    // Test 0
    Tensor<type, 2> matrix;

    assert_true(static_cast<Index>(means_binary_column(matrix)(0)) == 0, LOG);

    // Test 1
    Tensor<type, 2> matrix_1(4,2);
    matrix_1.setValues({{0,1},{1,1},{0,0},{1,0}});

    Tensor<type, 1> solution(2);
    solution.setValues({0.5, 0.5});

    assert_true(static_cast<type>(means_binary_column(matrix_1)(0)) - static_cast<type>(solution(0)) < static_cast<type>(1.0e-6), LOG);
    assert_true(static_cast<type>(means_binary_column(matrix_1)(1)) - static_cast<type>(solution(1)) < static_cast<type>(1.0e-6), LOG);

    // Test 2
    Tensor<type, 2> matrix_2(2,4);
    matrix_2(0,0) = 1;
    matrix_2(0,1) = 1;
    matrix_2(0,2) = 1;
    matrix_2(0,3) = 1;
    matrix_2(1,0) = 1;
    matrix_2(1,1) = 1;
    matrix_2(1,2) = 0;
    matrix_2(1,3) = 0;

    Tensor<type, 1> solution_2(2);
    solution_2.setValues({1.0, 1.0});

    assert_true(static_cast<type>(means_binary_column(matrix_2)(0)) - static_cast<type>(solution_2(0)) < static_cast<type>(1.0e-6), LOG);
    assert_true(static_cast<type>(means_binary_column(matrix_2)(1)) - static_cast<type>(solution_2(1)) < static_cast<type>(1.0e-6), LOG);

    // Test 0
    matrix.resize(2,2);
    matrix.setZero();

    assert_true(static_cast<Index>(means_binary_columns(matrix)(0)) == 0, LOG);

    // Test 1
    matrix_1.resize(3,3);
    matrix_1.setValues({{1,0,5},{1,0,1},{0,1,7}});

    solution.resize(2);
    solution.setValues({3, 7});

    assert_true(static_cast<Index>(means_binary_columns(matrix_1)(0)) == static_cast<Index>(solution(0)), LOG);
    assert_true(static_cast<Index>(means_binary_columns(matrix_1)(1)) == static_cast<Index>(solution(1)), LOG);

    // Test 2
    matrix_2.resize(3,3);
    matrix_2.setValues({{1, 0, 7}, {1, 1, 8}, {0, 0, 5}});

    Tensor<type, 1> solution_1(2);
    solution_1.setValues({7.5, 8});

    Tensor<type, 1> means(2);
    means = means_binary_columns(matrix_2);
    assert_true(means(0) - solution_1(0) < static_cast<type>(1.0e-7), LOG);
    assert_true(means(1) - solution_1(1) < static_cast<type>(1.0e-7), LOG);

    // Test missing values
    Tensor<type, 2> matrix_m(3, 4);
    matrix_m(0,0) = 1.0;
    matrix_m(0,1) = 0.0;
    matrix_m(0,2) = 7.0;
    matrix_m(1,0) = 1.0;
    matrix_m(1,1) = 1.0;
    matrix_m(1,2) = 8.0;
    matrix_m(2,0) = 0.0;
    matrix_m(2,1) = 0.0;
    matrix_m(2,2) = 5.0;
    matrix_m(3,0) = 1.0;
    matrix_m(3,0) = 1.0;
    matrix_m(3,0) = static_cast<type>(NAN);

    Tensor<type, 1> solution_m(2);
    solution_m.setValues({7.5, 8});

    assert_true(means_binary_columns(matrix_m)(0) - solution_m[0] < static_cast<type>(1.0e-7), LOG);
    assert_true(means_binary_columns(matrix_m)(1) - solution_m[1] < static_cast<type>(1.0e-7), LOG);
}


void StatisticsTest::test_weighted_mean()
{
    cout << "test_weighted_mean\n";

    // Test 1
    Tensor<type, 1> vector_1(4);
    vector_1.setValues({1,1,1,1});

    Tensor<type, 1> weights_1(4);
    weights_1.setValues({0.25,0.25,0.25,0.25});

    assert_true(static_cast<type>(weighted_mean(vector_1, weights_1)) - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);

    // Test 2
    Tensor<type, 1> vector_2(4);
    vector_2.setValues({2,1,4,1});

    Tensor<type, 1> weights_2(4);
    weights_2.setValues({static_cast<type>(0.2),static_cast<type>(0.3),static_cast<type>(0.4),static_cast<type>(0.1)});

    assert_true(static_cast<type>(weighted_mean(vector_2, weights_2)) - static_cast<type>(2.4) < static_cast<type>(1.0e-6), LOG);
}


void StatisticsTest::test_percentiles()
{
    cout << "test_percentiles\n";

    // Test 0

//    Tensor<type, 1> empty_vector;
//    Tensor<type, 1> percentiles_empty = OpenNN::percentiles(empty_vector);
//    assert_true(::isnan(percentiles_empty[0]), LOG);

    // Test 1

    Tensor<type, 1> vector(10);

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

    // Test 2

    Tensor<type, 1> vector_2(21);

    vector_2.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

    Tensor<type, 1> percentiles_2 = OpenNN::percentiles(vector_2);

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

    // Test 3

    Tensor<type, 1> vector_3(14);

    vector_3.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 19, 32});

    Tensor<type, 1> percentiles_3 = OpenNN::percentiles(vector_3);

    Tensor<type, 1> solution_3(10);
    solution_3.setValues({1, 2, 4, 5, 6.5, 8, 9, 15, 19, 32});

    assert_true((percentiles_3[0] - solution_3[0]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[1] - solution_3[1]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[2] - solution_3[2]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[3] - solution_3[3]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[4] - solution_3[4]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[5] - solution_3[5]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[6] - solution_3[6]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[7] - solution_3[7]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[8] - solution_3[8]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_3[9] - solution_3[9]) < static_cast<type>(1.0e-7), LOG);

    // Test missing values

    Tensor<type, 1> vector_m(21); //@todo

    vector_m.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

    vector_m[20] = static_cast<type>(NAN);

    Tensor<type, 1> percentiles_m = OpenNN::percentiles(vector);

    Tensor<type, 1> solution_m(10);
    solution_m.setValues({2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20});

    assert_true((percentiles_m[0] - solution_m[0]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[1] - solution_m[1]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[2] - solution_m[2]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[3] - solution_m[3]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[4] - solution_m[4]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[5] - solution_m[5]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[6] - solution_m[6]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[7] - solution_m[7]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[8] - solution_m[8]) < static_cast<type>(1.0e-7), LOG);
    assert_true((percentiles_m[9] - solution_m[9]) < static_cast<type>(1.0e-7), LOG);

}


void StatisticsTest::test_means_by_categories()
{
    cout << "test_means_by_categories\n";

    // Test 1

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

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

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
   test_weighted_mean();


   // Mean binary
   test_means_binary_columns();


   // Median
   test_median();


   // Variance
   test_variance();


   // Assymetry
   test_calculate_asymmetry();


   // Kurtosis
   test_calculate_kurtosis();


   // Standard deviation
   test_standard_deviation();


   // Quartiles
   test_quartiles();


   // Box plot
   test_box_plot();


   // Descriptives struct


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


   // Norm
   test_l2_norm();


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
