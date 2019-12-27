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
   descriptives.set_minimum(5.0);

   assert_true(descriptives.minimum == 5.0, LOG);
}


void StatisticsTest::test_set_maximum()
{
   cout << "test_set_maximun\n";

   Descriptives descriptives;
   descriptives.set_maximum(5.0);

   assert_true(descriptives.maximum == 5.0, LOG);
}


void StatisticsTest::test_set_mean()
{
   cout << "test_set_mean\n";

   Descriptives descriptives;
   descriptives.set_mean(5.0);

   assert_true(descriptives.mean == 5.0, LOG);
}


void StatisticsTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   Descriptives descriptives;
   descriptives.set_standard_deviation(3.0);

   assert_true(descriptives.standard_deviation == 3.0, LOG);
}


void StatisticsTest::test_has_mean_zero_standard_deviation_one()
{
    cout << "test_set_standard_deviation\n";

    Descriptives descriptives(-4.0 ,5.0 ,0.0 ,1.0);

    assert_true(descriptives.has_mean_zero_standard_deviation_one(), LOG);
}


void StatisticsTest::test_has_minimum_minus_one_maximum_one()
{
    cout << "test_set_has_minimum_minus_one_maximum_one\n";

    Descriptives descriptives(-1.0 ,1.0 ,3.0 ,2.0);

    assert_true(descriptives.has_minimum_minus_one_maximum_one(), LOG);
}


void StatisticsTest::test_constructor()
{
   cout << "test_constructor\n";
}


void StatisticsTest::test_destructor()
{  
   cout << "test_destructor\n";
}


void StatisticsTest::test_get_bins_number()
{
    cout << "test_get_bins_number\n";

    Histogram histogram;

    assert_true(histogram.get_bins_number() == 0, LOG);

}


void StatisticsTest::test_count_empty_bins()
{
    cout << "test_count_empty_bins\n";

    Histogram histogram;

    assert_true(histogram.count_empty_bins() == 0, LOG);
}


void StatisticsTest::test_calculate_minimum_frequency()
{
    cout << "test_calculate_minimun_frecuency\n";

    Vector<double> values({1, 2, 3, 4, 5});

    Histogram histogram;

    histogram = OpenNN::histogram(values, 5);

    assert_true(histogram.calculate_minimum_frequency() == 1, LOG);
}


void StatisticsTest::test_calculate_maximum_frequency()
{
    cout << "test_calculate_maximum_frequency\n";

    Vector<double> values({1, 2, 3, 4, 5});

    Histogram histogram;

    histogram = OpenNN::histogram(values, 5);

    assert_true(histogram.calculate_maximum_frequency() == 1, LOG);
}


void StatisticsTest::test_calculate_most_populated_bin()
{
    cout << "test_calculate_most_populated_bin\n";

    Vector<double> values({1, 2, 3, 4, 5, 5, 5, 5});

    Histogram histogram;

    histogram = OpenNN::histogram(values, 5);

    assert_true(histogram.calculate_most_populated_bin() == 4, LOG);
}


void StatisticsTest::test_calculate_minimal_centers()
{
    cout << "test_calculate_minimal_centers\n";

    Histogram histogram;

    Vector<double> vector({1, 1, 1, 1, 1, 2, 2, 6, 4, 8, 1, 4, 7});

    histogram = OpenNN::histogram(vector);

    Vector<double> solution({2.75, 3.45, 4.85, 5.55});

    assert_true((histogram.calculate_minimal_centers()[0] - solution[0]) < 1.0e-7, LOG);
    assert_true((histogram.calculate_minimal_centers()[1] - solution[1]) < 1.0e-7, LOG);
    assert_true((histogram.calculate_minimal_centers()[2] - solution[2]) < 1.0e-7, LOG);
    assert_true((histogram.calculate_minimal_centers()[3] - solution[3]) < 1.0e-7, LOG);

}


void StatisticsTest::test_calculate_maximal_centers()
{
    cout << "test_calculate_maximal_centers\n";

    Histogram histogram;

    Vector<double> vector({1, 1, 1, 1, 1, 2, 2, 6, 4, 8, 8, 8, 1, 4, 7});

    histogram = OpenNN::histogram(vector);

    Vector<double> solution({2.75, 3.45, 4.85, 5.55});

    assert_true((histogram.calculate_minimal_centers()[0] - 1.35) < 1.0e-7, LOG);
}


void StatisticsTest::test_calculate_bin()
{
    cout << "test_calculate_bin\n";

    Vector<double> vector;
    size_t bin;
    Histogram histogram;
    vector.set(1.0, 1.0, 11.0);
    histogram = OpenNN::histogram(vector, 10);

    // Test
    bin = histogram.calculate_bin(vector[0]);
    assert_true(bin == 0, LOG);

    // Test
    bin = histogram.calculate_bin(vector[1]);
    assert_true(bin == 1, LOG);

    // Test
    bin = histogram.calculate_bin(vector[2]);
    assert_true(bin == 2, LOG);
}


void StatisticsTest::test_calculate_frequency()
{
    cout << "test_calculate_frequency\n";

    Vector<double> vector;
    size_t frequency;
    Histogram histogram;

    // Test
    vector.set(0.0, 1.0, 9.0);
    histogram = OpenNN::histogram(vector, 10);
    frequency = histogram.calculate_frequency(vector[9]);

    assert_true(frequency == 1, LOG);
}


void StatisticsTest::test_minimum()
{

   cout << "test_calculate_minimum\n";

   Vector<double> vector(3);
   vector[0] = -1;
   vector[1] = 0;
   vector[2] = 1;

   Vector<double> minimun(1);

   assert_true(abs(minimum(vector) - (-1)) < 1.0e-3, LOG);
}


void StatisticsTest::test_minimum_missing_values()
{
   cout << "test_minimum_missing_values\n";

   Vector<double> vector;
   Vector<size_t> missing_values;
   double minimum;

   // Test
   vector.set(1, 1.0);
   minimum = minimum_missing_values(vector);

   assert_true(minimum == 1.0, LOG);

   vector.set(3);
   vector[0] = -1;
   vector[1] = 0;
   vector[2] = 1;

   minimum = minimum_missing_values(vector);

   assert_true(abs(minimum -(-1.0)) < 1.0e-3, LOG);

   // Test missing values
   vector.set(3);
   vector[0] = -1;
   vector[1] = static_cast<double>(NAN);
   vector[2] = 1;

   assert_true(abs(minimum_missing_values(vector) - (-1.0)) < 1.0e-3, LOG);
}


void StatisticsTest::test_maximum_missing_values()
{
   cout << "test_maximum_missing_values\n";

   Vector<double> vector;
   double maximum;

   //Test
   vector.set(1, 1);
   maximum = maximum_missing_values(vector);

   assert_true(maximum == 1.0, LOG);

   //Test
   vector.set(3);
   vector[0] = -1;
   vector[1] = 0;
   vector[2] = 1;

   maximum = maximum_missing_values(vector);

   assert_true(abs(maximum - 1.0) < 1.0e-3, LOG);

   //Test maximum missing values
   vector.set(3);
   vector[0] = -1.0;
   vector[1] = static_cast<double>(NAN);
   vector[2] = 1.0;

   assert_true(abs(maximum_missing_values(vector) - 1.0) < 1.0e-3 , LOG);
}


void StatisticsTest::test_calculate_mean()
{
   cout << "test_calculate_mean\n";

   Vector<double> vector(1, 1.0);

   assert_true(mean(vector) == 1.0, LOG);

   vector.set(2);
   vector[0] = -1.0;
   vector[1] = 1.0;

   assert_true(mean(vector) == 0.0, LOG);

   // Test missing values
   Vector<double> vector1;
   Vector<double> vector2;

   vector1.set(5);
   vector1[0] = 1.0;
   vector1[1] = static_cast<double>NAN;
   vector1[2] = 2.0;
   vector1[3] = 3.0;
   vector1[4] = 4.0;

   vector2.set(4);
   vector2[0] = 1.0;
   vector2[1] = 2.0;
   vector2[2] = 3.0;
   vector2[3] = 4.0;

   assert_true(abs(mean(vector2) - mean_missing_values(vector1)) < 1.0e-3, LOG);
}


void StatisticsTest::test_standard_deviation()
{
   cout << "test_standard_deviation\n";

   Vector<double> vector;

   double standard_deviation;

   // Test
   vector.set(1, 1.0);
   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(abs(standard_deviation - 0.0) < 1.0e-3, LOG);

   // Test
   vector.set(2);
   vector[0] = -1.0;
   vector[1] = 1.0;

   standard_deviation = OpenNN::standard_deviation(vector);

   assert_true(abs(standard_deviation-1.4142) < 1.0e-3, LOG);
}


void StatisticsTest::test_calculate_median()
{
    cout << "test_calculate_median\n";

    Vector<double> vector;
    vector.set(4);
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;

    assert_true(abs(median(vector) - 2.5) < 1.0e-3, LOG);

    vector.set(5);
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    vector[4] = 5.0;

    assert_true(abs(median(vector) - 3.0) < 1.0e-3, LOG);
}


void StatisticsTest::test_calculate_median_missing_values()
{
    cout << "test_calculate_median_missing_values\n";

    Matrix<double> matrix;
    matrix.set(3,2);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 1.0;
    matrix(1, 0) = static_cast<double>(NAN);
    matrix(1, 1) = static_cast<double>(NAN);
    matrix(2, 0) = 3.0;
    matrix(2, 1) = 3.0;

    Vector<double> solution({2.0, 2.0});
    Vector<double> vector;
    vector.set(4);
    vector[0] = 3.0;
    vector[1] = static_cast<double>(NAN);
    vector[2] = 1.0;
    vector[3] = static_cast<double>(NAN);

    //Test median missing values vector
    assert_true(abs(median_missing_values(vector) - 2.0) < 1.0e-3, LOG);
    assert_true(median_missing_values(matrix) == solution, LOG);
}


void  StatisticsTest::test_standard_deviation_missing_values()
{
    cout << "test_standard_deviation_missing_values\n";

    Vector<double> vector;
    Vector<double> vector_1;
    double standard_deviation_missing_values;

    //Test
    vector.set(1);
    vector[0] = static_cast<double>(NAN);

    standard_deviation_missing_values = OpenNN::standard_deviation_missing_values(vector);

    assert_true(standard_deviation_missing_values == 0.0, LOG);

    // Test missing values
    vector.set(5);
    vector[0] = 1.0;
    vector[1] = static_cast<double>(NAN);
    vector[2] = 2.0;
    vector[3] = 3.0;
    vector[4] = 4.0;

    vector_1.set(4);
    vector_1[0] = 1.0;
    vector_1[1] = 2.0;
    vector_1[2] = 3.0;
    vector_1[3] = 4.0;

    assert_true(abs(OpenNN::standard_deviation(vector_1) - OpenNN::standard_deviation_missing_values(vector)) < 1.0e-3 , LOG);
 }


void StatisticsTest::test_variance()
{
    cout << "test_variance\n";
    Vector<double> vector;
    vector.set(1);
    vector[0] = 1;

    assert_true(abs(variance(vector) - 0.0) < 1.0e-3, LOG);

    vector.set(3);
    vector[0] = 2.0;
    vector[1] = 1.0;
    vector[2] = 2.0;
    assert_true(abs(variance(vector) - 0.333333333) < 1.0e-6, LOG);
}


void StatisticsTest::test_calculate_variance_missing_values()
{
    cout << "test_calculate_variance_missing_values";

    // Test variance missing values
    Vector<double> vector;
    Vector<double> vector_1;

    vector.set(3);
    vector[0] = 1.0;
    vector[1] = static_cast<double>(NAN);
    vector[2] = 2.0;

    vector_1.set(2);
    vector_1[0] = 1.0;
    vector_1[1] = 2.0;

    assert_true((variance_missing_values(vector) - variance(vector_1)) < 1.0e-3, LOG);
    assert_true(abs(variance_missing_values(vector_1) - 0.5) < 1.0e-3, LOG);
}


void StatisticsTest::test_calculate_asymmetry()
{
    cout << "test_calculate_asymmetry\n";

    Vector<double> vector;
    vector.set(4);
    vector[0] = 1.0;
    vector[0] = 5.0;
    vector[0] = 3.0;
    vector[0] = 9.0;

    double asymmetry = OpenNN::asymmetry(vector);

    assert_true((asymmetry - 0.75) < 1.0e-3, LOG);
}


void StatisticsTest::test_calculate_asymmetry_missing_values()
{
    cout << "test_calculate_asymmetry_missing_values\n";

    Vector<double> vector;
    Vector<double> vector_missing_values;

    vector.set(4);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = 3.0;
    vector[3] = 9.0;

    vector_missing_values.set(5);
    vector_missing_values[0] = 1.0;
    vector_missing_values[1] = 5.0;
    vector_missing_values[2] = static_cast<double>(NAN);
    vector_missing_values[3] = 3.0;
    vector_missing_values[4] = 9.0;

    double asymmetry = OpenNN::asymmetry(vector);
    double asymmetry_missing_values = OpenNN::asymmetry_missing_values(vector_missing_values);

    assert_true(abs(asymmetry - asymmetry_missing_values) < 1.0e-3, LOG);
}


void StatisticsTest::test_calculate_kurtosis()
{
    cout << "test_calculate_kurtosis\n";

    Vector<double> vector;

    vector.set(4);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = 3.0;
    vector[3] = 9.0;

    double kurtosis = OpenNN::kurtosis(vector);

    assert_true(abs(kurtosis - (-1.9617)) < 1.0e-3, LOG);
}


void StatisticsTest::test_calculate_kurtosis_missing_values()
{
    cout << "test_calculate_kurtosis_missing_values\n";

    Vector<double> vector;
    Vector<double> vector_missing_values;

    vector.set(4);
    vector[0] = 1.0;
    vector[1] = 5.0;
    vector[2] = 3.0;
    vector[3] = 9.0;

    vector_missing_values.set(5);
    vector_missing_values[0] = 1.0;
    vector_missing_values[1] = 5.0;
    vector_missing_values[2] = static_cast<double>(NAN);
    vector_missing_values[3] = 3.0;
    vector_missing_values[4] = 9.0;

    double kurtosis = OpenNN::kurtosis(vector);
    double kurtosis_missing_values = OpenNN:: kurtosis_missing_values(vector_missing_values);

    assert_true(abs(kurtosis - kurtosis_missing_values) < 1.0e-3, LOG);
}


void StatisticsTest::test_quartiles()
{
   cout << "test_quartiles\n";

   Vector<double> vector;
   vector.set(1);
   vector[0] = 0.0;

   Vector<double> quartiles = OpenNN::quartiles(vector);

   assert_true(quartiles[0] == 0.0, LOG);
   assert_true(quartiles[1] == 0.0, LOG);
   assert_true(quartiles[2] == 0.0, LOG);

   vector.set(2);
   vector[0] = 0.0;
   vector[1] = 1.0;

   Vector<double> quartiles1 = OpenNN::quartiles(vector);

   assert_true(quartiles1[0] == 0.25, LOG);
   assert_true(quartiles1[1] == 0.5, LOG);
   assert_true(quartiles1[2] == 0.75, LOG);

   vector.set(3);
   vector[0] = 0.0;
   vector[1] = 1.0;
   vector[2] = 2.0;

   Vector<double> quartiles2 = OpenNN::quartiles(vector);

   assert_true(quartiles2[0] == 0.5, LOG);
   assert_true(quartiles2[1] == 1.0, LOG);
   assert_true(quartiles2[2] == 1.5, LOG);

   vector.set(4);
   vector[0] = 0.0;
   vector[1] = 1.0;
   vector[2] = 2.0;
   vector[3] = 3.0;

   Vector<double> quartiles3 = OpenNN::quartiles(vector);

   assert_true(vector.count_less_equal_to(quartiles3[0])*100.0/vector.size() == 25.0, LOG);
   assert_true(vector.count_less_equal_to(quartiles3[1])*100.0/vector.size() == 50.0, LOG);
   assert_true(vector.count_less_equal_to(quartiles3[2])*100.0/vector.size() == 75.0, LOG);

   vector.set(5);
   vector[0] = 0.0;
   vector[1] = 1.0;
   vector[2] = 2.0;
   vector[3] = 3.0;
   vector[4] = 4.0;

   Vector<double> quartiles4 = OpenNN::quartiles(vector);

   assert_true(quartiles4[0] == 1.0, LOG);
   assert_true(quartiles4[1] == 2.0, LOG);
   assert_true(quartiles4[2] == 3.0, LOG);

   vector.set(6);
   vector[0] = 0.0;
   vector[1] = 1.0;
   vector[2] = 2.0;
   vector[3] = 3.0;
   vector[4] = 4.0;
   vector[5] = 5.0;

   Vector<double> quartiles5 = OpenNN::quartiles(vector);

   assert_true(quartiles5[0] == 1.0, LOG);
   assert_true(quartiles5[1] == 2.5, LOG);
   assert_true(quartiles5[2] == 4.0, LOG);

}


void StatisticsTest::test_calculate_histogram()
{
   cout << "test_calculate_histogram\n";

   Vector<double> vector;
   Vector<double> centers;
   Vector<size_t> frequencies;
   Histogram histogram;

   // Test
   vector.set(0.0, 1.0, 9.0);

   histogram = OpenNN::histogram(vector, 10);

   assert_true(histogram.get_bins_number() == 10, LOG);

   centers = histogram.centers;
   frequencies = histogram.frequencies;

   assert_true(abs(centers[0] - 0.45) < 1.0e-12, LOG);
   assert_true(abs(centers[1] - 1.35) < 1.0e-12, LOG);
   assert_true(abs(centers[2] - 2.25) < 1.0e-12, LOG);
   assert_true(abs(centers[3] - 3.15) < 1.0e-12, LOG);
   assert_true(abs(centers[4] - 4.05) < 1.0e-12, LOG);
   assert_true(abs(centers[5] - 4.95) < 1.0e-12, LOG);
   assert_true(abs(centers[6] - 5.85) < 1.0e-12, LOG);
   assert_true(abs(centers[7] - 6.75) < 1.0e-12, LOG);
   assert_true(abs(centers[8] - 7.65) < 1.0e-12, LOG);
   assert_true(abs(centers[9] - 8.55) < 1.0e-12, LOG);

   assert_true(frequencies[0] == 1, LOG);
   assert_true(frequencies[1] == 1, LOG);
   assert_true(frequencies[2] == 1, LOG);
   assert_true(frequencies[3] == 1, LOG);
   assert_true(frequencies[4] == 1, LOG);
   assert_true(frequencies[5] == 1, LOG);
   assert_true(frequencies[6] == 1, LOG);
   assert_true(frequencies[7] == 1, LOG);
   assert_true(frequencies[8] == 1, LOG);
   assert_true(frequencies[9] == 1, LOG);
   assert_true(histogram.frequencies.calculate_sum() == 10, LOG);

   // Test
   vector.set(20);
   vector.randomize_normal();

   histogram = OpenNN::histogram(vector, 10);

   assert_true(histogram.frequencies.calculate_sum() == 20, LOG);
}


void StatisticsTest::test_calculate_histograms()
{
    cout << "test_calculate_histograms\n";

    Matrix<double> matrix(3,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 1.0;
    matrix(0,2) = 1.0;
    matrix(1,0) = 2.0;
    matrix(1,1) = 2.0;
    matrix(1,2) = 2.0;
    matrix(2,0) = 3.0;
    matrix(2,1) = 3.0;
    matrix(2,2) = 3.0;
    Vector<Histogram> histogram(matrix.get_columns_number());
    histogram = histograms(matrix, 3);
    Vector<size_t> solution({1, 1, 1});

    assert_true(histogram[0].frequencies == solution, LOG);
    assert_true(histogram[1].frequencies == solution, LOG);
    assert_true(histogram[2].frequencies == solution, LOG);
 }


void StatisticsTest::test_total_frequencies()
{
    cout << "test_total_frequencies\n";

    Vector<double> vector1;
    Vector<double> vector2;
    Vector<double> vector3;
    Vector<size_t> total_frequencies;
    Vector<Histogram> histograms(2);

    // Test
    vector1.set(0.0, 1, 9.0);
    vector2.set(5);
    vector2[0] = 0.0;
    vector2[1] = 2.0;
    vector2[2] = 6.0;
    vector2[3] = 6.0;
    vector2[4] = 9.0;

    histograms[0] = histogram(vector1, 10);
    histograms[1] = histogram(vector2, 10);

    total_frequencies = OpenNN::total_frequencies(histograms);

    assert_true(total_frequencies[0] == 1, LOG);
    assert_true(total_frequencies[1] == 0, LOG);
}



void StatisticsTest::test_histograms_missing_values()
{
    cout << "test_histograms_missing_values\n";

    Matrix<double> matrix(3,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 1.0;
    matrix(0,2) = static_cast<double>(NAN);
    matrix(1,0) = 2.0;
    matrix(1,1) = 2.0;
    matrix(1,2) = 1.0;
    matrix(2,0) = 3.0;
    matrix(2,1) = 3.0;
    matrix(2,2) = 2.0;
    Vector<Histogram> histograms(3);
    histograms = histograms_missing_values(matrix, 3);
    Vector<size_t> solution({1, 1, 1});
    Vector<size_t> solution_missing_values({1, 0, 1});

    assert_true(histograms[0].frequencies == solution, LOG);
    assert_true(histograms[1].frequencies == solution, LOG);
    assert_true(histograms[2].frequencies == solution_missing_values, LOG);
}


void StatisticsTest::test_calculate_minimal_index()
{
   cout << "test_calculate_minimal_index\n";

   Vector<double> vector;
   vector.set(1);
   vector[0] = static_cast<double>(NAN);

   assert_true(minimal_index(vector) == 0.0, LOG);

   vector.set(3);
   vector[0] = 1.0;
   vector[1] = 0.0;
   vector[2] = -1.0;

   assert_true(minimal_index(vector) == 2.0, LOG);
}

void StatisticsTest::test_calculate_maximal_index()
{
   cout << "test_calculate_maximal_index\n";

   Vector<double> vector;
   vector.set(1);
   vector[0] = static_cast<double>(NAN);

   assert_true(maximal_index(vector) == 0.0, LOG);

   vector.set(3);
   vector[0] = -1;
   vector[1] = 0;
   vector[2] = 1;

   assert_true(maximal_index(vector) == 2.0, LOG);
}


void StatisticsTest::test_calculate_minimal_indices()
{
    cout << "test_calculate_minimal_indices\n";

    Vector<double> vector;
    Vector<size_t> min_indices;

    // Test
    vector.set(1);
    vector[0] = static_cast<double>(NAN);
    min_indices = minimal_indices(vector, 0);

    assert_true(min_indices.empty(), LOG);

    // Test
    vector.set(4);
    vector[0] = 0;
    vector[1] = 0;
    vector[2] = 0;
    vector[3] = 0;

    min_indices = minimal_indices(vector, 2);

    assert_true(min_indices[1] == 0 || min_indices[1] == 1, LOG);

    //Test

    vector.set(5);
    vector[0] = 0;
    vector[1] = 1;
    vector[2] = 0;
    vector[3] = 2;
    vector[4] = 0;

    min_indices = minimal_indices(vector, 5);

    assert_true(min_indices[0] == 0 || min_indices[0] == 2 || min_indices[0] == 4, LOG);
    assert_true(min_indices[1] == 0 || min_indices[1] == 2 || min_indices[1] == 4, LOG);
    assert_true(min_indices[2] == 0 || min_indices[2] == 2 || min_indices[2] == 4, LOG);
    assert_true(min_indices[3] == 1, LOG);
    assert_true(min_indices[4] == 3, LOG);

    // Test

    vector.set(4);
    vector[0] = -1.0;
    vector[1] = 2.0;
    vector[2] = -3.0;
    vector[3] = 4.0;

    min_indices = minimal_indices(vector, 2);

    assert_true(min_indices[0] == 2, LOG);
    assert_true(min_indices[1] == 0, LOG);

}

void StatisticsTest::test_calculate_maximal_indices()
{
    cout << "test_calculate_maximal_indices\n";

    Vector<double> vector({0, 1, 2, 3, 4, 5, 6});

    Vector<size_t> solution({6, 5, 4, 3});

    Vector<size_t> maximal_indices = OpenNN::maximal_indices(vector, 4);

    assert_true((maximal_indices[0] - solution[0]) < 1.0e-7, LOG);
    assert_true((maximal_indices[1] - solution[1]) < 1.0e-7, LOG);
    assert_true((maximal_indices[2] - solution[2]) < 1.0e-7, LOG);
    assert_true((maximal_indices[3] - solution[3]) < 1.0e-7, LOG);
}


void StatisticsTest::test_calculate_norm()
{
   cout << "test_calculate_norm\n";

   Vector<double> vector;

   assert_true(l2_norm(vector) == 0.0, LOG);

   vector.set(2);
   vector.initialize(1);

   assert_true(abs(l2_norm(vector) - sqrt(2.0)) < 1.0e-6, LOG);
}


void StatisticsTest::test_calculate_quartiles_missing_values()
{
    cout << "test_calculate_quartiles_missing_values\n";

    Vector<double> vector;
    vector.set(5);
    vector[0] = 1;
    vector[1] = 2;
    vector[2] = 3;
    vector[3] = static_cast<double>(NAN);
    vector[4] = 4;
    Vector<double> solution({1.5, 2.5, 3.5});

    assert_true(quartiles_missing_values(vector) == solution , LOG);

    Vector<double> solution_2({2.0, 3.0, 4.0});
    vector.set(6);
    vector[0] = 1;
    vector[1] = 2;
    vector[2] = 3;
    vector[3] = static_cast<double>(NAN);
    vector[4] = 4;
    vector[5] = 5;

    assert_true(quartiles_missing_values(vector) == solution_2, LOG);
}


void StatisticsTest::test_calculate_box_plot()
{
    cout << "test_calculate_box_plot\n";

    Vector<double> vector({2.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    BoxPlot boxplot = box_plot(vector);

    BoxPlot solution(2.0, 2.5, 5.5, 7.5, 9.0);

    assert_true((boxplot.minimum - solution.minimum) < 1.0e-7, LOG);
    assert_true((boxplot.first_quartile - solution.first_quartile) < 1.0e-7, LOG);
    assert_true((boxplot.median - solution.median) < 1.0e-7, LOG);
    assert_true((boxplot.third_quartile - solution.third_quartile) < 1.0e-7, LOG);
    assert_true((boxplot.maximum - solution.maximum) < 1.0e-7, LOG);
}


void StatisticsTest::test_calculate_box_plot_missing_values()
{
    Vector<double> vector({2.0, 2.0, 3.0, 5.0, 6.0, 7.0, static_cast<double>(NAN), 8.0, 9.0});

    BoxPlot boxplot = box_plot_missing_values(vector);

    BoxPlot solution(2.0, 2.5, 5.5, 7.5, 9.0);

    assert_true((boxplot.minimum - solution.minimum) < 1.0e-7, LOG);
    assert_true((boxplot.first_quartile - solution.first_quartile) < 1.0e-7, LOG);
    assert_true((boxplot.median - solution.median) < 1.0e-7, LOG);
    assert_true((boxplot.third_quartile - solution.third_quartile) < 1.0e-7, LOG);
    assert_true((boxplot.maximum - solution.maximum) < 1.0e-7, LOG);
}


void StatisticsTest::test_calculate_histogram_missing_values()
{
    cout << "test_calculate_histogram_missing_values\n";

    //Histogram_missing_values
    Histogram graphic;
    Vector<double> centers;
    Vector<size_t> frequencies;
    Vector<double> vector;
    vector.set(5);
    vector[0] = 1;
    vector[1] = 3;
    vector[2] = 2;
    vector[3] = 4;
    vector[4] = static_cast<double>(NAN);

    graphic = histogram_missing_values(vector, 4);

    centers = graphic.centers;
    frequencies = graphic.frequencies;

    //Normal histogram

    Histogram graphic_2;

    Vector<double> centers_2;
    Vector<size_t> frequencies_2;

    Vector<double> vector_2;
    vector_2.set(4);

    vector_2[0] = 1;
    vector_2[1] = 3;
    vector_2[2] = 2;
    vector_2[3] = 4;

    graphic_2 = histogram(vector, 4);

    centers_2 = graphic_2.centers;
    frequencies_2 = graphic_2.frequencies;

    assert_true(centers_2 == centers , LOG);
    assert_true(frequencies_2 == frequencies, LOG );
}


void StatisticsTest::test_descriptives_missing_values()
{
    cout << "descriptives_missing_values\n";

    Vector<double> vector;
    vector.set(5);

    vector[0] = 1;
    vector[1] = 1;
    vector[2] = static_cast<double>(NAN);
    vector[3] = 1;
    vector[4] = 1;

    Descriptives descriptives;
    descriptives = OpenNN::descriptives_missing_values(vector);

    double minimun = descriptives.minimum ;
    double maximun = descriptives.maximum;
    double average = descriptives.mean;
    double standard_dev = descriptives.standard_deviation;

    Vector<double> vector_2;
    vector_2.set(4);

    vector[0] = 1;
    vector[1] = 1;
    vector[2] = 1;
    vector[3] = 1;

    Descriptives descriptives_2;

    descriptives_2 = OpenNN::descriptives_missing_values(vector);

    double minimun_2 = descriptives_2.minimum ;
    double maximun_2 = descriptives_2.maximum;
    double average_2 = descriptives_2.mean;
    double standard_dev_2 = descriptives_2.standard_deviation;

    assert_true(abs(minimun - minimun_2) < 1.0e-3, LOG);
    assert_true(abs(maximun - maximun_2) < 1.0e-3, LOG);
    assert_true(abs(average - average_2) < 1.0e-3, LOG);
    assert_true(abs(standard_dev - standard_dev_2) < 1.0e-3, LOG);
}

void StatisticsTest::test_calculate_means_binary_column()
{
    cout << "test_calculate_means_binary_column";

    Matrix<double> matrix(3,2);
    matrix(0,0) = 1;
    matrix(0,1) = 1;
    matrix(0,2) = 1;
    matrix(0,3) = 1;
    matrix(1,0) = 1;
    matrix(1,1) = 1;
    matrix(1,2) = 0;
    matrix(1,3) = 0;
    Vector<double> solution({1.0, 1.0});

    assert_true(means_binary_column(matrix) == solution, LOG);
}

void StatisticsTest::test_means_binary_columns()
{
    cout << "test_means_binary_columns\n";

    Matrix<double> matrix(3,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 0.0;
    matrix(0,2) = 7.0;
    matrix(1,0) = 1.0;
    matrix(1,1) = 1.0;
    matrix(1,2) = 8.0;
    matrix(2,0) = 0.0;
    matrix(2,1) = 0.0;
    matrix(2,2) = 5.0;
    Vector<double> solution({7.5, 8});

    Vector<double> means;
    means.set(matrix.get_columns_number());
    means = means_binary_columns(matrix);
    assert_true(means == solution, LOG);
}


void StatisticsTest::test_weighted_mean()
{
    cout << "test_weighted_mean\n";

    Vector<double> vector;
    vector.set(4);
    vector[0] = 1;
    vector[1] = 1;
    vector[2] = 1;
    vector[3] = 1;

    Vector<double> weights;
    weights.set(4);
    weights[0] = 0.25;
    weights[1] = 0.25;
    weights[2] = 0.25;
    weights[3] = 0.25;

    assert_true(weighted_mean(vector, weights) == 1.0, LOG);
}


void StatisticsTest::test_calculate_mean_missing_values()
{
    cout << "test_calculate_mean_missing_values\n";

    Vector<double> vector;
    vector.set(4);
    vector[0] = 1;
    vector[1] = 1;
    vector[2] = static_cast<double>(NAN);
    vector[3] = 1;

    assert_true(mean_missing_values(vector) == 1.0, LOG);
}


void StatisticsTest::test_percentiles()
{
    cout << "test_percentiles\n";

    Vector<double> vector(20);

    vector.initialize_sequential();

    Vector<double> percentiles = OpenNN::percentiles(vector);

    Vector<double> solution({2.5,4.5,6.5,8.5,10.5,12.5,14.5,16.5,18.5,19});

    assert_true((percentiles[0] - solution[0]) < 1.0e-7, LOG);
    assert_true((percentiles[1] - solution[1]) < 1.0e-7, LOG);
    assert_true((percentiles[2] - solution[2]) < 1.0e-7, LOG);
    assert_true((percentiles[3] - solution[3]) < 1.0e-7, LOG);
    assert_true((percentiles[4] - solution[4]) < 1.0e-7, LOG);
    assert_true((percentiles[5] - solution[5]) < 1.0e-7, LOG);
    assert_true((percentiles[6] - solution[6]) < 1.0e-7, LOG);
    assert_true((percentiles[7] - solution[7]) < 1.0e-7, LOG);
    assert_true((percentiles[8] - solution[8]) < 1.0e-7, LOG);
    assert_true((percentiles[9] - solution[9]) < 1.0e-7, LOG);
}


void StatisticsTest::test_percentiles_missing_values()
{
    cout << "test_percentiles_missing_values\n";

    Vector<double> vector(21);

    vector.initialize_sequential();

    Vector<double> percentiles = OpenNN::percentiles(vector);

    percentiles[21] = static_cast<double>(NAN);

    Vector<double> solution({2.5,4.5,6.5,8.5,10.5,12.5,14.5,16.5,18.5,20});

    assert_true((percentiles[0] - solution[0]) < 1.0e-7, LOG);
    assert_true((percentiles[1] - solution[1]) < 1.0e-7, LOG);
    assert_true((percentiles[2] - solution[2]) < 1.0e-7, LOG);
    assert_true((percentiles[3] - solution[3]) < 1.0e-7, LOG);
    assert_true((percentiles[4] - solution[4]) < 1.0e-7, LOG);
    assert_true((percentiles[5] - solution[5]) < 1.0e-7, LOG);
    assert_true((percentiles[6] - solution[6]) < 1.0e-7, LOG);
    assert_true((percentiles[7] - solution[7]) < 1.0e-7, LOG);
    assert_true((percentiles[8] - solution[8]) < 1.0e-7, LOG);
    assert_true((percentiles[9] - solution[9]) < 1.0e-7, LOG);
}



void StatisticsTest::test_means_binary_columns_missing_values()
{
    cout << "calculate_means_binary_columns_missing_values\n";

    Matrix<double> matrix(4,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 0.0;
    matrix(0,2) = 7.0;
    matrix(1,0) = 1.0;
    matrix(1,1) = 1.0;
    matrix(1,2) = 8.0;
    matrix(2,0) = 0.0;
    matrix(2,1) = 0.0;
    matrix(2,2) = 5.0;
    matrix(3,0) = 1.0;
    matrix(3,0) = 1.0;
    matrix(3,0) = static_cast<double>(NAN);

    Vector<double> solution({7.5, 8});
    assert_true(means_binary_columns_missing_values(matrix) == solution, LOG);
}


void StatisticsTest::test_minimum_matrix()
{
    cout << "test_maximum_matrix\n";

    Matrix<double> matrix(4,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 0.0;
    matrix(0,2) = 7.0;
    matrix(1,0) = 1.0;
    matrix(1,1) = 1.0;
    matrix(1,2) = 8.0;
    matrix(2,0) = -8.0;
    matrix(2,1) = 0.0;
    matrix(2,2) = 5.0;
    matrix(3,0) = 1.0;
    matrix(3,0) = 1.0;
    matrix(3,0) = 7.0;

    assert_true(abs(minimum_matrix(matrix) - (-8.0)) < 1.0e-6, LOG);

    Matrix<double> matrix_missing_values(4,3);
    matrix_missing_values(0,0) = 1.0;
    matrix_missing_values(0,1) = 0.0;
    matrix_missing_values(0,2) = 7.0;
    matrix_missing_values(1,0) = 1.0;
    matrix_missing_values(1,1) = 1.0;
    matrix_missing_values(1,2) = 8.0;
    matrix_missing_values(2,0) = static_cast<double>(NAN);
    matrix_missing_values(2,1) = 0.0;
    matrix_missing_values(2,2) = 5.0;
    matrix_missing_values(3,0) = 1.0;
    matrix_missing_values(3,0) = 1.0;
    matrix_missing_values(3,0) = 7.0;

    assert_true(abs(minimum_matrix(matrix_missing_values) - (0.0)) < 1.0e-6, LOG);
}


void StatisticsTest::test_maximum_matrix()
{
    cout << "test_maximum_matrix\n";

    Matrix<double> matrix(4,3);
    matrix(0,0) = 1.0;
    matrix(0,1) = 0.0;
    matrix(0,2) = 7.0;
    matrix(1,0) = 1.0;
    matrix(1,1) = 1.0;
    matrix(1,2) = 8.0;
    matrix(2,0) = -8.0;
    matrix(2,1) = 0.0;
    matrix(2,2) = 5.0;
    matrix(3,0) = 1.0;
    matrix(3,0) = 1.0;
    matrix(3,0) = 7.0;

    assert_true(abs(maximum_matrix(matrix) - 8.0) < 1.0e-6, LOG);

    Matrix<double> matrix_missing_values(4,3);
    matrix_missing_values(0,0) = 1.0;
    matrix_missing_values(0,1) = 0.0;
    matrix_missing_values(0,2) = 7.0;
    matrix_missing_values(1,0) = 1.0;
    matrix_missing_values(1,1) = 1.0;
    matrix_missing_values(1,2) = 8.0;
    matrix_missing_values(2,0) = static_cast<double>(NAN);
    matrix_missing_values(2,1) = 0.0;
    matrix_missing_values(2,2) = 5.0;
    matrix_missing_values(3,0) = 1.0;
    matrix_missing_values(3,0) = 1.0;
    matrix_missing_values(3,0) = 7.0;

    assert_true(abs(maximum_matrix(matrix_missing_values) - 8.0) < 1.0e-6, LOG);
}



void StatisticsTest::test_means_by_categories()
{
    cout << "test_means_by_categories\n";

    Matrix<double> matrix({Vector<double>({1,2,3,1,2,3}),Vector<double>({6,2,3,12,2,3})});

    Vector<double> solution({9.0, 2.0, 3.0});

    assert_true(means_by_categories(matrix) == solution, LOG);
}



void StatisticsTest::test_means_by_categories_missing_values()
{
    cout << "test_means_by_categories_missing_values\n";

    Matrix<double> matrix({Vector<double>({1,1,1,2,2,2}),Vector<double>({1,1,1,2,6,static_cast<double>(NAN)})});

    Vector<double> solution({1.0, 4.0});

    assert_true(means_by_categories_missing_values(matrix) == solution, LOG);
}


void StatisticsTest::run_test_case()
{
   cout << "Running descriptives test case...\n";

   // Constructor and destructor methods
   test_constructor();
   test_destructor();

   // Descriptives
   test_set_mean();
   test_set_standard_deviation();
   test_has_mean_zero_standard_deviation_one();
   test_has_minimum_minus_one_maximum_one();

   // Minimum
   test_set_minimum();
   test_minimum();
   test_minimum_missing_values();
   test_maximum_matrix();

   // Maximun
   test_set_maximum();
   test_maximum_missing_values();
   test_maximum_matrix();

   // Mean
   test_calculate_mean();
   test_weighted_mean();
   test_calculate_mean_missing_values();

   // Mean binary
   test_means_binary_columns();
   test_means_binary_columns_missing_values();

   // Median
   test_calculate_median();
   test_calculate_median_missing_values();

   // Variance
   test_variance();
   test_calculate_variance_missing_values();

   // Assymetry
   test_calculate_asymmetry();
   test_calculate_asymmetry_missing_values();

   // Kurtosis
   test_calculate_kurtosis();
   test_calculate_kurtosis_missing_values();

   // Standard deviation
   test_standard_deviation();
   test_standard_deviation_missing_values();

   // Quartiles
   test_quartiles();
   test_calculate_quartiles_missing_values();

   // Box plot
   test_calculate_box_plot();
   test_calculate_box_plot_missing_values();

   // Descriptives struct
   test_descriptives_missing_values();

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
   test_calculate_histogram();
   test_total_frequencies();
   test_calculate_histograms();
   test_calculate_histogram_missing_values();
   test_histograms_missing_values();

   // Minimal indices
   test_calculate_minimal_index();
   test_calculate_minimal_indices();

   // Maximal indices
   test_calculate_maximal_index();
   test_calculate_maximal_indices();

   // Normality
   test_calculate_norm();

   // Percentiles
   test_percentiles();
   test_percentiles_missing_values();

   // Means by categories
   test_means_by_categories();
   test_means_by_categories_missing_values();


   cout << "End of descriptives test case.\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
