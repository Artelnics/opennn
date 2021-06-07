//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   T E S T   C L A S S                                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set_test.h"

DataSetTest::DataSetTest() : UnitTesting() 
{
    data_set.set_display(false);
}


DataSetTest::~DataSetTest()
{
}


void DataSetTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default constructor

   DataSet data_set_1;

   assert_true(data_set_1.get_variables_number() == 0, LOG);
   assert_true(data_set_1.get_samples_number() == 0, LOG);

   // Samples and variables number constructor

   DataSet data_set_2(1, 2);

   assert_true(data_set_2.get_samples_number() == 1, LOG);
   assert_true(data_set_2.get_variables_number() == 2, LOG);

   // Inputs, targets and samples numbers constructor

   DataSet data_set_3(1, 1, 1);

   assert_true(data_set_3.get_variables_number() == 2, LOG);
   assert_true(data_set_3.get_samples_number() == 1, LOG);
   assert_true(data_set_3.get_target_variables_number() == 1,LOG);
   assert_true(data_set_3.get_input_variables_number() == 1,LOG);
}


void DataSetTest::test_destructor()
{
   cout << "test_destructor\n";

   DataSet* data_set_pointer = new DataSet(1, 1, 1);

   delete data_set_pointer;
}


void DataSetTest::test_get_samples_number()
{
   cout << "test_get_samples_number\n";

   assert_true(data_set.get_samples_number() == 0, LOG);
}


void DataSetTest::test_get_variables_number() 
{
   cout << "test_get_variables_number\n";

   assert_true(data_set.get_variables_number() == 0, LOG);
}


void DataSetTest::test_get_variables() 
{
   cout << "test_get_variables\n";

   data_set.set(1, 3, 2);

   assert_true(data_set.get_input_variables_number() == 3, LOG);
   assert_true(data_set.get_target_variables_number() == 2, LOG);
   assert_true(data_set.get_unused_variables_number() == 0, LOG);
   assert_true(data_set.get_used_variables_number() == 5, LOG);
}


void DataSetTest::test_get_data() 
{
   cout << "test_get_data\n";

   data_set.set(1,1,1);

   data_set.set_data_constant(0.0);

   data = data_set.get_data();

   assert_true(data.dimension(0) == 1, LOG);
   assert_true(data.dimension(1) == 2, LOG);
   assert_true(data(0) - static_cast<type>(0.0) < static_cast<type>(1.0e-6), LOG);

   data_set.set(2,3,2);

   data_set.set_data_constant(1.0);

   data = data_set.get_data();

   assert_true(data.dimension(0) == 2, LOG);
   assert_true(data.dimension(1) == 5, LOG);
}


void DataSetTest::test_get_training_data()
{
   cout << "test_get_training_data\n";

   Tensor<type, 2> training_data;

   Tensor<type, 2> solution;

   // Test

   data.resize(3, 3);
   data.setValues({{1,4,6},{4,3,6},{7,8,9}});

   data_set.set(data);

   training_indices.resize(2);

   training_indices.setValues({0,1});

   data_set.set_testing();
   data_set.set_training(training_indices);

   training_data = data_set.get_training_data();

   solution.resize(2, 3);
   solution.setValues({{1,4,6},{4,3,6}});

   assert_true(training_data(0,0) - solution(0,0) < static_cast<type>(1.0e-6), LOG);
   assert_true(training_data(0,1) - solution(0,1) < static_cast<type>(1.0e-6), LOG);
   assert_true(training_data(0,2) - solution(0,2) < static_cast<type>(1.0e-6), LOG);
   assert_true(training_data(1,0) - solution(1,0) < static_cast<type>(1.0e-6), LOG);
   assert_true(training_data(1,1) - solution(1,1) < static_cast<type>(1.0e-6), LOG);
   assert_true(training_data(1,2) - solution(1,2) < static_cast<type>(1.0e-6), LOG);
}


void DataSetTest::test_get_selection_data()
{
   cout << "test_get_selection_data\n";

   Tensor<Index, 1> selection_indices;

   Tensor<type, 2> selection_data;
   Tensor<type, 2> solution;

   // Test

   data.resize(3, 3);

   data.setValues({{1,4,6},{4,3,6},{7,8,9}});

   data_set.set(data);
   data_set.set_training();

   selection_indices.resize(2);
   selection_indices.setValues({0,1});

   data_set.set_selection(selection_indices);

   selection_data = data_set.get_selection_data();

   solution.resize(2, 3);
   solution.setValues({{1,4,6},{4,3,6}});

   assert_true(selection_data(0,0) - solution(0,0) < static_cast<type>(1.0e-6), LOG);
   assert_true(selection_data(0,1) - solution(0,1) < static_cast<type>(1.0e-6), LOG);
   assert_true(selection_data(0,2) - solution(0,2) < static_cast<type>(1.0e-6), LOG);
   assert_true(selection_data(1,0) - solution(1,0) < static_cast<type>(1.0e-6), LOG);
   assert_true(selection_data(1,1) - solution(1,1) < static_cast<type>(1.0e-6), LOG);
   assert_true(selection_data(1,2) - solution(1,2) < static_cast<type>(1.0e-6), LOG);
}


void DataSetTest::test_get_testing_data()
{
   cout << "test_get_testing_data\n";

   Tensor<type, 2> testing_data;

   Tensor<type, 2> solution;

   // Test

   data.resize(3, 3);
   data.setValues({{1,4,6},{4,3,6},{7,8,9}});

   data_set.set(data);

   testing_indices.resize(2);
   testing_indices.setValues({0,1});

   data_set.set_training();
   data_set.set_testing(testing_indices);

   testing_data = data_set.get_testing_data();

   solution.resize(2, 3);
   solution.setValues({{1,4,6},{4,3,6}});

   assert_true(testing_data(0,0) - solution(0,0) < static_cast<type>(1.0e-6), LOG);
   assert_true(testing_data(0,1) - solution(0,1) < static_cast<type>(1.0e-6), LOG);
   assert_true(testing_data(0,2) - solution(0,2) < static_cast<type>(1.0e-6), LOG);
   assert_true(testing_data(1,0) - solution(1,0) < static_cast<type>(1.0e-6), LOG);
   assert_true(testing_data(1,1) - solution(1,1) < static_cast<type>(1.0e-6), LOG);
   assert_true(testing_data(1,2) - solution(1,2) < static_cast<type>(1.0e-6), LOG);

}


void DataSetTest::test_get_input_data()
{
   cout << "test_get_inputs\n";

   data_set.set(1, 3, 2);

   Index samples_number = data_set.get_samples_number();
   Index inputs_number = data_set.get_input_variables_number();

   Tensor<type, 2> input_data = data_set.get_input_data();

   Index rows_number = input_data.dimension(0);
   Index columns_number = input_data.dimension(1);

   assert_true(samples_number == rows_number, LOG);
   assert_true(inputs_number == columns_number, LOG);
}


void DataSetTest::test_get_target_data()
{
   cout << "test_get_targets\n";

   Index samples_number;
   Index targets_number;

   Tensor<type, 2> target_data;

   Index rows_number;
   Index columns_number;

   // Test

   data_set.set(1, 3, 2);

   samples_number = data_set.get_samples_number();
   targets_number = data_set.get_target_variables_number();

   target_data = data_set.get_target_data();

   rows_number = target_data.dimension(0);
   columns_number = target_data.dimension(1);

   assert_true(samples_number == rows_number, LOG);
   assert_true(targets_number == columns_number, LOG);
}


void DataSetTest::test_get_sample()
{
   cout << "test_get_sample\n";

   Tensor<type, 1> sample;

   // Test

   data_set.set(1, 1, 1);
   data_set.set_data_constant(1.0);

   sample = data_set.get_sample_data(0);

   assert_true(sample.size() == 2, LOG);
   assert_true(sample(0) - static_cast<type>(1.0) < static_cast<type>(1.0e-6), LOG);

   // Test several variables

   data_set.set(4, 3, 1);
   data_set.set_data_constant(1.0);

   Tensor<Index, 1> indices_variables(2);
   indices_variables.setValues({1,3});
   Tensor<type, 1> sample_0 = data_set.get_sample_data(0, indices_variables);
   Tensor<type, 1> sample_1 = data_set.get_sample_data(1, indices_variables);

   assert_true(sample_0(0) - sample_1(0) < static_cast<type>(1.0e-6) && sample_0(1) - sample_1(1) < static_cast<type>(1.0e-6), LOG);
}


void DataSetTest::test_set() 
{
   cout << "test_set\n";

   // Samples and inputs and target variables

   data_set.set(1, 2, 3);

   assert_true(data_set.get_samples_number() == 1, LOG);
   assert_true(data_set.get_input_variables_number() == 2, LOG);
   assert_true(data_set.get_target_variables_number() == 3, LOG);

   data = data_set.get_data();

   assert_true(data.dimension(0) == 1, LOG);
   assert_true(data.dimension(1) == 5, LOG);
}


void DataSetTest::test_set_samples_number() 
{
   cout << "test_set_samples_number\n";

   data_set.set(1,1,1);

   data_set.set_samples_number(2);

   assert_true(data_set.get_samples_number() == 2, LOG);
}


void DataSetTest::test_set_columns_number()
{
   cout << "test_set_columns_number\n";

   data_set.set(1, 1);
   data_set.set_columns_number(2);

   assert_true(data_set.get_variables_number() == 2, LOG);
}


void DataSetTest::test_set_data()
{
   cout << "test_set_data\n";

   data_set.set(1, 1, 1);

   data.resize(1,2);
   data.setValues({{1,2}});

   data_set.set_data(data);

   data = data_set.get_data();

   assert_true(abs(data(0,0)) < static_cast<type>(1.0e-6), LOG);
   assert_true(abs(data(0,1)) < static_cast<type>(1.0e-6), LOG);
}


void DataSetTest::test_calculate_variables_descriptives()
{
   cout << "test_calculate_variables_descriptives\n";

   Tensor<Descriptives, 1> descriptives;

   ofstream file;
   string data_string;

   // Test

   data_set.set(1, 1);

   data_set.set_data_constant(0.0);

   descriptives = data_set.calculate_variables_descriptives();

   assert_true(descriptives.size() == 1, LOG);

   assert_true(descriptives[0].minimum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[0].maximum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[0].mean < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[0].standard_deviation < static_cast<type>(1.0e-6), LOG);

   // Test

   data_set.set(2, 2, 2);

   data_set.set_data_constant(0.0);

   descriptives = data_set.calculate_variables_descriptives();

   assert_true(descriptives.size() == 4, LOG);

   assert_true(descriptives[0].minimum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[0].maximum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[0].mean < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[0].standard_deviation < static_cast<type>(1.0e-6), LOG);

   assert_true(descriptives[1].minimum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[1].maximum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[1].mean < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[1].standard_deviation < static_cast<type>(1.0e-6), LOG);

   assert_true(descriptives[2].minimum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[2].maximum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[2].mean < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[2].standard_deviation < static_cast<type>(1.0e-6), LOG);

   assert_true(descriptives[3].minimum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[3].maximum < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[3].mean < static_cast<type>(1.0e-6), LOG);
   assert_true(descriptives[3].standard_deviation < static_cast<type>(1.0e-6), LOG);

   const string data_file_name = "../data/data.dat";

   data_set.set_data_file_name(data_file_name);

   data_set.set_separator(' ');
   data_set.set_missing_values_label("?");

   data_string = "-1000 ? 0 \n 3 4 ? \n ? 4 1";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

    assert_true(abs(data_set.calculate_variables_descriptives()[0].minimum - (-1000)) < 1.0e-4, LOG);
    assert_true(abs(data_set.calculate_variables_descriptives()[0].minimum - 4.0) < 1.0e-4, LOG);
    assert_true(abs(data_set.calculate_variables_descriptives()[0].minimum - 0.0) < 1.0e-4, LOG);

}


void DataSetTest::test_calculate_input_variables_descriptives() //@todo
{
   cout << "test_calculate_input_variables_descriptives\n";

   Tensor<Descriptives, 1> descriptives;

   Tensor<Index, 1> indices;

   // Test

   data.resize(2, 3);
   data.setValues({{1.0,2.0,3.0}, {1.0,2.0,3.0}});

   data_set.set(data);

   indices.resize(2);
   indices.setValues({0, 1});

   descriptives = data_set.calculate_input_variables_descriptives();

   assert_true(descriptives[0].mean == 2.0, LOG);
   assert_true(descriptives[0].standard_deviation == 1.0, LOG);
   assert_true(descriptives[0].minimum == 1.0, LOG);
   assert_true(descriptives[0].maximum == 3.0, LOG);

   descriptives = data_set.calculate_input_variables_descriptives();

   assert_true(descriptives[0].mean == 2.0, LOG);
   assert_true(descriptives[0].standard_deviation == 1.0, LOG);
   assert_true(descriptives[0].minimum == 1.0, LOG);
   assert_true(descriptives[0].maximum == 3.0, LOG);
}


void DataSetTest::test_calculate_autocorrelations()
{
    cout << "test_calculate_autocorrelations\n";

    Tensor<type, 2> autocorrelations;

    // Test

    data.resize(4,3);
    data.setValues({{5,2,8}, {7,8,7}, {3,6,4}, {8,1,6}});

    data_set.set_data(data);
    data_set.set_lags_number(4);
    data_set.set_steps_ahead_number(1);
    data_set.transform_time_series();

    autocorrelations = data_set.calculate_autocorrelations(data_set.get_lags_number());

    assert_true(autocorrelations.dimension(1) == 10, LOG);
    assert_true(autocorrelations.dimension(0) == 2, LOG);
}


void DataSetTest::test_calculate_cross_correlations()
{
    cout << "test_calculate_cross_correlations\n";

    Tensor<type, 3> cross_correlations;
    Tensor<type, 2> cross_correlation_matrix;
    Tensor<type, 1> cross_correlation_lag_vector;

    // Test

    data.resize(6,3);
    data.setValues({{5,2,8}, {7,8,7}, {3,6,4}, {8,1,6}, {5,8,6}, {6,3,4}});

    data_set.set_data(data);
    data_set.set_lags_number(6);
    data_set.set_steps_ahead_number(1);
    data_set.transform_time_series();

    cross_correlations = data_set.calculate_cross_correlations(data_set.get_lags_number());

    //How to extract a lag vector

    cross_correlation_matrix = cross_correlations.chip(0,1);
    cross_correlation_lag_vector = cross_correlation_matrix.chip(0,0);

//    assert_true(cross_correlations.dimension(0) == 6, LOG);
}


void DataSetTest::test_calculate_data_distributions()
{
   cout << "test_calculate_data_distributions\n";

   Tensor<Histogram, 1> histograms;

   // Test

   data.resize(3,3);
   data(0,0) = 2.0;
   data(0,1) = 2.0;
   data(0,2) = 1.0;
   data(1,0) = 1.0;
   data(1,1) = 1.0;
   data(1,2) = 1.0;
   data(2,0) = 1.0;
   data(2,1) = 2.0;
   data(2,2) = 2.0;

   data_set.set_data(data);
   histograms = data_set.calculate_columns_distribution();
   Tensor<Index, 1> sol(2);
   sol.setValues({1, 1});
   Tensor<Index, 1> sol_2(2);
   sol_2.setValues({2, 2});
   Tensor<type, 1> centers(2);
   centers.setValues({0,1});

//   assert_true(histograms(0).frequencies == sol_2, LOG);
//   assert_true(histograms[1].frequencies == sol_2, LOG);
//   assert_true(histograms[2].frequencies == sol, LOG);

//   assert_true(histograms(0).centers == centers, LOG);
//   assert_true(histograms[1].centers == centers, LOG);
//   assert_true(histograms[2].centers == centers, LOG);
}


void DataSetTest::test_filter_data()
{
   cout << "test_filter_data\n";

   Index samples_number;
   Index inputs_number;
   Index targets_number;

   Tensor<type, 1> minimums(2);
   Tensor<type, 1> maximums(2);

   // Test

   samples_number = 1;
   inputs_number = 1;
   targets_number = 1;

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(1.0);

   minimums.setValues({2, 0.0});
   maximums.setValues({2, 0.5});

   data_set.filter_data(minimums, maximums);

   data = data_set.get_data();

   assert_true(data_set.get_sample_use(0) == DataSet::UnusedSample, LOG);
   assert_true(data_set.get_sample_use(1) == DataSet::UnusedSample, LOG);
}


void DataSetTest::test_scale_data()
{
   cout << "test_scale_data\n";

   Tensor<Descriptives, 1> data_descriptives;
   Tensor<type, 2> scaled_data;

    // Test

   data_set.set(2,2,2);
   data_set.set_data_constant(0.0);

   data.setValues({{1, 2}, {3, 4}});
   data_set.set_data(data);

   data_set.set_display(false);

   data = data_set.get_data();

//   data_descriptives = data_set.scale_data_minimum_maximum();

   scaled_data = data_set.get_data();

//   assert_true(scaled_data == data, LOG);

//   Tensor<type, 2> scaled_data;

    // Test

   data_set.set(2,2,2);
   data_set.set_data_constant(0.0);

   data_set.set_display(false);

   data = data_set.get_data();

//   data_descriptives = data_set.scale_data_mean_standard_deviation();

   scaled_data = data_set.get_data();

//   assert_true(scaled_data == data, LOG);
}


void DataSetTest::test_unuse_constant_columns()
{
   cout << "test_unuse_constant_columns\n";

   // Test 

   data_set.set(1, 2, 1);

   data_set.set_data_constant(0.0);

   data_set.unuse_constant_columns();

   assert_true(data_set.get_input_columns_number() == 0, LOG);
   assert_true(data_set.get_target_columns_number() == 1, LOG);
}


void DataSetTest::test_initialize_data()
{
   cout << "test_initialize_data\n";

   data.resize(3,3);

   data_set.set(data);

   data_set.set_data_constant(2);

   Tensor<type, 2> solution(3, 3);
   solution.setValues({{2,2,2},{2,2,2},{2,2,2}});

   assert_true(data_set.get_data()(0) == solution(0), LOG);
   assert_true(data_set.get_data()(1) == solution(1), LOG);
   assert_true(data_set.get_data()(2) == solution(2), LOG);
   assert_true(data_set.get_data()(3) == solution(3), LOG);
   assert_true(data_set.get_data()(4) == solution(4), LOG);
   assert_true(data_set.get_data()(5) == solution(5), LOG);
   assert_true(data_set.get_data()(6) == solution(6), LOG);

}


void DataSetTest::test_calculate_target_columns_distribution()
{
    cout << "test_calculate_target_columns_distribution\n";

    // Test two classes

    data.resize(4, 5);
    data.setValues({{2,5,6,9,8},{2,9,1,9,4},{6,5,6,7,3},{0,static_cast<type>(NAN),1,0,1}});

    Tensor<Index, 1> target_indices(1);
    target_indices.setValues({3});

    Tensor<Index, 1> input_variables_indices(3);
    input_variables_indices.setValues({0, 1, 2});

    data_set.set(data);

    Tensor<Index, 1> target_distribution = data_set.calculate_target_distribution();

    Tensor<Index, 1> solution(2);
    solution[0] = 2;
    solution[1] = 2;

    assert_true(target_distribution(0) == solution(0), LOG);
    assert_true(target_distribution(1) == solution(1), LOG);

    // Test more two classes

    data.resize(6, 6);
    data.setValues({{2,5,6,9,8,7},{2,9,1,9,4,5},{6,5,6,7,3,2},{0,static_cast<type>(NAN),1,0,2,2},{static_cast<type>(NAN),static_cast<type>(NAN),1,0,0,2}});

    Tensor<Index, 1> target_indices_1(2);
    target_indices_1.setValues({2,3});

    Tensor<Index, 1> inputs_indices_1(2);
    inputs_indices_1.setValues({0, 1});

    data_set.set_data(data);

    Tensor<Index, 1> calculate_target_distribution_1 = data_set.calculate_target_distribution();

    assert_true(calculate_target_distribution_1[0] == 6, LOG);
    assert_true(calculate_target_distribution_1[1] == 3, LOG);
    assert_true(calculate_target_distribution_1[2] == 2, LOG);
}


void DataSetTest::test_unuse_most_populated_target()
{
    cout << "test_unused_most_populated_target\n";

    Tensor<Index, 1> unused_samples_indices;

    // Test

    data_set.set(5,2,5);
    data_set.set_data_constant(0.0);

//    unused_samples_indices = data_set.unuse_most_populated_target(7);

    //assert_true(unused_samples_indices.size() == 5, LOG);
    //assert_true(data_set.get_used_samples_number() == 0, LOG);
    //assert_true(data_set.get_unused_samples_number() == 5, LOG);

    // Test

    data_set.set(100, 7,5);
    data_set.set_data_constant(1.0);

    //unused_samples_indices = ds2.unuse_most_populated_target(99);

    //assert_true(unused_samples_indices.size() == 99, LOG);
    //assert_true(ds2.get_used_samples_number() == 1, LOG);
    //assert_true(ds2.get_unused_samples_number() == 99, LOG);

    // Test

    data_set.set(1, 10,10);
    data_set.set_data_random();

//    unused_samples_indices = ds3.unuse_most_populated_target(50);

    //assert_true(unused_samples_indices.size() == 1, LOG);
    //assert_true(ds3.get_used_samples_number() == 0, LOG);
    //assert_true(ds3.get_unused_samples_number() == 1, LOG);
}


void DataSetTest::test_clean_Tukey_outliers()
{
    cout << "test_clean_Tukey_outliers\n";

    Tensor<type, 1> sample;

    Tensor<Tensor<Index, 1>, 1> outliers_indices;

    // Test

    data_set.set(100, 5, 1);
    data_set.set_data_random();

    sample.resize(6);
    sample.setValues({1.0, 1.9, 10.0, 1.1, 1.8});

    //data_set.set_sample(9, sample);

    outliers_indices = data_set.calculate_Tukey_outliers(1.5);

//    data_set.set_samples_unused(outliers_samples);

//    assert_true(data_set.get_unused_samples_number() == 1, LOG);
//    assert_true(outliers_number == 1, LOG);
//    assert_true(outliers_samples[0] == 9, LOG);
}


void DataSetTest::test_calculate_euclidean_distance()
{
   cout << "test_calculate_euclidean_distance\n";

   assert_true(0 == 1, LOG);

   assert_true(false, LOG);

}


void DataSetTest::test_calculate_distance_matrix()
{
   cout << "test_calculate_distance_matrix\n";
}


void DataSetTest::test_calculate_k_nearest_neighbors()
{
   cout << "test_k_nearest_neighbors\n";

//   Tensor kneware = data_set.calculate_k_nearest_neighbors();
}


void DataSetTest::test_calculate_average_reachability()
{
   cout << "test_calculate_average_reachability\n";
}


void DataSetTest::test_calculate_LOF_outliers()
{
   cout << "test_calculate_LOF_outliers\n";
}


void DataSetTest::test_unuse_local_outlier_factor_outliers()
{
   cout << "test_unuse_local_outlier_factor_outliers\n";
}


void DataSetTest::test_is_constant_numeric()
{
    cout << "test_is_constant_numeric\n";

    data_set.set_data_file_name("../../datasets/constant_variables.csv");

    data_set.set_separator(DataSet::Comma);

    data_set.read_csv();

    assert_true(data_set.get_column_type(0) == 0, LOG);
    assert_true(data_set.get_column_type(1) == 4, LOG);
    assert_true(data_set.get_column_type(2) == 1, LOG);
    assert_true(data_set.get_column_type(3) == 4, LOG);
    assert_true(data_set.get_column_type(4) == 1, LOG);
    assert_true(data_set.get_column_type(5) == 4, LOG);
    assert_true(data_set.get_column_type(6) == 0, LOG);
}


void DataSetTest::test_read_csv() 
{
   cout << "test_read_csv\n";

   ofstream file;

   string data_string;

   string data_file_name;

   // Test

   data_set.set(2, 2, 2);
   data_set.set_separator(',');
   data_file_name = "../data/data.dat";
   data_set.set_data_file_name(data_file_name);

   data_set.set_data_constant(0.0);

   data_set.set_display(false);

   data_set.save_data();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(is_equal(data, 0.0), LOG);

   // Test

   data_set.set_separator(' ');

   data_string = "\n\t\n   1 \t 2   \n\n\n   3 \t 4   \n\t\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data.dimension(0) == 2, LOG);
   assert_true(data.dimension(1) == 2, LOG);

   assert_true(abs(data(0,0) - 1) < 1.0e-4, LOG);
   assert_true(abs(data(0,1) - 2) < 1.0e-4, LOG);
   assert_true(abs(data(1,0) - 3) < 1.0e-4, LOG);
   assert_true(abs(data(1,1) - 4) < 1.0e-4, LOG);

   assert_true(data_set.get_samples_number() == 2, LOG);
   assert_true(data_set.get_variables_number() == 2, LOG);

   // Test

   data_set.set_separator('\t');

   data_string = "\n\n\n1 \t 2\n3 \t 4\n\n\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true((data(0,0) - 1.0) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(' ');

   data_string = "\n"
                 "x y\n"
                 "\n"
                 "1   2\n"
                 "3   4\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data_set.get_header_line(), LOG);
   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   assert_true(data.dimension(0) == 2, LOG);
   assert_true(data.dimension(1) == 2, LOG);

   assert_true((data(0,0) - 1.0) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(',');

   data_string = "\tx \t ,\t y \n"
                 "\t1 \t, \t 2 \n"
                 "\t3 \t, \t 4 \n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   assert_true((data(0,0) - 1.0) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(',');

   data_string = "x , y\n"
                 "1 , 2\n"
                 "3 , 4\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   assert_true((data(0,0) - 1.0 ) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0 ) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0 ) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0 ) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(',');

   data_string =
   "5.1,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_samples_number() == 4, LOG);
   assert_true(data_set.get_variables_number() == 7, LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(',');

   data_string =
   "5.1,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 7, LOG);
   assert_true(data_set.get_input_variables_number() == 4, LOG);
   assert_true(data_set.get_target_variables_number() == 3, LOG);
   assert_true(data_set.get_samples_number() == 4, LOG);

   data = data_set.get_data();

   assert_true((data(0,0) - 5.1) < 1.0e-4, LOG);
   assert_true((data(0,4) - 1) < 1.0e-4, LOG);
   assert_true((data(0,5) - 0) < 1.0e-4, LOG);
   assert_true((data(0,6) - 0) < 1.0e-4, LOG);
   assert_true((data(1,4) - 0) < 1.0e-4, LOG);
   assert_true((data(1,5) - 1) < 1.0e-4, LOG);
   assert_true((data(1,6) - 0) < 1.0e-4, LOG);
   assert_true((data(2,4) - 0) < 1.0e-4, LOG);
   assert_true((data(2,5) - 1) < 1.0e-4, LOG);
   assert_true((data(2,6) - 0) < 1.0e-4, LOG);
   assert_true((data(3,4) - 0) < 1.0e-4, LOG);
   assert_true((data(3,5) - 0) < 1.0e-4, LOG);
   assert_true((data(3,6) - 1) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(',');
   data_set.set_missing_values_label("NaN");

   data_string =
   "sepal length,sepal width,petal length,petal width,class\n"
   "NaN,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica\n"
   "0.0,0.0,0.0,0.0,NaN\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 7, LOG);
   assert_true(data_set.get_input_variables_number() == 4, LOG);
   assert_true(data_set.get_target_variables_number() == 3, LOG);

   assert_true(data_set.get_variable_name(0) == "sepal length", LOG);
   assert_true(data_set.get_variable_name(1) == "sepal width", LOG);
   assert_true(data_set.get_variable_name(2) == "petal length", LOG);
   assert_true(data_set.get_variable_name(3) == "petal width", LOG);
   assert_true(data_set.get_variable_name(4) == "Iris-setosa", LOG);
   assert_true(data_set.get_variable_name(5) == "Iris-versicolor", LOG);
   assert_true(data_set.get_variable_name(6) == "Iris-virginica", LOG);

   assert_true(data_set.get_samples_number() == 5, LOG);

   data = data_set.get_data();

   assert_true(data(0,4) == 1.0, LOG);
   assert_true(data(0,5) == 0.0, LOG);
   assert_true(data(0,6) == 0.0, LOG);
   assert_true(data(1,4) == 0.0, LOG);
   assert_true(data(1,5) == 1.0, LOG);
   assert_true(data(1,6) == 0.0, LOG);
   assert_true(data(2,4) == 0.0, LOG);
   assert_true(data(2,5) == 1.0, LOG);
   assert_true(data(2,6) == 0.0, LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(',');
   data_set.set_missing_values_label("NaN");

   data_string =
   "0,0,0\n"
   "0,0,NaN\n"
   "0,0,0\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   // Test

   data_set.set_separator(' ');

   data_string = "1 2\n3 4\n5 6\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data_set.set_variable_name(0, "x");
   data_set.set_variable_name(1, "y");

   data_set.save("../data/data_set.xml");

   data_set.load("../data/data_set.xml");

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(' ');

   data_string = "1 true\n"
                 "3 false\n"
                 "5 true\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 2, LOG);
   assert_true(data_set.get_input_variables_number() == 1, LOG);
   assert_true(data_set.get_target_variables_number() == 1, LOG);

   data = data_set.get_data();

   assert_true(data(0,1) == 1.0, LOG);
   assert_true(data(1,1) == 0.0, LOG);
   assert_true(data(2,1) == 1.0, LOG);

   // Test

   data_set.set_separator('\t');
   data_set.set_missing_values_label("NaN");

   data_string =
   "f	52	1100	32	145490	4	no\n"
   "f	57	8715	1	242542	1	NaN\n"
   "m	44	5145	28	79100	5	no\n"
   "f	57	2857	16	1	1	NaN\n"
   "f	47	3368	44	63939	1	yes\n"
   "f	59	5697	14	45278	1	no\n"
   "m	86	1843	1	132799	2	yes\n"
   "m	67	4394	25	6670	2	no\n"
   "m	40	6619	23	168081	1	no\n"
   "f	12	4204	17	1	2	no\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 7, LOG);
   assert_true(data_set.get_input_variables_number() == 6, LOG);
   assert_true(data_set.get_target_variables_number() == 1, LOG);

   data = data_set.get_data();

   assert_true(data.dimension(0) == 10, LOG);
   assert_true(data.dimension(1) == 7, LOG);
}


void DataSetTest::test_read_adult_csv()
{
    cout << "test_read_adult_csv\n";

    data_set.set_missing_values_label("?");
    data_set.set_separator(',');
    data_set.set_data_file_name("../../datasets/adult.data");
    data_set.set_has_columns_names(false);
    data_set.read_csv();

    assert_true(data_set.get_samples_number() == 1000, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Categorical, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Categorical, LOG);
}


void DataSetTest::test_read_airline_passengers_csv()
{
    cout << "test_read_airline_passengers_csv\n";

    try
    {
        //data_set.set("../../datasets/adult.data",',',true);

        assert_true(data_set.get_column_type(0) == DataSet::DateTime, LOG);
        assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    }
    catch (exception&)
    {
        assert_true(true, LOG);
        cout << "Exception, date below 1970" << endl;
    }
}


void DataSetTest::test_read_car_csv()
{
    cout << "test_read_car_csv\n";

    try
    {
        //data_set.set("../../datasets/car.data",',',false);

        assert_true(data_set.get_samples_number() == 1728, LOG);
        assert_true(data_set.get_column_type(0) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(1) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(2) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(3) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(4) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(5) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(6) == DataSet::Categorical, LOG);
    }
    catch (exception)
    {
        //Exception
        assert_true(true,LOG);

    }
}


void DataSetTest::test_read_empty_csv()
{
    cout << "test_read_empty_csv\n";

    try
    {
        //data_set.set("../../datasets/empty.csv",',',false);

        assert_true(data_set.get_samples_number() == 1, LOG);
        assert_true(data_set.get_variables_number() == 0, LOG);
    }
    catch (exception)
    {
        //Exception, File is empty
        assert_true(true,LOG);
    }
}


void DataSetTest::test_read_heart_csv()
{
    cout << "test_read_heart_csv\n";

    //data_set.set("../../datasets/heart.csv",',',true);

    assert_true(data_set.get_samples_number() == 303, LOG);
    assert_true(data_set.get_variables_number() == 14, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(8) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(9) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(10) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(11) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(12) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(13) == DataSet::Binary, LOG);
}


void DataSetTest::test_read_iris_csv()
{
    cout << "test_read_iris_csv\n";

    DataSet data_set("../../datasets/iris.data",',',false);

    assert_true(data_set.get_samples_number() == 150, LOG);
    assert_true(data_set.get_variables_number() == 7, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Categorical, LOG);
}


void DataSetTest::test_read_mnsit_csv()
{
    cout << "test_read_mnist_csv\n";

    DataSet data_set("../../datasets/mnist.csv",',',false);

    assert_true(data_set.get_samples_number() == 100, LOG);
    assert_true(data_set.get_variables_number() == 785, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(20) == DataSet::Binary, LOG);
}


void DataSetTest::test_read_one_variable_csv()
{
    cout << "test_read_one_variable_csv\n";

    DataSet data_set("../../datasets/one_variable.csv",',',false);

    assert_true(data_set.get_samples_number() == 7, LOG);
    assert_true(data_set.get_variables_number() == 1, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
}


void DataSetTest::test_read_pollution_csv()
{
    cout << "test_read_pollution_csv\n";

    DataSet data_set("../../datasets/pollution.csv",',',true);

    assert_true(data_set.get_samples_number() == 1000, LOG);
    assert_true(data_set.get_variables_number() == 13, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::DateTime, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(8) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(9) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(10) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(11) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(12) == DataSet::Numeric, LOG);
}


void DataSetTest::test_read_urinary_inflammations_csv()
{
    cout << "test_read_urinary_inflammations_csv\n";

    DataSet data_set("../../datasets/urinary_inflammations.csv",';',true);

    assert_true(data_set.get_samples_number() == 120, LOG);
    assert_true(data_set.get_variables_number() == 8, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Binary, LOG);
}


void DataSetTest::test_read_wine_csv()
{
    cout << "test_read_wine_csv\n";

    DataSet data_set("../../datasets/wine.data",',',false);

    assert_true(data_set.get_samples_number() == 178, LOG);
    assert_true(data_set.get_variables_number() == 14, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(8) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(9) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(10) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(11) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(12) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(13) == DataSet::Numeric, LOG);
}


void DataSetTest::test_read_binary_csv()
{
    cout << "test_read_binary_csv\n";

    DataSet data_set("../../datasets/binary.csv",',',false);

    assert_true(data_set.get_samples_number() == 8, LOG);
    assert_true(data_set.get_variables_number() == 3, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Binary, LOG);
}

void DataSetTest::test_print_data_preview()
{
    cout << "test_print_data_preview\n";

    //data_set.set("../../datasets/iris.data",',',false);

//    data_set.print_data_preview();

}

void DataSetTest::test_transform_time_series()
{
    cout << "test_transform_time_series\n";

    data.resize(9, 2);
    data.setValues({{1,10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}, {6, 60}, {7, 70}, {8, 80}, {9, 90}});

    data_set.set_data(data);

    data_set.set_variable_name(0, "x");
    data_set.set_variable_name(1, "y");

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    // Tests for transform_time_series, transform_time_series_data, transform_time_series_colums

    assert_true(data_set.get_columns_number() == 6, LOG);
    assert_true(data_set.get_variables_number() == 6, LOG);
    assert_true(data_set.get_samples_number() == 7, LOG);

    assert_true(data_set.get_input_variables_number() == 4, LOG);
    assert_true(data_set.get_target_variables_number() == 2, LOG);

    assert_true(data_set.get_target_columns_number() == 2, LOG);

    assert_true(data_set.get_variable_name(0) == "x_lag_1", LOG);
    assert_true(data_set.get_variable_name(1) == "y_lag_1", LOG);
    assert_true(data_set.get_variable_name(2) == "x_lag_0", LOG);
    assert_true(data_set.get_variable_name(3) == "y_lag_0", LOG);
}


void DataSetTest::test_get_time_series_data()
{
    cout << "test_get_time_series_data\n";

    data.resize(5,2);
    data.setValues({{1,10}, {2,20}, {3,30}, {4,40}, {5,50}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    Tensor<type, 2> time_series_data = data_set.get_time_series_data();

    assert_true(time_series_data.dimension(0) == 5, LOG);
    assert_true(time_series_data.dimension(1) == 2, LOG);
}


void DataSetTest::test_get_time_series_columns()
{
    cout << "test_get_time_series_columns\n";

    data.resize(5,2);
    data.setValues({{1,10}, {2,20}, {3,30}, {4,40}, {5,50}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    assert_true(data_set.get_time_series_columns()(0).name == "column_1", LOG);
    assert_true(data_set.get_time_series_columns()(0).column_use == 1, LOG);
    assert_true(data_set.get_time_series_columns()(0).type == 0, LOG);
}


void DataSetTest::test_get_time_series_columns_number()
{
    cout << "test_get_time_series_columns_number\n";

    data.resize(5,2);
    data.setValues({{1,10}, {2,20}, {3,30}, {4,40}, {5,50}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    assert_true(data_set.get_time_series_columns_number() == 2, LOG);
}


void DataSetTest::test_get_time_series_column_data()
{
    cout << "test_get_time_series_column_data\n";

    DataSet old_data_set;

    // Test

    data.resize(4,2);
    data.setValues({{1,10},{2,20},{3,30},{4,40}});

    old_data_set.set_data(data);

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    assert_true(data_set.get_time_series_column_data(1)(1) == old_data_set.get_column_data(1)(1), LOG);
}


void DataSetTest::test_get_time_series_columns_names()
{
    cout << "test_get_time_series_columns_names\n";

    // Test

    data.resize(4,2);
    data.setValues({{0,0},{1,10},{2,20},{3,30}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    assert_true(data_set.get_time_series_columns_names()(0) == "column_1", LOG);
}


void DataSetTest::test_set_time_series_data()
{
    cout << "test_set_time_series_data\n";

    data.resize(4,2);
    data.setValues({{0,0},{1,10},{2,20},{3,30}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    data.resize(5,3);
    data.setValues({{15,14,13},{12,11,10},{9,8,7},{6,5,4},{3,2,1}});

    data_set.set_time_series_data(data);

    assert_true(data_set.get_time_series_data()(0) == 15, LOG);
    assert_true(data_set.get_time_series_data()(1) == 12, LOG);
    assert_true(data_set.get_time_series_data()(2) == 9, LOG);
}


void DataSetTest::test_set_time_index()
{
    cout << "test_set_time_index\n";

    data.resize(4,2);
    data.setValues({{1,10},{2,20},{3,30},{4,40}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    Index old_time_index;
    old_time_index = data_set.get_time_index();

    data_set.set_time_index(1000);

    Index new_time_index;
    new_time_index = data_set.get_time_index();

    // set_time_index and get_time_index tests

    assert_true(old_time_index != new_time_index,LOG);
    assert_true(new_time_index == 1000,LOG);
}


void DataSetTest::test_has_time_columns()
{
    cout << "test_has_time_columns\n";

    data.resize(4,2);
    data.setValues({{1,10},{2,20},{3,30},{4,40}});

    data_set.set_data(data);

    data_set.set_column_type(0,DataSet::ColumnType::DateTime);
    data_set.set_column_type(1,DataSet::ColumnType::DateTime);

    assert_true(data_set.has_time_columns(), LOG);
}


void DataSetTest::test_save_time_series_data_binary()
{
    cout << "test:_save_time_series_data_binary";

    data.resize(4,2);
    data.setValues({{0,0},{1,10},{2,20},{3,30}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    const string dataFileName = "D:/prueba";
    data_set.set_data_file_name("dataFileName");
    data_set.save_time_series_data_binary(dataFileName);

    data_set.load_time_series_data_binary(dataFileName);

    // test_save_time_series_data_binary test_load_time_series_binary
    assert_true(data_set.get_time_series_data()(0) == 0, LOG);
    assert_true(data_set.get_time_series_data()(1) == 1, LOG);
    assert_true(data_set.get_time_series_data()(2) == 2, LOG);
}


void DataSetTest::test_set_steps_ahead_number()
{
    cout << "test_set_steps_ahead_nuber\n";

    data.resize(4,2);
    data.setValues({{0,0},{1,10},{2,20},{3,30}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    // set_lags_number and get_lags_number test
    assert_true(data_set.get_lags_number() == 2, LOG);
}


void DataSetTest::test_set_lags_number()
{
    cout << "test_set_lags_number\n";

    // Test

    data.resize(4,2);
    data.setValues({{0,0},{1,10},{2,20},{3,30}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    assert_true(data_set.get_steps_ahead() == 2, LOG);
}


void DataSetTest::test_scrub_missing_values()
{
    cout << "test_scrub_missing_values\n";

    const string data_file_name = "../data/data.dat";

    ofstream file;

    data_set.set_data_file_name(data_file_name);

    string data_string;

    // Test

    data_set.set_separator(' ');
    data_set.set_missing_values_label("NaN");
//    data_set.set_file_type("dat");

    data_string = "0 0 0\n"
                  "0 0 NaN\n"
                  "0 0 0\n";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.scrub_missing_values();

//    samples = data_set.get_samples();

//    assert_true(samples.get_use(1) == Samples::Unused, LOG);

    // Test

    data_set.set_separator(' ');
    data_set.set_missing_values_label("NaN");
//    data_set.set_file_type("dat");

    data_string = "NaN 3   3\n"
                  "2   NaN 3\n"
                  "0   1   NaN\n";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

//    data_set.read_csv();

//    data_set.scrub_missing_values();

//    samples = data_set.get_samples();

    data = data_set.get_data();

    assert_true(abs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(abs(data(1,1) - 2.0) < 1.0e-3, LOG);
    assert_true(abs(data(2,2) - 3.0) < 1.0e-3, LOG);
}


void DataSetTest::test_empty()
{
    cout << "test_emprty\n";

    data.resize(9,9);

    data_set.set(data);

//    assert_true(abs(data_set.empty() - 0) < 1.0e-6 , LOG);
}


void DataSetTest::test_filter_column()
{
    cout << "test_filter_column\n";

    Tensor<Index, 1> indices;

    // Test

    data.resize(3,3);
    data.setValues({{1,2,3},{4,2,8},{7,8,6}});

    data_set.set_data(data);

    // Test

//    assert_true(data_set.filter_column(1, 1, 3) == solution, LOG);
//    assert_true(data_set.filter_columm(0, 4, 5) == solution_1, LOG);

}


void DataSetTest::test_calculate_variables_means()
{
    cout << "test_calculate_variables_means\n";

    data.setValues({{1, 2, 3, 4},{2, 2, 2, 2},{1, 1, 1, 1}});

    data_set.set_data(data);

//    Tensor<Index, 1> index({0, 1});
//    Tensor<type, 1> means = data_set.calculate_variables_means(index);
//    Tensor<type, 1> solution(2, 2.0);

//    assert_true(means == solution, LOG);
}


void DataSetTest::test_calculate_used_targets_mean()
{
    cout << "test_calculate_used_targets_mean\n";

    data.resize(3, 4);
    data.setValues({{1, static_cast<type>(NAN), 1, 1},{2, 2, 2, 2},{3, 3, static_cast<type>(NAN), 3}});
    Tensor<Index, 1> indices(3);
    indices.setValues({0, 1, 2});
    Tensor<Index, 1> training_indexes(3);
    training_indexes.setValues({0, 1, 2});
//    data_set.set_data(matrix);

//    data_set.set_training(training_indexes);

    // Test targets
//    data_set.set_target_variables_indices(indices);

    Tensor<type, 1> means = data_set.calculate_used_targets_mean();
    Tensor<type, 1> solutions(3);
    solutions.setValues({1.0 , 2.0, 3.0});

//    assert_true(means == solutions, LOG);

    // Test target
    Tensor<Index, 1> index_target(1);
    index_target.setValues({0});
    Tensor<Index, 1> indexes_inputs(2);
    indexes_inputs.setValues({1, 2});

//    data_set.set_target_variables_indices(index_target);
//    data_set.set_input_variables_indices(indexes_inputs);

//    Tensor<type, 1> mean = data_set.calculate_training_targets_mean();
//    Tensor<type, 1> solution({1.0});

//    assert_true(mean == solution, LOG);

}


void DataSetTest::test_calculate_selection_targets_mean()
{
    cout << "test_calculate_selection_targets_mean\n";

    data.resize(4, 3);
    data.setValues({{1, static_cast<type>(NAN), 6, 9},{1, 2, 5, 2},{3, 2, static_cast<type>(NAN), 4}});
    Tensor<Index, 1> indexes_targets(1);
    indexes_targets.setValues({2});
    Tensor<Index, 1> selection_indexes(2);
    selection_indexes.setValues({0, 1});
    data_set.set_data(data);

    data_set.set_input();

    data_set.set_selection(selection_indexes);

    // Test targets
//    data_set.set_target_variables_indices(indexes_targets);

    Tensor<type, 1> means = data_set.calculate_selection_targets_mean();
    Tensor<type, 1> solutions(2);
    solutions.setValues({2.0, 3.0});

//    assert_true(means == solutions, LOG);
}


void DataSetTest::test_calculate_input_target_correlations()
{
    cout << "test_calculate_input_target_correlations\n";

    data.resize(3, 4);

    data.setValues({{1, 1, 1, 1},{2, 2, 2, 2},{3, 3, 3, 3}});

    data_set.set_data(data);

    Tensor<Index, 1> input_variables_indices;

    input_variables_indices.resize(2);

    input_variables_indices.setValues({0, 1});

//    data_set.set_input_variables_indices(input_variables_indices);

//    Tensor<type, 2> correlations_targets = data_set.calculate_inputs_targets_correlations();

    // Test linear correlation

//    assert_true(correlations_targets - 1.0 < 1.0e-3, LOG);

    // Test logistic correlation

}


void DataSetTest::test_calculate_total_input_correlations()
{
    cout << "test_calculate_total_input_correlations\n";    

    data.resize(3, 4);
    data.setValues({{1, 1, 1, 1},{2, 2, 2, 2},{3, 3, 3, 3}});

    data_set.set_data(data);

    Tensor<Index, 1> input_variables_indices(2);
    input_variables_indices.setValues({0, 1});

    Tensor<Index, 1> target_variables_indices(1);
    target_variables_indices.setValues({2});

    Tensor<type, 1> solution(2);
    solution.setValues({1, 1});

//    data_set.set_input_variables_indices(input_variables_indices);

//    Tensor<type, 1> correlations_inputs = data_set.calculate_total_input_correlations();

//    assert_true(correlations_inputs == solution, LOG);
}


void DataSetTest::test_unuse_repeated_samples()
{
    cout << "test_unuse_repeated_samples\n";

    Tensor<Index, 1> indices;

    // Test

    data.resize(3, 3);
    data.setValues({{1,2,2}, {1,2,2}, {1,6,6}});

    data_set.set_data(data);

    indices.resize(1);
    indices.setValues({2});

    assert_true(data_set.unuse_repeated_samples().size() == 1, LOG);

    // Test

    data.resize(3, 4);
    data.setValues({{1,2,2,2},{1,2,2,2},{1,6,6,6}});

    data_set.set_data(data);

    indices.resize(2);
    indices.setValues({2, 3});

//    assert_true(ds_1.unuse_repeated_samples() == indices_1, LOG);

    data.resize(3, 5);
    data.setValues({{1,2,2,4,4},{1,2,2,4,4},{1,6,6,4,4}});

    data_set.set_data(data);

    indices.resize(2);
    indices.setValues({2,4});

//    assert_true(ds_2.unuse_repeated_samples() == indices_2, LOG);
}


void DataSetTest::test_unuse_uncorrelated_columns()
{
    cout << "test_unuse_uncorrelated_columns\n";

    data.resize(3, 3);
    data.setValues({{1,0,0},{1,0,0},{1,0,1}});

}


void DataSetTest::test_calculate_training_negatives()
{
    cout << "test_calculate_training_negatives\n";

    Index training_negatives;
    Index target_index;

    // Test

    data.resize(3, 3);

    data.setValues({{ 1, 1, 1},
                      {-1,-1, 0},
                      { 0, 1, 1}});

    data_set.set_data(data);

    training_indices.resize(2);
    training_indices.setValues({0,1});

    input_variables_indices(2);
    input_variables_indices.setValues({0, 1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({2});

    target_index = 2;

    data_set.set_testing();
    data_set.set_training(training_indices);

//    data_set.set_input_variables_indices(input_variables_indices);
//    data_set.set_target_variables_indices(target_indices);

    training_negatives = data_set.calculate_training_negatives(target_index);

    data = data_set.get_data();

    assert_true(training_negatives == 1, LOG);
}


/// @todo

void DataSetTest::test_calculate_selection_negatives()
{
    cout << "test_calculate_selection_negatives\n";

    Tensor<Index, 1> selection_indices;
    Tensor<Index, 1> input_variables_indices;
    Tensor<Index, 1> target_indices;

    // Test

    data.resize(3, 3);

    data.setValues({{1, 1, 1},{-1,-1,-1},{0,1,1}});

    data_set.set_data(data);
    selection_indices.resize(2);
    selection_indices.setValues({0,1});
    input_variables_indices.resize(2);
    input_variables_indices.setValues({0, 1});
    target_indices.resize(1);
    target_indices.setValues({2});
    Index target_index = 2;

    data_set.set_testing();
    data_set.set_selection(selection_indices);

//    data_set.set_input_variables_indices(input_variables_indices);
//    data_set.set_target_variables_indices(target_indices);

    Index selection_negatives = data_set.calculate_selection_negatives(target_index);

    data = data_set.get_data();

    assert_true(selection_negatives == 0, LOG);
}


void DataSetTest::run_test_case()
{
   cout << "Running data set test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_samples_number();
   test_get_variables_number();
   test_get_variables();

    // Data methods

   test_empty();
   test_get_data();
   test_get_training_data();
   test_get_selection_data();
   test_get_input_data();
   test_get_target_data();
   test_get_testing_data();

   // Sample methods

   test_get_sample();

   // Set methods

   test_set();

   // Data methods

   test_set_data();
   test_set_samples_number();
   test_set_columns_number();

   // Data resizing methods

   test_unuse_constant_columns();
   test_unuse_repeated_samples();
   test_unuse_uncorrelated_columns();

   // Initialization methods

   test_initialize_data();

   // Statistics methods

   test_calculate_variables_descriptives();
   test_calculate_input_variables_descriptives();
   test_calculate_used_targets_mean();
   test_calculate_selection_targets_mean();

   // Histrogram methods

   test_calculate_data_distributions();

   // Filtering methods

   test_filter_data();
   test_filter_column();

   // Data scaling

   test_scale_data();

   // Classificatios methods

   // Correlations

   test_calculate_input_target_correlations();
   test_calculate_total_input_correlations();

   // Pattern recognition methods

   test_calculate_target_columns_distribution();
   test_unuse_most_populated_target();

   // Outlier detection

   test_clean_Tukey_outliers();

   test_calculate_euclidean_distance();
   test_calculate_distance_matrix();
   test_calculate_k_nearest_neighbors();
   test_calculate_average_reachability();

   // Data generation

   // Serialization methods

   test_read_csv();
   test_read_adult_csv();
   test_read_airline_passengers_csv();
   test_read_car_csv();
   test_read_empty_csv();
   test_read_heart_csv();
   test_read_iris_csv();
   test_read_mnsit_csv();
   test_read_one_variable_csv();
   test_read_pollution_csv();
   test_read_urinary_inflammations_csv();
   test_read_wine_csv();
   test_read_binary_csv();
   test_calculate_training_negatives();
   test_calculate_selection_negatives();
   test_scrub_missing_values();

   // Time series

   test_transform_time_series();
   test_get_time_series_data();
   test_get_time_series_columns();
   test_get_time_series_columns_number();
   test_get_time_series_column_data();
   test_get_time_series_columns_names();
   test_set_lags_number();
   test_set_steps_ahead_number();
   test_set_time_series_data();
   test_set_time_index();
   test_save_time_series_data_binary();
   test_has_time_columns();

   // Test if constant variables

   test_is_constant_numeric();

   // Test print data preview

   test_print_data_preview();

   test_calculate_cross_correlations();
   test_calculate_autocorrelations();

   cout << "End of data set test case.\n\n";
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
