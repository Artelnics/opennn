//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   T E S T   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "testing_analysis_test.h"

TestingAnalysisTest::TestingAnalysisTest() : UnitTesting() 
{
    testing_analysis.set_neural_network_pointer(&neural_network);
    testing_analysis.set_data_set_pointer(&data_set);
}


TestingAnalysisTest::~TestingAnalysisTest()
{
}


void TestingAnalysisTest::test_constructor()
{
    cout << "test_constructor\n";

    // Neural network and data set constructor

    TestingAnalysis testing_analysis(&neural_network,&data_set);

    assert_true(testing_analysis.get_neural_network_pointer() != nullptr, LOG);

    assert_true(testing_analysis.get_data_set_pointer() != nullptr, LOG);
}


void TestingAnalysisTest::test_destructor()
{
    cout << "test_destructor\n";

    TestingAnalysis* testing_analysis = new TestingAnalysis;
    delete testing_analysis;
}


void TestingAnalysisTest::test_calculate_error_data()
{
    cout << "test_calculate_error_data\n";

    Tensor<type, 3> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_error_data();

    assert_true(error_data.size() == 3, LOG);
    assert_true(error_data.dimension(0) == 1, LOG);
    assert_true(error_data.dimension(1) == 3, LOG);
    assert_true(static_cast<double>(error_data(0,0,0)) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_percentage_error_data()
{
    cout << "test_calculate_percentage_error_data\n";

    Tensor<type, 2> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_percentage_error_data();

    assert_true(error_data.size() == 1, LOG);
    assert_true(error_data.dimension(1) == 1, LOG);
    assert_true(static_cast<double>(error_data(0,0)) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_absolute_errors_descriptives()
{
    cout << "test_calculate_absolute_errors_descriptives\n";

    Tensor<Descriptives, 1> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_absolute_errors_descriptives();

    assert_true(error_data.size() == 1, LOG);
    assert_true(static_cast<double>(error_data[0].minimum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data[0].maximum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data[0].mean) == 0.0, LOG);
    assert_true(static_cast<double>(error_data[0].standard_deviation) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_percentage_errors_descriptives()
{
    cout << "test_calculate_percentage_error_descriptives\n";

    Tensor<Descriptives, 1> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_percentage_errors_descriptives();

    assert_true(error_data.size() == 1, LOG);
    assert_true(static_cast<double>(error_data[0].standard_deviation) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_error_data_descriptives()
{
    cout << "test_calculate_error_data_descriptives\n";

    Tensor<Tensor<Descriptives, 1>, 1> error_data_statistics;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data_statistics = testing_analysis.calculate_error_data_descriptives();

    assert_true(error_data_statistics.size() == 1, LOG);
    assert_true(error_data_statistics[0].size() == 3, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].minimum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].maximum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].mean) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].standard_deviation) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].minimum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].maximum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].mean) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].standard_deviation) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_error_data_histograms()
{
    cout << "test_calculate_error_data_histograms\n";

    Tensor<Histogram, 1> error_data_histograms;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data_histograms = testing_analysis.calculate_error_data_histograms();

    assert_true(error_data_histograms.size() == 1, LOG);
    assert_true(error_data_histograms[0].get_bins_number() == 10, LOG);
}


void TestingAnalysisTest::test_calculate_maximal_errors()
{
    cout << "test_calculate_maximal_errors\n";

    Tensor<Tensor<Index, 1>, 1> maximal_errors;

    // Test

    samples_number = 4;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    maximal_errors = testing_analysis.calculate_maximal_errors(2);

    assert_true(maximal_errors.rank() == 1, LOG);
    assert_true(maximal_errors[0](0) == 0 , LOG);
}


void TestingAnalysisTest::test_linear_regression()
{
    cout << "test_linear_regression\n";

    Index neurons_number;

    Tensor<Correlation, 1> linear_correlation;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;
    neurons_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set_testing();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    linear_correlation = testing_analysis.linear_correlation();

    assert_true(linear_correlation.size() == 1, LOG);
    assert_true(isnan(linear_correlation(0).a), LOG);
    assert_true(isnan(linear_correlation(0).b), LOG);
    assert_true(isnan(linear_correlation(0).r), LOG);
}


void TestingAnalysisTest::test_save()
{
    cout << "test_save\n";

    string file_name = "../data/linear_correlation.dat";

    testing_analysis.save(file_name);
}


void TestingAnalysisTest::test_perform_linear_regression()
{
    cout << "test_perform_linear_regression\n";

    // DataSet

    samples_number = 1;
    inputs_number = 1;
    neurons_number = 2;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_data_constant(type(0));

    data_set.set_testing();

    // Neural Network

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, targets_number});
    neural_network.set_parameters_constant(type(0));

    // Testing Analysis

    Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> goodness_of_fit_analysis = testing_analysis.perform_goodness_of_fit_analysis();

    assert_true(goodness_of_fit_analysis.size() == 1 , LOG);
    assert_true(goodness_of_fit_analysis[0].determination - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

}


void TestingAnalysisTest::test_calculate_confusion()
{
    cout << "test_calculate_confusion\n";

    // Samples* i;

    Tensor<type, 2> actual;
    Tensor<type, 2> predicted;

    // Test

    actual.resize(4, 3);
    predicted.resize(4, 3);

    actual(0,0) = type(1); actual(0,1) = type(0); actual(0,2) = type(0);
    actual(1,0) = type(0); actual(1,1) = type(1); actual(1,2) = type(0);
    actual(2,0) = type(0); actual(2,1) = type(1); actual(2,2) = type(0);
    actual(3,0) = type(0); actual(3,1) = type(0); actual(3,2) = type(1);

    predicted(0,0) = type(1); predicted(0,1) = type(0); predicted(0,2) = type(0);
    predicted(1,0) = type(0); predicted(1,1) = type(1); predicted(1,2) = type(0);
    predicted(2,0) = type(0); predicted(2,1) = type(1); predicted(2,2) = type(0);
    predicted(3,0) = type(0); predicted(3,1) = type(0); predicted(3,2) = type(1);

    Tensor<Index, 2> confusion = testing_analysis.calculate_confusion_multiple_classification(actual, predicted);

    Tensor<Index, 0> sum = confusion.sum();

    assert_true(sum(0) == 4 + 12, LOG);

    assert_true(confusion.dimension(0) == 4, LOG);
    assert_true(confusion.dimension(1) == 4, LOG);
    assert_true(confusion(0,0) == 1, LOG);
    assert_true(confusion(1,1) == 2, LOG);
    assert_true(confusion(2,2) == 1, LOG);

    assert_true(confusion(0,3) == confusion(0,0) + confusion(0,1) + confusion(0,2), LOG);
    assert_true(confusion(1,3) == confusion(1,0) + confusion(1,1) + confusion(1,2), LOG);
    assert_true(confusion(2,3) == confusion(2,0) + confusion(2,1) + confusion(2,2), LOG);

    assert_true(confusion(3,0) == confusion(0,0) + confusion(1,0) + confusion(2,0), LOG);
    assert_true(confusion(3,1) == confusion(0,1) + confusion(1,1) + confusion(2,1), LOG);
    assert_true(confusion(3,2) == confusion(0,2) + confusion(1,2) + confusion(2,2), LOG);

    assert_true(confusion(3,3) == 4, LOG);


}


void TestingAnalysisTest::test_calculate_binary_classification_test()
{
    cout << "test_calculate_binary_classification_test\n";

    // DataSet

    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_data_constant(type(0));

    data_set.set_testing();

    // Neural Network

    neural_network.set(NeuralNetwork::ProjectType::Classification, {1, 1, 1});
    neural_network.set_parameters_constant(type(0));

    // Testing Analysis

   Tensor<type, 1> binary = testing_analysis.calculate_binary_classification_tests();

   assert_true(binary.size() == 15 , LOG);

   assert_true(binary[0] == 0 , LOG);
   assert_true(binary[1] == 1 , LOG);
   assert_true(binary[2] == 0 , LOG);
   assert_true(binary[3] == 0 , LOG);
   assert_true(binary[4] == 0 , LOG);
   assert_true(binary[5] == 0 , LOG);
   assert_true(binary[6] == 0 , LOG);
   assert_true(binary[7] == 0 , LOG);
   assert_true(binary[8] == 1 , LOG);
   assert_true(binary[9] == 1 , LOG);
   assert_true(binary[10] == 0 , LOG);
   assert_true(binary[11] == 0 , LOG);
   assert_true(binary[12] == 0 , LOG);
   assert_true(binary[13] == -1 , LOG);
   assert_true(binary[14] == -1 , LOG);

}


void TestingAnalysisTest::test_calculate_Wilcoxon_parameter()
{
    cout << "test_calculate_Wilcoxon_parameter\n";

    type wilcoxon_parameter;

    // Test

    type x = type(1.5);
    type y = type(2.5);

    wilcoxon_parameter = testing_analysis.calculate_Wilcoxon_parameter(x, y);

    assert_true(abs(wilcoxon_parameter) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    wilcoxon_parameter = testing_analysis.calculate_Wilcoxon_parameter(y ,x);

    assert_true(abs(wilcoxon_parameter - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    x = y;

    wilcoxon_parameter = testing_analysis.calculate_Wilcoxon_parameter(x, y);

    assert_true(abs(wilcoxon_parameter - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void TestingAnalysisTest::test_calculate_roc_curve()
{
    cout << "test_calculate_roc_curve\n";

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<type, 2> roc_curve;

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0.0);
    targets(1,0) = type(0.0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(0.0);
    outputs(1,0) = type(0.0);
    outputs(2,0) = type(1);
    outputs(3,0) = type(1);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    assert_true(roc_curve.dimension(1) == 3, LOG);
    assert_true(roc_curve.dimension(0) == 201, LOG);

    assert_true(roc_curve(0, 0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(0, 1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(1, 0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(1, 1) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(2, 0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(2, 1) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(3, 0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(3, 1) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(4, 0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(4, 1) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0.0);
    targets(1,0) = type(1);
    targets(2,0) = type(1);
    targets(3,0) = type(0.0);

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.12);
    outputs(1,0) = static_cast<type>(0.78);
    outputs(2,0) = static_cast<type>(0.84);
    outputs(3,0) = static_cast<type>(0.99);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    assert_true(roc_curve.dimension(1) == 3, LOG);
    assert_true(roc_curve.dimension(0) == 201, LOG);

    assert_true(roc_curve(0, 0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(0, 1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(1, 0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(1, 1) - type(0.5) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(2, 0) - type(0.5) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(2, 1) - type(0.5) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(3, 0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(3, 1) - type(0.5) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(4, 0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(roc_curve(4, 1) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
}


void TestingAnalysisTest::test_calculate_area_under_curve()
{
    cout << "test_calculate_area_under_curve\n";

    Tensor<type, 2> roc_curve;

    type area_under_curve;

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(0);
    outputs(1,0) = type(0);
    outputs(2,0) = type(1);
    outputs(3,0) = type(1);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

    assert_true(area_under_curve - type(1) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test @todo check this tests

//    targets.resize(4,1);

//    targets(0,0) = type(0);
//    targets(1,0) = type(0);
//    targets(2,0) = type(1);
//    targets(3,0) = type(1);

//    outputs.resize(4,1);

//    outputs(0,0) = type(0);
//    outputs(1,0) = type(1);
//    outputs(2,0) = type(0);
//    outputs(3,0) = type(1);

//    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

//    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

//    assert_true(area_under_curve - type(0.5) < type(NUMERIC_LIMITS_MIN), LOG);

//    // Test

//    targets.resize(4,1);

//    targets(0,0) = type(0.0);
//    targets(1,0) = type(0.0);
//    targets(2,0) = type(1);
//    targets(3,0) = type(1);

//    outputs.resize(4,1);

//    outputs(0,0) = static_cast<type>(0.78);
//    outputs(1,0) = static_cast<type>(0.84);
//    outputs(2,0) = static_cast<type>(0.12);
//    outputs(3,0) = static_cast<type>(0.99);

//    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

//    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

//    assert_true(area_under_curve - type(0.5) < type(NUMERIC_LIMITS_MIN), LOG);

//    // Test

//    targets.resize(4,1);

//    targets(0,0) = type(0.0);
//    targets(1,0) = type(0.0);
//    targets(2,0) = type(1);
//    targets(3,0) = type(1);

//    outputs.resize(4,1);

//    outputs(0,0) = type(1);
//    outputs(1,0) = type(1);
//    outputs(2,0) = type(0.0);
//    outputs(3,0) = type(0.0);

//    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

//    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

//    assert_true(area_under_curve < type(NUMERIC_LIMITS_MIN), LOG);
}


void TestingAnalysisTest::test_calculate_optimal_threshold()
{
    cout << "test_calculate_optimal_threshold\n";

    type optimal_threshold;

    Tensor<type, 2> roc_curve;

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0.0);
    targets(1,0) = type(0.0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(0.0);
    outputs(1,0) = type(0.0);
    outputs(2,0) = type(1);
    outputs(3,0) = type(1);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

    assert_true(optimal_threshold - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0.0);
    targets(1,0) = type(0.0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(1);
    outputs(1,0) = type(1);
    outputs(2,0) = type(0.0);
    outputs(3,0) = type(0.0);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

    assert_true(optimal_threshold - type(1) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    targets.resize(5,1);

    targets(0,0) = type(0.0);
    targets(1,0) = type(1);
    targets(2,0) = type(0.0);
    targets(3,0) = type(1);
    targets(4,0) = type(0.0);

    outputs.resize(5,1);

    outputs(0,0) = static_cast<type>(0.33);
    outputs(1,0) = static_cast<type>(0.14);
    outputs(2,0) = static_cast<type>(0.12);
    outputs(3,0) = static_cast<type>(0.62);
    outputs(4,0) = static_cast<type>(0.85);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

    assert_true(optimal_threshold - type(0.62) < type(NUMERIC_LIMITS_MIN), LOG);
}


void TestingAnalysisTest::test_calculate_cumulative_gain()
{
    cout << "test_calculate_cumulative_chart\n";

    // Test

    targets.resize(4,1);

    targets(0,0) = type(1);
    targets(1,0) = type(0.0);
    targets(2,0) = type(1);
    targets(3,0) = type(0.0);

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.67);
    outputs(1,0) = static_cast<type>(0.98);
    outputs(2,0) = static_cast<type>(0.78);
    outputs(3,0) = static_cast<type>(0.45);

    Tensor<type, 2> cumulative_gain = testing_analysis.calculate_cumulative_gain(targets, outputs);

    assert_true(cumulative_gain.dimension(1) == 2, LOG);
    assert_true(cumulative_gain.dimension(0) == 21, LOG);
    assert_true(cumulative_gain(0, 0) - type(0.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(cumulative_gain(0, 1) - type(0.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(cumulative_gain(20, 0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(cumulative_gain(20, 1) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);
}


void TestingAnalysisTest::test_calculate_lift_chart()
{
    cout << "test_calculate_lift_chart\n";

    Tensor<type, 2> cumulative_gain;

    Tensor<type, 2> lift_chart;

    // Test

    targets.resize(4,1);

    targets(0,0) = type(1);
    targets(1,0) = type(0.0);
    targets(2,0) = type(1);
    targets(3,0) = type(0.0);

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.67);
    outputs(1,0) = static_cast<type>(0.87);
    outputs(2,0) = static_cast<type>(0.99);
    outputs(3,0) = static_cast<type>(0.88);

    cumulative_gain = testing_analysis.calculate_cumulative_gain(targets, outputs);

    lift_chart = testing_analysis.calculate_lift_chart(cumulative_gain);

    assert_true(lift_chart.dimension(1) == cumulative_gain.dimension(1), LOG);
    assert_true(lift_chart.dimension(0) == cumulative_gain.dimension(0), LOG);
}


void TestingAnalysisTest::test_calculate_calibration_plot()
{
    cout << "test_calculate_calibration_plot\n";

    Tensor<type, 2> calibration_plot;

    // Test

    targets.resize(10, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(0.0);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);
    targets(4, 0) = type(1);
    targets(5, 0) = type(1);
    targets(6, 0) = type(1);
    targets(7, 0) = type(0.0);
    targets(8, 0) = type(1);
    targets(9, 0) = type(0.0);

    outputs.resize(10, 1);

    outputs(0, 0) = static_cast<type>(0.09);
    outputs(1, 0) = static_cast<type>(0.19);
    outputs(2, 0) = static_cast<type>(0.29);
    outputs(3, 0) = static_cast<type>(0.39);
    outputs(4, 0) = static_cast<type>(0.49);
    outputs(5, 0) = static_cast<type>(0.59);
    outputs(6, 0) = static_cast<type>(0.58);
    outputs(7, 0) = static_cast<type>(0.79);
    outputs(8, 0) = static_cast<type>(0.89);
    outputs(9, 0) = static_cast<type>(0.99);

    calibration_plot = testing_analysis.calculate_calibration_plot(targets, outputs);

    assert_true(calibration_plot.dimension(1) == 2, LOG);
    assert_true(calibration_plot.dimension(0) == 11, LOG);
}


void TestingAnalysisTest::test_calculate_true_positive_samples()
{
    cout << "test_calculate_true_positive_samples\n";

    Tensor<Index, 1> true_positives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(0.0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});

    const type threshold = type(0.5);

    true_positives_indices = testing_analysis.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    assert_true(true_positives_indices.size() == 1, LOG);
    assert_true(true_positives_indices[0] == 1, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(0.0);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(0.0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_positives_indices = testing_analysis.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    const Tensor<bool, 0> not_empty = true_positives_indices.any();

    assert_true(!not_empty(0), LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_positives_indices = testing_analysis.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    assert_true(true_positives_indices.size() == 4, LOG);
    assert_true(true_positives_indices[0] == 0, LOG);
    assert_true(true_positives_indices[1] == 1, LOG);
    assert_true(true_positives_indices[2] == 2, LOG);
    assert_true(true_positives_indices[3] == 3, LOG);
}


void TestingAnalysisTest::test_calculate_false_positive_samples()
{
    cout << "test_calculate_false_positive_samples\n";

    Tensor<Index, 1> false_positives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(0.0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = type(0.5);

    false_positives_indices = testing_analysis.calculate_false_positive_samples(targets, outputs,testing_indices, threshold);

    assert_true(false_positives_indices.size() == 1, LOG);
    assert_true(false_positives_indices[0] == 2, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(0.0);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(0.0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    false_positives_indices = testing_analysis.calculate_false_positive_samples(targets, outputs, testing_indices, threshold);

    assert_true(false_positives_indices.size() == 4, LOG);
    assert_true(false_positives_indices[0] == 0, LOG);
    assert_true(false_positives_indices[1] == 1, LOG);
    assert_true(false_positives_indices[2] == 2, LOG);
    assert_true(false_positives_indices[3] == 3, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(0.0);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    false_positives_indices = testing_analysis.calculate_false_positive_samples(targets, outputs,testing_indices, threshold);

    const Tensor<bool, 0> not_empty = false_positives_indices.any();

    assert_true(!not_empty(0), LOG);

    assert_true(false_positives_indices.size() == 0, LOG);
}


void TestingAnalysisTest::test_calculate_false_negative_samples()
{
    cout << "test_calculate_false_negative_samples\n";

    Tensor<Index, 1> false_negatives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(0.0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = type(0.5);

    false_negatives_indices = testing_analysis.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    assert_true(false_negatives_indices.size() == 1, LOG);
    assert_true(false_negatives_indices[0] == 3, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(0.0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(0.0);
    outputs(3, 0) = type(0.0);

    false_negatives_indices = testing_analysis.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    assert_true(false_negatives_indices.size() == 0, LOG);

    const Tensor<bool, 0> not_empty = false_negatives_indices.any();

    assert_true(!not_empty(0), LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(0.0);
    outputs(2, 0) = type(0.0);
    outputs(3, 0) = type(0.0);

    false_negatives_indices = testing_analysis.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    assert_true(false_negatives_indices.size() == 4, LOG);
    assert_true(false_negatives_indices[0] == 0, LOG);
    assert_true(false_negatives_indices[1] == 1, LOG);
    assert_true(false_negatives_indices[2] == 2, LOG);
    assert_true(false_negatives_indices[3] == 3, LOG);
}


void TestingAnalysisTest::test_calculate_true_negative_samples()
{
    cout << "test_calculate_true_negative_samples\n";

    Tensor<Index, 1> true_negatives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(0.0);
    targets(2, 0) = type(0.0);
    targets(3, 0) = type(0.0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(0.0);
    outputs(2, 0) = type(0.0);
    outputs(3, 0) = type(0.0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = type(0.5);

    true_negatives_indices = testing_analysis.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    assert_true(true_negatives_indices.size() == 4, LOG);
    assert_true(true_negatives_indices[0] == 0, LOG);
    assert_true(true_negatives_indices[1] == 1, LOG);
    assert_true(true_negatives_indices[2] == 2, LOG);
    assert_true(true_negatives_indices[3] == 3, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(0.0);
    targets(2, 0) = type(1);
    targets(3, 0) = type(0.0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_negatives_indices = testing_analysis.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    const Tensor<bool, 0> not_empty = true_negatives_indices.any();

    assert_true(!not_empty(0), LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0.0);
    targets(1, 0) = type(0.0);
    targets(2, 0) = type(1);
    targets(3, 0) = type(0.0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0.0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_negatives_indices = testing_analysis.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    assert_true(true_negatives_indices.size() == 1, LOG);
    assert_true(true_negatives_indices[0] == 0, LOG);
}


void TestingAnalysisTest::test_calculate_multiple_classification_rates()
{
    cout << "test_calculate_multiple_classification_rates\n";

    Tensor<Index, 1> testing_indices;

    Tensor<Tensor<Index,1>, 2> multiple_classification_rates;

    // Test

    targets.resize(9, 3);

    targets(0,0) = type(1); targets(0,1) = type(0); targets(0,2) = type(0);
    targets(1,0) = type(0); targets(1,1) = type(1); targets(1,2) = type(0);
    targets(2,0) = type(0); targets(2,1) = type(0); targets(2,2) = type(1);
    targets(3,0) = type(1); targets(3,1) = type(0); targets(3,2) = type(0);
    targets(4,0) = type(0); targets(4,1) = type(1); targets(4,2) = type(0);
    targets(5,0) = type(0); targets(5,1) = type(0); targets(5,2) = type(1);
    targets(6,0) = type(1); targets(6,1) = type(0); targets(6,2) = type(0);
    targets(7,0) = type(0); targets(7,1) = type(1); targets(7,2) = type(0);
    targets(8,0) = type(0); targets(8,1) = type(0); targets(8,2) = type(1);

    outputs.resize(9, 3);

    outputs(0,0) = type(1); outputs(0,1) = type(0); outputs(0,2) = type(0);
    outputs(1,0) = type(0); outputs(1,1) = type(1); outputs(1,2) = type(0);
    outputs(2,0) = type(0); outputs(2,1) = type(0); outputs(2,2) = type(1);
    outputs(3,0) = type(0); outputs(3,1) = type(1); outputs(3,2) = type(0);
    outputs(4,0) = type(1); outputs(4,1) = type(0); outputs(4,2) = type(0);
    outputs(5,0) = type(0); outputs(5,1) = type(1); outputs(5,2) = type(0);
    outputs(6,0) = type(0); outputs(6,1) = type(0); outputs(6,2) = type(1);
    outputs(7,0) = type(0); outputs(7,0) = type(0); outputs(7,2) = type(1);
    outputs(8,0) = type(1); outputs(8,1) = type(0); outputs(8,2) = type(0);

    testing_indices.resize(9);
    testing_indices.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8});

    multiple_classification_rates = testing_analysis.calculate_multiple_classification_rates(targets, outputs, testing_indices);

    assert_true(multiple_classification_rates(0,0)(0) == 0, LOG);
    assert_true(multiple_classification_rates(0,1)(0) == 3, LOG);
    assert_true(multiple_classification_rates(0,2)(0) == 6, LOG);
    assert_true(multiple_classification_rates(1,0)(0) == 4, LOG);
    assert_true(multiple_classification_rates(1,1)(0) == 1, LOG);
    assert_true(multiple_classification_rates(1,2)(0) == 7, LOG);
    assert_true(multiple_classification_rates(2,0)(0) == 8, LOG);
    assert_true(multiple_classification_rates(2,1)(0) == 5, LOG);
    assert_true(multiple_classification_rates(2,2)(0) == 2, LOG);
}


void TestingAnalysisTest::run_test_case()
{
    cout << "Running testing analysis test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Error data methods

    test_calculate_error_data();
    test_calculate_percentage_error_data();
    test_calculate_error_data_descriptives();
    test_calculate_absolute_errors_descriptives();
    test_calculate_percentage_errors_descriptives();
    test_calculate_error_data_histograms();
    test_calculate_maximal_errors();

    // Linear regression analysis methodsta

    test_linear_regression();
    test_save();
    test_perform_linear_regression();

    // Binary classification test methods

    test_calculate_binary_classification_test();

    // Confusion matrix methods

    test_calculate_confusion();

    // ROC curve methods

    test_calculate_Wilcoxon_parameter();
    test_calculate_roc_curve();
    test_calculate_area_under_curve();
    test_calculate_optimal_threshold();

    // Lift chart methods

    test_calculate_cumulative_gain();
    test_calculate_lift_chart();

    // Calibration plot

    test_calculate_calibration_plot();

    // Binary classification rates

    test_calculate_true_positive_samples();
    test_calculate_false_positive_samples();
    test_calculate_false_negative_samples();
    test_calculate_true_negative_samples();


    // Multiple classification rates

    test_calculate_multiple_classification_rates();

    cout << "End of testing analysis test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
