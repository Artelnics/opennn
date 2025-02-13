#include "pch.h"
#include "../opennn/testing_analysis.h"

TEST(TestingAnalysis, DefaultConstructor)
{
//    TestingAnalysis testing_analysis(&neural_network, &data_set);

//    EXPECT_EQ(testing_analysis.get_neural_network());

//    EXPECT_EQ(testing_analysis.get_data_set());
}


TEST(TestingAnalysis, ErrorData)
{
    /*
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index targets_number = 1;

    DataSet data_set(samples_number, { inputs_number }, { targets_number });
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { targets_number });
    neural_network.set_parameters_constant(type(0));

    TestingAnalysis testing_analysis(&neural_network, &data_set);
/*
    Tensor<type, 3> error_data = testing_analysis.calculate_error_data();

    EXPECT_EQ(error_data.size(), 3);
    EXPECT_EQ(error_data.dimension(0), 1);
    EXPECT_EQ(error_data.dimension(1), 3);
    //EXPECT_EQ(static_cast<double>(error_data(0, 0, 0)), 0.0);
*/
}


TEST(TestingAnalysis, PercentageErrorData)
{
/*
    Tensor<type, 2> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_percentage_error_data();

    EXPECT_EQ(error_data.size() == 1);
    EXPECT_EQ(error_data.dimension(1) == 1);
    EXPECT_EQ(static_cast<double>(error_data(0,0)) == 0.0);
    */
}


TEST(TestingAnalysis, AbsoluteErrorDescriptives)
{
/*
    Tensor<Descriptives, 1> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_absolute_errors_descriptives();

    EXPECT_EQ(error_data.size() == 1);
    EXPECT_EQ(static_cast<double>(error_data[0].minimum) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data[0].maximum) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data[0].mean) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data[0].standard_deviation) == 0.0);
*/
}


TEST(TestingAnalysis, PercentageErrorDescriptives)
{
/*
    Tensor<Descriptives, 1> error_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data = testing_analysis.calculate_percentage_errors_descriptives();

    EXPECT_EQ(error_data.size() == 1);
    EXPECT_EQ(static_cast<double>(error_data[0].standard_deviation) == 0.0);
*/
}


TEST(TestingAnalysis, ErrorDataDescriptives)
{
/*
    Tensor<Tensor<Descriptives, 1>, 1> error_data_statistics;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data_statistics = testing_analysis.calculate_error_data_descriptives();

    EXPECT_EQ(error_data_statistics.size() == 1);
    EXPECT_EQ(error_data_statistics[0].size() == 3);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][0].minimum) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][0].maximum) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][0].mean) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][0].standard_deviation) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][2].minimum) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][2].maximum) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][2].mean) == 0.0);
    EXPECT_EQ(static_cast<double>(error_data_statistics[0][2].standard_deviation) == 0.0);
*/
}


TEST(TestingAnalysis, ErrorDataHistograms)
{
/*
    Tensor<Histogram, 1> error_data_histograms;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    error_data_histograms = testing_analysis.calculate_error_data_histograms();

    EXPECT_EQ(error_data_histograms.size() == 1);
    EXPECT_EQ(error_data_histograms[0].get_bins_number() == 10);
*/
}


TEST(TestingAnalysis, MaximalErrors)
{
/*
    Tensor<Tensor<Index, 1>, 1> maximal_errors;

    samples_number = 4;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    maximal_errors = testing_analysis.calculate_maximal_errors(2);

    EXPECT_EQ(maximal_errors.rank() == 1);
    EXPECT_EQ(maximal_errors[0](0) == 0 );
*/
}


TEST(TestingAnalysis, LinearRegression)
{
/*
    Index neurons_number;

    Tensor<Correlation, 1> linear_correlation;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;
    neurons_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Testing);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    linear_correlation = testing_analysis.linear_correlation();

    EXPECT_EQ(linear_correlation.size() == 1);
    EXPECT_EQ(isnan(linear_correlation(0).a));
    EXPECT_EQ(isnan(linear_correlation(0).b));
    EXPECT_EQ(isnan(linear_correlation(0).r));

    // DataSet

    samples_number = 1;
    inputs_number = 1;
    neurons_number = 2;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_data_constant(type(0));

    data_set.set(DataSet::SampleUse::Testing);

    // Neural Network

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    // Testing Analysis

    Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> goodness_of_fit_analysis = testing_analysis.perform_goodness_of_fit_analysis();

    EXPECT_EQ(goodness_of_fit_analysis.size() == 1 );
    EXPECT_EQ(goodness_of_fit_analysis[0].determination - type(1) < type(NUMERIC_LIMITS_MIN));
*/
}


TEST(TestingAnalysis, Confusion)
{
/*
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

    EXPECT_EQ(sum(0) == 4 + 12);

    EXPECT_EQ(confusion.dimension(0) == 4);
    EXPECT_EQ(confusion.dimension(1) == 4);
    EXPECT_EQ(confusion(0,0) == 1);
    EXPECT_EQ(confusion(1,1) == 2);
    EXPECT_EQ(confusion(2,2) == 1);

    EXPECT_EQ(confusion(0,3) == confusion(0,0) + confusion(0,1) + confusion(0,2));
    EXPECT_EQ(confusion(1,3) == confusion(1,0) + confusion(1,1) + confusion(1,2));
    EXPECT_EQ(confusion(2,3) == confusion(2,0) + confusion(2,1) + confusion(2,2));

    EXPECT_EQ(confusion(3,0) == confusion(0,0) + confusion(1,0) + confusion(2,0));
    EXPECT_EQ(confusion(3,1) == confusion(0,1) + confusion(1,1) + confusion(2,1));
    EXPECT_EQ(confusion(3,2) == confusion(0,2) + confusion(1,2) + confusion(2,2));

    EXPECT_EQ(confusion(3,3) == 4);
*/
}


TEST(TestingAnalysis, BinaryClassificationTests)
{
/*
    // DataSet

    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_data_constant(type(0));

    data_set.set(DataSet::SampleUse::Testing);

    // Neural Network

    neural_network.set(NeuralNetwork::ModelType::Classification, {1}, {1}, {1});
    neural_network.set_parameters_constant(type(0));

    // Testing Analysis

   Tensor<type, 1> binary = testing_analysis.calculate_binary_classification_tests();

   EXPECT_EQ(binary.size() == 15 );

   EXPECT_EQ(binary[0] == 0 );
   EXPECT_EQ(binary[1] == 1 );
   EXPECT_EQ(binary[2] == 0 );
   EXPECT_EQ(binary[3] == 0 );
   EXPECT_EQ(binary[4] == 0 );
   EXPECT_EQ(binary[5] == 0 );
   EXPECT_EQ(binary[6] == 0 );
   EXPECT_EQ(binary[7] == 0 );
   EXPECT_EQ(binary[8] == 1 );
   EXPECT_EQ(binary[9] == 1 );
   EXPECT_EQ(binary[10] == 0 );
   EXPECT_EQ(binary[11] == 0 );
   EXPECT_EQ(binary[12] == 0 );
   EXPECT_EQ(binary[13] == -1 );
   EXPECT_EQ(binary[14] == -1 );
*/
}


TEST(TestingAnalysis, RocCurve)
{
/*
    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<type, 2> roc_curve;

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

    EXPECT_EQ(roc_curve.dimension(1) == 3);
    EXPECT_EQ(roc_curve.dimension(0) == 201);

    EXPECT_EQ(roc_curve(0, 0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(0, 1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(1, 0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(1, 1) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(2, 0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(2, 1) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(3, 0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(3, 1) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(4, 0) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(4, 1) - type(1) < type(NUMERIC_LIMITS_MIN));

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0);
    targets(1,0) = type(1);
    targets(2,0) = type(1);
    targets(3,0) = type(0);

    outputs.resize(4,1);

    outputs(0,0) = type(0.12);
    outputs(1,0) = type(0.78);
    outputs(2,0) = type(0.84);
    outputs(3,0) = type(0.99);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    EXPECT_EQ(roc_curve.dimension(1) == 3);
    EXPECT_EQ(roc_curve.dimension(0) == 201);

    EXPECT_EQ(roc_curve(0, 0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(0, 1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(1, 0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(1, 1) - type(0.5) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(2, 0) - type(0.5) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(2, 1) - type(0.5) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(3, 0) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(3, 1) - type(0.5) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(4, 0) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(roc_curve(4, 1) - type(1) < type(NUMERIC_LIMITS_MIN));
*/
}


TEST(TestingAnalysis, AreaUnderCurve)
{
/*
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

    EXPECT_EQ(area_under_curve - type(1) < type(NUMERIC_LIMITS_MIN));

    targets.resize(4,1);

    targets(0,0) = type(0);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(0);
    outputs(1,0) = type(1);
    outputs(2,0) = type(0);
    outputs(3,0) = type(1);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

    EXPECT_EQ(area_under_curve - type(0.5) < type(NUMERIC_LIMITS_MIN));

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(0.78);
    outputs(1,0) = type(0.84);
    outputs(2,0) = type(0.12);
    outputs(3,0) = type(0.99);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

    EXPECT_EQ(area_under_curve - type(0.5) < type(NUMERIC_LIMITS_MIN));

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(1);
    outputs(1,0) = type(1);
    outputs(2,0) = type(0);
    outputs(3,0) = type(0);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);

    EXPECT_EQ(area_under_curve < type(NUMERIC_LIMITS_MIN));
*/
}


TEST(TestingAnalysis, OptimalThreshold)
{
/*
    type optimal_threshold;

    Tensor<type, 2> roc_curve;

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

    optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

    EXPECT_EQ(optimal_threshold - type(1) < type(NUMERIC_LIMITS_MIN));

    // Test

    targets.resize(4,1);

    targets(0,0) = type(0);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(1);

    outputs.resize(4,1);

    outputs(0,0) = type(1);
    outputs(1,0) = type(1);
    outputs(2,0) = type(0);
    outputs(3,0) = type(0);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

    EXPECT_EQ(optimal_threshold - type(1) < type(NUMERIC_LIMITS_MIN));

    // Test

    targets.resize(5,1);

    targets(0,0) = type(0);
    targets(1,0) = type(1);
    targets(2,0) = type(0);
    targets(3,0) = type(1);
    targets(4,0) = type(0);

    outputs.resize(5,1);

    outputs(0,0) = type(0.33);
    outputs(1,0) = type(0.14);
    outputs(2,0) = type(0.12);
    outputs(3,0) = type(0.62);
    outputs(4,0) = type(0.85);

    roc_curve = testing_analysis.calculate_roc_curve(targets, outputs);

    optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

    EXPECT_EQ(optimal_threshold - type(0.62) < type(NUMERIC_LIMITS_MIN));
*/
}


TEST(TestingAnalysis, CumulativeGain)
{
/*
    targets.resize(4,1);

    targets(0,0) = type(1);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(0);

    outputs.resize(4,1);

    outputs(0,0) = type(0.67);
    outputs(1,0) = type(0.98);
    outputs(2,0) = type(0.78);
    outputs(3,0) = type(0.45);

    Tensor<type, 2> cumulative_gain = testing_analysis.calculate_cumulative_gain(targets, outputs);

    EXPECT_EQ(cumulative_gain.dimension(1) == 2);
    EXPECT_EQ(cumulative_gain.dimension(0) == 21);
    EXPECT_EQ(cumulative_gain(0, 0) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(cumulative_gain(0, 1) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(cumulative_gain(20, 0) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(cumulative_gain(20, 1) - type(1) < type(NUMERIC_LIMITS_MIN));
*/
}


TEST(TestingAnalysis, LiftChart)
{
/*
    Tensor<type, 2> cumulative_gain;

    Tensor<type, 2> lift_chart;

    // Test

    targets.resize(4,1);

    targets(0,0) = type(1);
    targets(1,0) = type(0);
    targets(2,0) = type(1);
    targets(3,0) = type(0);

    outputs.resize(4,1);

    outputs(0,0) = type(0.67);
    outputs(1,0) = type(0.87);
    outputs(2,0) = type(0.99);
    outputs(3,0) = type(0.88);

    cumulative_gain = testing_analysis.calculate_cumulative_gain(targets, outputs);

    lift_chart = testing_analysis.calculate_lift_chart(cumulative_gain);

    EXPECT_EQ(lift_chart.dimension(1) == cumulative_gain.dimension(1));
    EXPECT_EQ(lift_chart.dimension(0) == cumulative_gain.dimension(0));
*/
}


TEST(TestingAnalysis, CalibrationPlot)
{
/*
    Tensor<type, 2> calibration_plot;

    // Test

    targets.resize(10, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(0);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);
    targets(4, 0) = type(1);
    targets(5, 0) = type(1);
    targets(6, 0) = type(1);
    targets(7, 0) = type(0);
    targets(8, 0) = type(1);
    targets(9, 0) = type(0);

    outputs.resize(10, 1);

    outputs(0, 0) = type(0.09);
    outputs(1, 0) = type(0.19);
    outputs(2, 0) = type(0.29);
    outputs(3, 0) = type(0.39);
    outputs(4, 0) = type(0.49);
    outputs(5, 0) = type(0.59);
    outputs(6, 0) = type(0.58);
    outputs(7, 0) = type(0.79);
    outputs(8, 0) = type(0.89);
    outputs(9, 0) = type(0.99);

    calibration_plot = testing_analysis.calculate_calibration_plot(targets, outputs);

    EXPECT_EQ(calibration_plot.dimension(1) == 2);
    EXPECT_EQ(calibration_plot.dimension(0) == 11);
*/
}

/*
void TestingAnalysisTest::test_calculate_true_positive_samples()
{
    Tensor<Index, 1> true_positives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});

    const type threshold = type(0.5);

    true_positives_indices = testing_analysis.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_positives_indices.size() == 1);
    EXPECT_EQ(true_positives_indices[0] == 1);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(0);
    targets(2, 0) = type(0);
    targets(3, 0) = type(0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_positives_indices = testing_analysis.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    const Tensor<bool, 0> not_empty = true_positives_indices.any();

    EXPECT_EQ(!not_empty(0));

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

    EXPECT_EQ(true_positives_indices.size() == 4);
    EXPECT_EQ(true_positives_indices[0] == 0);
    EXPECT_EQ(true_positives_indices[1] == 1);
    EXPECT_EQ(true_positives_indices[2] == 2);
    EXPECT_EQ(true_positives_indices[3] == 3);
}


void TestingAnalysisTest::test_calculate_false_positive_samples()
{
    Tensor<Index, 1> false_positives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = type(0.5);

    false_positives_indices = testing_analysis.calculate_false_positive_samples(targets, outputs,testing_indices, threshold);

    EXPECT_EQ(false_positives_indices.size() == 1);
    EXPECT_EQ(false_positives_indices[0] == 2);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(0);
    targets(2, 0) = type(0);
    targets(3, 0) = type(0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    false_positives_indices = testing_analysis.calculate_false_positive_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_positives_indices.size() == 4);
    EXPECT_EQ(false_positives_indices[0] == 0);
    EXPECT_EQ(false_positives_indices[1] == 1);
    EXPECT_EQ(false_positives_indices[2] == 2);
    EXPECT_EQ(false_positives_indices[3] == 3);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(0);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    false_positives_indices = testing_analysis.calculate_false_positive_samples(targets, outputs,testing_indices, threshold);

    const Tensor<bool, 0> not_empty = false_positives_indices.any();

    EXPECT_EQ(!not_empty(0));

    EXPECT_EQ(false_positives_indices.size() == 0);
}


void TestingAnalysisTest::test_calculate_false_negative_samples()
{
    Tensor<Index, 1> false_negatives_indices;

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = type(0.5);

    false_negatives_indices = testing_analysis.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_negatives_indices.size() == 1);
    EXPECT_EQ(false_negatives_indices[0] == 3);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(0);
    targets(3, 0) = type(0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(1);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(0);
    outputs(3, 0) = type(0);

    false_negatives_indices = testing_analysis.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_negatives_indices.size() == 0);

    const Tensor<bool, 0> not_empty = false_negatives_indices.any();

    EXPECT_EQ(!not_empty(0));

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(1);
    targets(2, 0) = type(1);
    targets(3, 0) = type(1);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(0);
    outputs(2, 0) = type(0);
    outputs(3, 0) = type(0);

    false_negatives_indices = testing_analysis.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_negatives_indices.size() == 4);
    EXPECT_EQ(false_negatives_indices[0] == 0);
    EXPECT_EQ(false_negatives_indices[1] == 1);
    EXPECT_EQ(false_negatives_indices[2] == 2);
    EXPECT_EQ(false_negatives_indices[3] == 3);
}


void TestingAnalysisTest::test_calculate_true_negative_samples()
{
    Tensor<Index, 1> true_negatives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(0);
    targets(2, 0) = type(0);
    targets(3, 0) = type(0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(0);
    outputs(2, 0) = type(0);
    outputs(3, 0) = type(0);

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = type(0.5);

    true_negatives_indices = testing_analysis.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_negatives_indices.size() == 4);
    EXPECT_EQ(true_negatives_indices[0] == 0);
    EXPECT_EQ(true_negatives_indices[1] == 1);
    EXPECT_EQ(true_negatives_indices[2] == 2);
    EXPECT_EQ(true_negatives_indices[3] == 3);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(1);
    targets(1, 0) = type(0);
    targets(2, 0) = type(1);
    targets(3, 0) = type(0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_negatives_indices = testing_analysis.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    const Tensor<bool, 0> not_empty = true_negatives_indices.any();

    EXPECT_EQ(!not_empty(0));

    // Test

    targets.resize(4, 1);

    targets(0, 0) = type(0);
    targets(1, 0) = type(0);
    targets(2, 0) = type(1);
    targets(3, 0) = type(0);

    outputs.resize(4, 1);

    outputs(0, 0) = type(0);
    outputs(1, 0) = type(1);
    outputs(2, 0) = type(1);
    outputs(3, 0) = type(1);

    true_negatives_indices = testing_analysis.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_negatives_indices.size() == 1);
    EXPECT_EQ(true_negatives_indices[0] == 0);
}


void TestingAnalysisTest::test_calculate_multiple_classification_rates()
{
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

    EXPECT_EQ(multiple_classification_rates.size() == 9);

    EXPECT_EQ(multiple_classification_rates(0,0)(0) == 0);
    EXPECT_EQ(multiple_classification_rates(0,1)(0) == 3);
    EXPECT_EQ(multiple_classification_rates(0,2)(0) == 6);
    EXPECT_EQ(multiple_classification_rates(1,0)(0) == 4);
    EXPECT_EQ(multiple_classification_rates(1,1)(0) == 1);
    EXPECT_EQ(multiple_classification_rates(1,2)(0) == 7);
    EXPECT_EQ(multiple_classification_rates(2,0)(0) == 8);
    EXPECT_EQ(multiple_classification_rates(2,1)(0) == 5);
    EXPECT_EQ(multiple_classification_rates(2,2)(0) == 2);

}

}

*/

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
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
