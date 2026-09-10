#include "tests/pch.h"
#include "gtest/gtest.h"

#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/evaluation/evaluation.h"

using namespace opennn;

TEST(Evaluation, ErrorData)
{
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index targets_number = 1;

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_constant(type(0));
    dataset.set_sample_roles("Testing");

    ApproximationNetwork network({ inputs_number }, {}, { targets_number });
    network.set_parameters_random();

    Evaluation evaluation(&network, &dataset);

    Tensor3 error_data = evaluation.calculate_error_data();

    EXPECT_EQ(error_data.size(), 3);
    EXPECT_EQ(error_data.dimension(0), 1);
    EXPECT_EQ(error_data.dimension(1), 3);
}

TEST(Evaluation, PercentageErrorData)
{
    MatrixR error_data;

    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index targets_number = 1;

    TabularDataset dataset;
    dataset.set(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_constant(type(0));
    dataset.set_sample_roles("Testing");

    ApproximationNetwork network({inputs_number}, {}, {targets_number});
    network.set_parameters_random();

    Evaluation evaluation(&network, &dataset);
    error_data = evaluation.calculate_percentage_error_data();

    EXPECT_EQ(error_data.size(), 1);
    EXPECT_EQ(error_data.cols(), 1);
}

TEST(Evaluation, ErrorDataDescriptives)
{
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index targets_number = 1;

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_constant(type(0));
    dataset.set_sample_roles("Testing");

    ApproximationNetwork network({ inputs_number }, {}, { targets_number });
    network.set_parameters_random();

    Evaluation evaluation(&network, &dataset);

    const vector<vector<Descriptives>> error_data_descriptives =
        evaluation.calculate_error_data_descriptives();

    ASSERT_EQ(ssize(error_data_descriptives), targets_number);
    ASSERT_FALSE(error_data_descriptives[0].empty());
    EXPECT_NEAR(error_data_descriptives[0][0].standard_deviation, type(0), 1e-5);
}

TEST(Evaluation, ErrorDataHistograms)
{
    vector<Histogram> error_data_histograms;

    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index targets_number = 1;

    TabularDataset dataset;
    dataset.set(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_constant(type(0));
    dataset.set_sample_roles("Testing");

    ApproximationNetwork network({inputs_number}, {}, {targets_number});
    network.set_parameters_random();

    Evaluation evaluation(&network, &dataset);
    error_data_histograms = evaluation.calculate_error_data_histograms();

    EXPECT_EQ(error_data_histograms.size(), 1);
}

TEST(Evaluation, ReconstructionErrors)
{
    MatrixR targets(2, 2);
    targets << 1.0f, 2.0f,
               3.0f, 4.0f;

    MatrixR reconstructions(2, 2);
    reconstructions << 1.0f, 4.0f,
                       2.0f, 4.0f;

    Evaluation evaluation;
    const VectorR errors = evaluation.calculate_reconstruction_errors(targets, reconstructions);

    ASSERT_EQ(errors.size(), 2);
    EXPECT_FLOAT_EQ(errors(0), 1.0f);
    EXPECT_FLOAT_EQ(errors(1), 0.5f);
}

TEST(Evaluation, ReconstructionErrorStatisticsUsePopulationDeviation)
{
    VectorR errors(4);
    errors << 1.0f, 2.0f, 3.0f, 4.0f;

    Evaluation evaluation;
    const Evaluation::ReconstructionErrorStatistics statistics =
        evaluation.calculate_reconstruction_error_statistics(errors);

    EXPECT_FLOAT_EQ(statistics.minimum, 1.0f);
    EXPECT_FLOAT_EQ(statistics.maximum, 4.0f);
    EXPECT_FLOAT_EQ(statistics.mean, 2.5f);
    EXPECT_NEAR(statistics.population_standard_deviation, sqrt(1.25f), 1.0e-6f);
    EXPECT_NEAR(evaluation.calculate_anomaly_threshold(statistics),
                2.5f + sqrt(1.25f),
                1.0e-6f);
}

TEST(Evaluation, AnomalyPredictionIncludesThresholdEquality)
{
    VectorR errors(3);
    errors << 0.9f, 1.0f, 1.1f;

    Evaluation evaluation;
    const VectorI predictions = evaluation.calculate_anomaly_predictions(errors, 1.0f);

    ASSERT_EQ(predictions.size(), 3);
    EXPECT_EQ(predictions(0), 0);
    EXPECT_EQ(predictions(1), 1);
    EXPECT_EQ(predictions(2), 1);
}

TEST(Evaluation, BinaryClassificationTestsFromData)
{
    MatrixR targets(4, 1);
    targets << 1.0f, 1.0f, 0.0f, 0.0f;

    MatrixR predictions(4, 1);
    predictions << 1.0f, 0.0f, 1.0f, 0.0f;

    Evaluation evaluation;
    const VectorR tests = evaluation.calculate_binary_classification_tests(targets, predictions);

    EXPECT_FLOAT_EQ(tests(0), 0.5f);
    EXPECT_FLOAT_EQ(tests(2), 0.5f);
    EXPECT_FLOAT_EQ(tests(3), 0.5f);
    EXPECT_FLOAT_EQ(tests(4), 0.5f);
    EXPECT_FLOAT_EQ(tests(7), 0.5f);
}

TEST(Evaluation, Confusion)
{
    MatrixR actual(4, 3);
    actual << type(1), type(0), type(0),
        type(0), type(1), type(0),
        type(0), type(1), type(0),
        type(0), type(0), type(1);

    MatrixR predicted(4, 3);
    predicted << type(1), type(0), type(0),
        type(0), type(1), type(0),
        type(0), type(1), type(0),
        type(0), type(0), type(1);

    Evaluation evaluation;
    MatrixI confusion = evaluation.calculate_confusion(actual, predicted);

    type sum = confusion.sum();

    EXPECT_EQ(sum, 4 + 12);

    EXPECT_EQ(confusion.rows(), 4);
    EXPECT_EQ(confusion.cols(), 4);
    EXPECT_EQ(confusion(0,0), 1);
    EXPECT_EQ(confusion(1,1), 2);
    EXPECT_EQ(confusion(2,2), 1);

    EXPECT_EQ(confusion(0,3), confusion(0,0) + confusion(0,1) + confusion(0,2));
    EXPECT_EQ(confusion(1,3), confusion(1,0) + confusion(1,1) + confusion(1,2));
    EXPECT_EQ(confusion(2,3), confusion(2,0) + confusion(2,1) + confusion(2,2));

    EXPECT_EQ(confusion(3,0), confusion(0,0) + confusion(1,0) + confusion(2,0));
    EXPECT_EQ(confusion(3,1), confusion(0,1) + confusion(1,1) + confusion(2,1));
    EXPECT_EQ(confusion(3,2), confusion(0,2) + confusion(1,2) + confusion(2,2));

    EXPECT_EQ(confusion(3,3), 4);
}

TEST(Evaluation, BinaryClassificationTests)
{
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index targets_number= 1;

    TabularDataset dataset;

    dataset.set(samples_number, {inputs_number}, {targets_number});

    dataset.set_data_constant(type(0));

    dataset.set_sample_roles("Testing");

    ClassificationNetwork network({1}, {1}, {1});

    Evaluation evaluation(&network, &dataset);

    VectorR binary = evaluation.calculate_binary_classification_tests();

    EXPECT_EQ(binary.size(), 15 );

    for(Index i = 0; i < binary.size(); i++)
        EXPECT_TRUE(isfinite(binary[i]) || binary[i] == type(-1));

}

TEST(Evaluation, PrintsMultipleClassificationTests)
{
    TabularDataset dataset(3, {1}, {3});
    MatrixR data(3, 4);
    data << 0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f;
    dataset.set_data(data);
    dataset.set_sample_roles("Testing");

    ClassificationNetwork network({1}, {}, {3});
    network.set_parameters(VectorR::Zero(
        network.get_parameters_buffer_size()));

    Evaluation evaluation(&network, &dataset);

    testing::internal::CaptureStdout();
    evaluation.print_multiple_classification_tests();
    const string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("Classification accuracy : 0.333333"), string::npos);
    EXPECT_NE(output.find("Confusion matrix"), string::npos);
}

TEST(Evaluation, RejectsBinaryNetworkForMultipleClassificationTests)
{
    TabularDataset dataset(1, {1}, {1});
    dataset.set_data_constant(0.0f);
    dataset.set_sample_roles("Testing");

    ClassificationNetwork network({1}, {}, {1});
    Evaluation evaluation(&network, &dataset);

    EXPECT_THROW(evaluation.print_multiple_classification_tests(), runtime_error);
}

TEST(Evaluation, RocCurve)
{
    Evaluation evaluation;
    MatrixR targets(4, 1), scores(4, 1);
    targets << 0, 0, 1, 1;
    scores << 0, 0, 1, 1;
    const MatrixR curve = evaluation.calculate_roc_curve(targets, scores);
    ASSERT_EQ(curve.rows(), 3);
    ASSERT_EQ(curve.cols(), 3);
    EXPECT_FLOAT_EQ(curve(0, 0), 1);
    EXPECT_FLOAT_EQ(curve(0, 1), 1);
    EXPECT_FLOAT_EQ(curve(1, 0), 0);
    EXPECT_FLOAT_EQ(curve(1, 1), 1);
    EXPECT_FLOAT_EQ(curve(1, 2), 1);
    EXPECT_FLOAT_EQ(curve(2, 0), 0);
    EXPECT_FLOAT_EQ(curve(2, 1), 0);
    EXPECT_GT(curve(2, 2), scores.maxCoeff());
}

TEST(Evaluation, RocExactScoresAndTies)
{
    Evaluation evaluation;
    MatrixR targets(4, 1), scores(4, 1);
    targets << 0, 0, 1, 1;
    for(const VectorR& values : vector<VectorR>{
            (VectorR(4) << 0, .001f, .002f, 1).finished(),
            (VectorR(4) << -10, -9, 20, 1000000).finished(),
            (VectorR(4) << 0, 0, 0, 0).finished(),
            (VectorR(4) << 1, 1, 0, 0).finished(),
            (VectorR(4) << 0, .5f, .5f, 1).finished()})
    {
        scores.col(0) = values;
        const MatrixR curve = evaluation.calculate_roc_curve(targets, scores);
        double wins = 0;
        for(Index p = 2; p < 4; ++p)
            for(Index n = 0; n < 2; ++n)
                wins += scores(p, 0) > scores(n, 0) ? 1.0
                      : scores(p, 0) == scores(n, 0) ? 0.5 : 0.0;
        EXPECT_NEAR(evaluation.calculate_area_under_curve(curve), wins / 4, 1e-7);
        for(Index row = 0; row < curve.rows(); ++row)
        {
            Index tp = 0, fp = 0;
            for(Index i = 0; i < 4; ++i)
                if(scores(i, 0) >= curve(row, 2))
                {
                    if(targets(i, 0) >= .5f) ++tp;
                    else ++fp;
                }
            EXPECT_FLOAT_EQ(curve(row, 0), float(fp) / 2);
            EXPECT_FLOAT_EQ(curve(row, 1), float(tp) / 2);
        }
        MatrixR reversed_targets = targets.colwise().reverse();
        MatrixR reversed_scores = scores.colwise().reverse();
        EXPECT_FLOAT_EQ(evaluation.calculate_area_under_curve(curve),
            evaluation.calculate_area_under_curve(
                evaluation.calculate_roc_curve(reversed_targets, reversed_scores)));
    }
}

TEST(Evaluation, RocRejectsInvalidData)
{
    Evaluation evaluation;
    MatrixR targets(2, 1), scores(2, 1);
    targets << 0, 1;
    scores << 0, 1;
    EXPECT_THROW(evaluation.calculate_roc_curve(MatrixR(0, 1), MatrixR(0, 1)), runtime_error);
    EXPECT_THROW(evaluation.calculate_roc_curve(targets, MatrixR::Zero(3, 1)), runtime_error);
    EXPECT_THROW(evaluation.calculate_roc_curve(targets, MatrixR::Zero(2, 2)), runtime_error);
    EXPECT_THROW(evaluation.calculate_roc_curve(MatrixR::Zero(2, 1), scores), runtime_error);
    scores(0, 0) = numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(evaluation.calculate_roc_curve(targets, scores), runtime_error);
    scores(0, 0) = numeric_limits<float>::infinity();
    EXPECT_THROW(evaluation.calculate_roc_curve(targets, scores), runtime_error);
}

TEST(Evaluation, AreaUnderCurve)
{
    MatrixR roc_curve;
    MatrixR targets;
    MatrixR outputs;

    type area_under_curve;

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

    Evaluation evaluation;

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    area_under_curve = evaluation.calculate_area_under_curve(roc_curve);

    EXPECT_NEAR(area_under_curve, type(1), type(EPSILON));

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

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    area_under_curve = evaluation.calculate_area_under_curve(roc_curve);

    EXPECT_NEAR(area_under_curve, type(0.5), type(EPSILON));

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

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    area_under_curve = evaluation.calculate_area_under_curve(roc_curve);

    EXPECT_NEAR(area_under_curve, type(0.5), type(EPSILON));

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

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    area_under_curve = evaluation.calculate_area_under_curve(roc_curve);

    EXPECT_LT(area_under_curve, type(EPSILON));
}

TEST(Evaluation, OptimalThreshold)
{
    type optimal_threshold;

    MatrixR roc_curve;
    MatrixR targets;
    MatrixR outputs;

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

    Evaluation evaluation;

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    optimal_threshold = evaluation.calculate_optimal_threshold(roc_curve);

    EXPECT_LT(optimal_threshold - type(1), type(EPSILON));

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

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    optimal_threshold = evaluation.calculate_optimal_threshold(roc_curve);

    EXPECT_LT(optimal_threshold - type(1), type(EPSILON));

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

    roc_curve = evaluation.calculate_roc_curve(targets, outputs);

    optimal_threshold = evaluation.calculate_optimal_threshold(roc_curve);

    EXPECT_LT(optimal_threshold - type(0.62), type(EPSILON));
}

TEST(Evaluation, TruePositiveSamples)
{
    vector<Index> true_positives_indices;
    MatrixR targets;
    MatrixR outputs;

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

    vector<Index> testing_indices = {0, 1, 2, 3};

    const type threshold = type(0.5);

    Evaluation evaluation;

    true_positives_indices = evaluation.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_positives_indices.size(), 1);
    EXPECT_EQ(true_positives_indices[0], 1);

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

    true_positives_indices = evaluation.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    const bool not_empty = !true_positives_indices.empty();

    EXPECT_EQ(not_empty, false);

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

    true_positives_indices = evaluation.calculate_true_positive_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_positives_indices.size(), 4);
    EXPECT_EQ(true_positives_indices[0], 0);
    EXPECT_EQ(true_positives_indices[1], 1);
    EXPECT_EQ(true_positives_indices[2], 2);
    EXPECT_EQ(true_positives_indices[3], 3);
}

TEST(Evaluation, FalsePositiveSamples)
{
    vector<Index> false_positives_indices;
    MatrixR targets;
    MatrixR outputs;

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

    vector<Index> testing_indices = {0, 1, 2, 3};
    const type threshold = type(0.5);

    Evaluation evaluation;

    false_positives_indices = evaluation.calculate_false_positive_samples(targets, outputs,testing_indices, threshold);

    EXPECT_EQ(false_positives_indices.size(), 1);
    EXPECT_EQ(false_positives_indices[0], 2);

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

    false_positives_indices = evaluation.calculate_false_positive_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_positives_indices.size(), 4);
    EXPECT_EQ(false_positives_indices[0], 0);
    EXPECT_EQ(false_positives_indices[1], 1);
    EXPECT_EQ(false_positives_indices[2], 2);
    EXPECT_EQ(false_positives_indices[3], 3);

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

    false_positives_indices = evaluation.calculate_false_positive_samples(targets, outputs,testing_indices, threshold);

    const bool not_empty = !false_positives_indices.empty();

    EXPECT_EQ(not_empty, false);

    EXPECT_EQ(false_positives_indices.size(), 0);
}

TEST(Evaluation, FalseNegativeSamples)
{
    vector<Index> false_negatives_indices;
    MatrixR targets;
    MatrixR outputs;

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

    vector<Index> testing_indices = {0, 1, 2, 3};
    const type threshold = type(0.5);

    Evaluation evaluation;

    false_negatives_indices = evaluation.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_negatives_indices.size(), 1);
    EXPECT_EQ(false_negatives_indices[0], 3);

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

    false_negatives_indices = evaluation.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_negatives_indices.size(), 0);

    const bool not_empty = !false_negatives_indices.empty();

    EXPECT_EQ(not_empty, false);

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

    false_negatives_indices = evaluation.calculate_false_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(false_negatives_indices.size(), 4);
    EXPECT_EQ(false_negatives_indices[0], 0);
    EXPECT_EQ(false_negatives_indices[1], 1);
    EXPECT_EQ(false_negatives_indices[2], 2);
    EXPECT_EQ(false_negatives_indices[3], 3);
}

TEST(Evaluation, TrueNegativeSamples)
{
    vector<Index> true_negatives_indices;
    MatrixR targets;
    MatrixR outputs;

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

    vector<Index> testing_indices = {0, 1, 2, 3};
    const type threshold = type(0.5);

    Evaluation evaluation;

    true_negatives_indices = evaluation.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_negatives_indices.size(), 4);
    EXPECT_EQ(true_negatives_indices[0], 0);
    EXPECT_EQ(true_negatives_indices[1], 1);
    EXPECT_EQ(true_negatives_indices[2], 2);
    EXPECT_EQ(true_negatives_indices[3], 3);

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

    true_negatives_indices = evaluation.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    const bool not_empty = !true_negatives_indices.empty();

    EXPECT_EQ(not_empty, false);

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

    true_negatives_indices = evaluation.calculate_true_negative_samples(targets, outputs, testing_indices, threshold);

    EXPECT_EQ(true_negatives_indices.size(), 1);
    EXPECT_EQ(true_negatives_indices[0], 0);
}

TEST(Evaluation, MultipleClassificationRates)
{
    Evaluation evaluation;

    MatrixR targets(9, 3);
    targets << type(1), type(0), type(0),
               type(0), type(1), type(0),
               type(0), type(0), type(1),
               type(1), type(0), type(0),
               type(0), type(1), type(0),
               type(0), type(0), type(1),
               type(1), type(0), type(0),
               type(0), type(1), type(0),
               type(0), type(0), type(1);

    MatrixR outputs(9, 3);
    outputs << type(1), type(0), type(0),
               type(0), type(1), type(0),
               type(0), type(0), type(1),
               type(0), type(1), type(0),
               type(1), type(0), type(0),
               type(0), type(1), type(0),
               type(0), type(0), type(1),
               type(0), type(0), type(1),
               type(1), type(0), type(0);

    const vector<Index> testing_indices = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    const Tensor<VectorI, 2> rates =
        evaluation.calculate_multiple_classification_rates(targets, outputs, testing_indices);

    ASSERT_EQ(rates.dimension(0), 3);
    ASSERT_EQ(rates.dimension(1), 3);

    EXPECT_EQ(rates(0, 0)(0), 0);
    EXPECT_EQ(rates(0, 1)(0), 3);
    EXPECT_EQ(rates(0, 2)(0), 6);
    EXPECT_EQ(rates(1, 0)(0), 4);
    EXPECT_EQ(rates(1, 1)(0), 1);
    EXPECT_EQ(rates(1, 2)(0), 7);
    EXPECT_EQ(rates(2, 0)(0), 8);
    EXPECT_EQ(rates(2, 1)(0), 5);
    EXPECT_EQ(rates(2, 2)(0), 2);
}

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
