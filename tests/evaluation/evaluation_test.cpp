// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "tests/pch.h"
#include "gtest/gtest.h"
#include <array>

#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/network/layers/dense_layer.h"

using namespace opennn;

namespace
{

class CountingEvaluationDataset : public TabularDataset
{
public:
    using TabularDataset::TabularDataset;
    mutable vector<vector<Index>> target_batches;
    mutable vector<FillMode> target_modes;

    void fill_targets(const vector<Index>& samples, const vector<Index>& features,
                      float* destination, FillMode mode, ColumnContiguity contiguity) const override
    {
        target_batches.push_back(samples);
        target_modes.push_back(mode);
        TabularDataset::fill_targets(samples, features, destination, mode, contiguity);
    }
};

}

TEST(Evaluation, BatchedInferenceReadsTargetsOnceAndPreservesSampleOrderAndShape)
{
    const ScopeExit reset_configuration([] { Configuration::instance().set(Device::CPU, Type::FP32); });
    for (const auto [device_kind, precision] : {
             pair{Device::CPU, Type::FP32}, pair{Device::CUDA, Type::FP32}, pair{Device::CUDA, Type::BF16}})
    {
        if (device_kind == Device::CUDA && !device::has_cuda_device()) continue;
        if (precision == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        SCOPED_TRACE(device_kind == Device::CPU ? "CPU" : precision == Type::FP32 ? "CUDA FP32" : "CUDA BF16");
        Configuration::instance().set(device_kind, precision);
        CountingEvaluationDataset dataset(9, {2}, {2, 3});
        MatrixR data(9, 8);
        for (Index row = 0; row < data.rows(); ++row)
            for (Index column = 0; column < data.cols(); ++column)
                data(row, column) = float(row * 8 + column) * 0.125f;
        dataset.set_data(data);
        dataset.set_shape(VariableRole::Target, {2, 3});
        dataset.set_sample_roles(SampleRole::None);
        const vector<Index> selected{0, 2, 3, 4, 6, 7, 8};
        dataset.set_sample_roles(selected, "Testing");
        ASSERT_EQ(dataset.get_target_shape(), (Shape{2, 3}));

        Network network;
        network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{6}, "Identity"));
        network.compile();
        network.get_parameters_map().setLinSpaced(-0.25f, 0.25f);
        if (network.is_gpu()) network.copy_parameters_device();
        const MatrixR inputs = data(selected, Eigen::seqN(0, 2));
        const MatrixR expected_outputs = network.calculate_outputs(inputs);
        const MatrixR expected_targets = data(selected, Eigen::seqN(2, 6));

        Evaluation evaluation(&network, &dataset);
        evaluation.set_batch_size(3);
        const auto [targets, outputs] = evaluation.get_targets_and_outputs("Testing");
        ASSERT_EQ(targets.rows(), 7);
        ASSERT_EQ(targets.cols(), 6);
        EXPECT_EQ((targets - expected_targets).squaredNorm(), 0.0f);
        ASSERT_EQ(outputs.rows(), expected_outputs.rows());
        ASSERT_EQ(outputs.cols(), expected_outputs.cols());
        EXPECT_LE((outputs - expected_outputs).cwiseAbs().maxCoeff(), precision == Type::BF16 ? 0.03f : 1.0e-5f);
        EXPECT_EQ(dataset.target_batches, (vector<vector<Index>>{{0, 2, 3}, {4, 6, 7}, {8}}));
        EXPECT_EQ(dataset.target_modes, (vector<FillMode>(3, FillMode::Inference)));
    }
}

TEST(Evaluation, RegressionAnalysisRequiresNetworkDatasetAndTestingSamples)
{
    ApproximationNetwork network({1}, {}, {1});
    TabularDataset dataset(1, {1}, {1});
    dataset.set_sample_roles("Training");
    for (Evaluation evaluation : {Evaluation{}, Evaluation{&network}, Evaluation{&network, &dataset}})
    {
        EXPECT_THROW(evaluation.perform_goodness_of_fit_analysis(), runtime_error);
        EXPECT_THROW(evaluation.calculate_error_data(), runtime_error);
        EXPECT_THROW(evaluation.calculate_percentage_error_data(), runtime_error);
    }
}

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

TEST(Evaluation, PerfectReconstructionsAreNormal)
{
    Evaluation evaluation;
    const VectorR errors = VectorR::Zero(4);
    const auto statistics = evaluation.calculate_reconstruction_error_statistics(errors);
    const float threshold = evaluation.calculate_anomaly_threshold(statistics);
    EXPECT_GT(threshold, 0.0f);
    EXPECT_EQ(evaluation.calculate_anomaly_predictions(errors, threshold).sum(), 0);

    VectorR positive_error(1);
    positive_error << nextafter(0.0f, 1.0f);
    EXPECT_EQ(evaluation.calculate_anomaly_predictions(positive_error, threshold)(0), 1);
}

TEST(Evaluation, AnomalyThresholdRejectsInvalidNumbers)
{
    Evaluation evaluation;
    Evaluation::ReconstructionErrorStatistics statistics;
    statistics.mean = 1.0f;
    statistics.population_standard_deviation = 2.0f;
    const float infinity = numeric_limits<float>::infinity();
    const float nan = numeric_limits<float>::quiet_NaN();
    for(const float invalid : {nan, infinity, -infinity, -1.0f})
        EXPECT_THROW(evaluation.calculate_anomaly_threshold(statistics, invalid), runtime_error);
    EXPECT_FLOAT_EQ(evaluation.calculate_anomaly_threshold(statistics, 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(evaluation.calculate_anomaly_threshold(statistics, 2.0f), 5.0f);

    for(const float invalid : {nan, infinity, -infinity, -1.0f})
    {
        statistics.population_standard_deviation = invalid;
        EXPECT_THROW(evaluation.calculate_anomaly_threshold(statistics), runtime_error);
    }
    statistics.population_standard_deviation = 0.0f;
    EXPECT_THROW(evaluation.calculate_anomaly_threshold(statistics, infinity), runtime_error);
    for(const float invalid : {nan, infinity, -infinity})
    {
        statistics.mean = invalid;
        EXPECT_THROW(evaluation.calculate_anomaly_threshold(statistics), runtime_error);
    }

    statistics.mean = numeric_limits<float>::max();
    statistics.population_standard_deviation = numeric_limits<float>::max();
    EXPECT_THROW(evaluation.calculate_anomaly_threshold(statistics, 1.0f), runtime_error);
    statistics.mean = 0.0f;
    EXPECT_THROW(evaluation.calculate_anomaly_threshold(statistics, 2.0f), runtime_error);
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

TEST(Evaluation, BinaryMetricsPreserveAllFieldsAndDegenerateConventions)
{
    struct Case
    {
        std::array<Index, 4> counts; // TP, FP, FN, TN
        std::array<float, 15> expected;
    };
    const vector<Case> cases{
        {{0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1}},
        {{1, 0, 0, 1}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1}},
        {{2, 0, 0, 0}, {1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 2}, {1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0}},
        {{0, 2, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1}},
        {{3, 2, 1, 4}, {0.7f, 0.3f, 0.75f, 2.0f/3, 0.6f, 2.25f, 0.375f, 2.0f/3,
                         1.0f/3, 0.4f, 0.25f, 0.8f, 0.40824829f, 5.0f/12, 0.4f}}
    };
    Evaluation evaluation;
    for (const Case& test : cases)
    {
        const Index rows = accumulate(test.counts.begin(), test.counts.end(), Index(0));
        MatrixR targets(rows, 1), outputs(rows, 1);
        Index row = 0;
        for (size_t cell = 0; cell < test.counts.size(); ++cell)
            for (Index sample = 0; sample < test.counts[cell]; ++sample, ++row)
            {
                targets(row, 0) = cell == 0 || cell == 2 ? 1.0f : 0.0f;
                outputs(row, 0) = cell < 2 ? 1.0f : 0.0f;
            }
        const VectorR actual = evaluation.calculate_binary_classification_tests(targets, outputs);
        ASSERT_EQ(actual.size(), 15);
        for (Index metric = 0; metric < actual.size(); ++metric)
            EXPECT_NEAR(actual(metric), test.expected[size_t(metric)], 1.0e-6f) << "metric " << metric;
    }
}

TEST(Evaluation, BinaryRateAnalysisGroupsOriginalTestingIndicesInOneResult)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(8, {1}, {1});
    MatrixR data(8, 2);
    data << 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.49f,
            0.49f, 0.5f, 0.49f, 0.49f, 0.5f, 1.0f, 0.1f, 0.1f;
    dataset.set_data(data);
    dataset.set_sample_roles(SampleRole::None);
    dataset.set_sample_roles(vector<Index>{1, 3, 4, 5, 6, 7}, "Testing");

    Network network;
    auto identity = make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity");
    identity->set_use_bias(false);
    network.add_layer(std::move(identity));
    network.compile();
    network.get_parameters_map().setOnes();
    Evaluation evaluation(&network, &dataset);
    evaluation.set_batch_size(2);
    const auto rates = evaluation.calculate_binary_classification_rates();
    EXPECT_EQ(rates.true_positives_indices, (vector<Index>{1, 6}));
    EXPECT_EQ(rates.false_positives_indices, (vector<Index>{3}));
    EXPECT_EQ(rates.false_negatives_indices, (vector<Index>{4}));
    EXPECT_EQ(rates.true_negatives_indices, (vector<Index>{5, 7}));
}

TEST(Evaluation, ClassificationRejectsMismatchedShapesAndSampleIndices)
{
    Evaluation evaluation;
    const MatrixR binary = MatrixR::Zero(2, 1);
    EXPECT_THROW(evaluation.calculate_confusion(binary, MatrixR::Zero(1, 1)), runtime_error);
    EXPECT_THROW(evaluation.calculate_confusion(binary, MatrixR::Zero(2, 2)), runtime_error);
    EXPECT_THROW(evaluation.calculate_confusion(MatrixR(2, 0), MatrixR(2, 0)), runtime_error);
    EXPECT_THROW(evaluation.calculate_true_positive_samples(binary, binary, {1}, 0.5f), runtime_error);
    EXPECT_THROW(evaluation.calculate_multiple_classification_rates(MatrixR::Zero(2, 2),
                 MatrixR::Zero(2, 2), {1}), runtime_error);
    EXPECT_THROW(evaluation.calculate_multiple_classification_rates(MatrixR::Zero(2, 2),
                 MatrixR::Zero(1, 2), {1, 2}), runtime_error);
    EXPECT_THROW(evaluation.calculate_binary_classification_tests(MatrixR::Zero(2, 2),
                 MatrixR::Zero(2, 2)), runtime_error);
    EXPECT_THROW(evaluation.calculate_area_under_curve_confidence_limit(binary, MatrixR::Zero(1, 1)), runtime_error);
    EXPECT_THROW(evaluation.calculate_cumulative_gain(binary, MatrixR::Zero(1, 1)), runtime_error);
    EXPECT_THROW(evaluation.calculate_lift_chart(MatrixR(0, 2)), runtime_error);
    EXPECT_THROW(evaluation.calculate_lift_chart(MatrixR::Zero(2, 1)), runtime_error);
    EXPECT_THROW(evaluation.calculate_lift_chart(MatrixR::Zero(2, 3)), runtime_error);
    const MatrixI empty = evaluation.calculate_confusion(MatrixR(0, 1), MatrixR(0, 1));
    EXPECT_EQ(empty.rows(), 3);
    EXPECT_EQ(empty.sum(), 0);
}

TEST(Evaluation, GainAndLiftKeepStableTiesAndTwentyBuckets)
{
    Evaluation evaluation;
    MatrixR targets(5, 1), scores(5, 1);
    targets << 1, 0, 1, 0, 1;
    scores << 0.8f, 0.8f, 0.8f, 0.2f, 0.1f;
    const MatrixR gain = evaluation.calculate_cumulative_gain(targets, scores);
    ASSERT_EQ(gain.rows(), 21);
    ASSERT_EQ(gain.cols(), 2);
    for (Index row = 0; row < gain.rows(); ++row)
        EXPECT_FLOAT_EQ(gain(row, 0), float(row) / 20.0f);
    EXPECT_FLOAT_EQ(gain(3, 1), 0.0f);
    EXPECT_FLOAT_EQ(gain(4, 1), 1.0f / 3);
    EXPECT_FLOAT_EQ(gain(8, 1), 1.0f / 3);
    EXPECT_FLOAT_EQ(gain(12, 1), 2.0f / 3);
    EXPECT_FLOAT_EQ(gain(16, 1), 2.0f / 3);
    EXPECT_FLOAT_EQ(gain(20, 1), 1.0f);
    const MatrixR lift = evaluation.calculate_lift_chart(gain);
    EXPECT_FLOAT_EQ(lift(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(lift(0, 1), 1.0f);
    EXPECT_NEAR(lift(4, 1), 5.0f / 3, 1.0e-6f);
    EXPECT_NEAR(lift(8, 1), 5.0f / 6, 1.0e-6f);
    EXPECT_NEAR(lift(12, 1), 10.0f / 9, 1.0e-6f);
    EXPECT_FLOAT_EQ(lift(20, 1), 1.0f);

    swap(targets(0, 0), targets(1, 0));
    const MatrixR reordered_gain = evaluation.calculate_cumulative_gain(targets, scores);
    EXPECT_FLOAT_EQ(reordered_gain(4, 1), 0.0f);
    EXPECT_FLOAT_EQ(reordered_gain(8, 1), 1.0f / 3);
    EXPECT_THROW(evaluation.calculate_cumulative_gain(MatrixR::Zero(5, 1), scores), runtime_error);
    EXPECT_FLOAT_EQ(evaluation.calculate_cumulative_gain(MatrixR::Ones(5, 1), scores)(20, 1), 1.0f);
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

TEST(Evaluation, BinarySampleGroupsPreserveThresholdAndSourceIndices)
{
    Evaluation evaluation;
    MatrixR targets(6, 1), outputs(6, 1);
    targets << 0.5f, 0.49f, 0.5f, 0.49f, QUIET_NAN, 0.5f;
    outputs << 0.5f, 0.5f, 0.49f, 0.49f, QUIET_NAN, QUIET_NAN;
    vector<Index> sample_indices{42, 7, 101, 13, 88, 6};
    const auto groups = [&]
    {
        return std::array<vector<Index>, 4>{
            evaluation.calculate_true_positive_samples(targets, outputs, sample_indices, 0.5f),
            evaluation.calculate_false_positive_samples(targets, outputs, sample_indices, 0.5f),
            evaluation.calculate_false_negative_samples(targets, outputs, sample_indices, 0.5f),
            evaluation.calculate_true_negative_samples(targets, outputs, sample_indices, 0.5f)};
    };
    const auto actual = groups();
    EXPECT_EQ(actual[0], (vector<Index>{42}));
    EXPECT_EQ(actual[1], (vector<Index>{7}));
    EXPECT_EQ(actual[2], (vector<Index>{101, 6}));
    EXPECT_EQ(actual[3], (vector<Index>{13, 88}));
    const MatrixI confusion = evaluation.calculate_confusion(targets, outputs);
    MatrixI expected(3, 3);
    expected << 1, 2, 3, 1, 2, 3, 2, 4, 6;
    EXPECT_EQ(confusion, expected);

    const std::array<pair<float, float>, 4> constant_cells{{{1, 1}, {0, 1}, {1, 0}, {0, 0}}};
    for (size_t cell = 0; cell < constant_cells.size(); ++cell)
    {
        targets.setConstant(constant_cells[cell].first);
        outputs.setConstant(constant_cells[cell].second);
        const auto constant_groups = groups();
        for (size_t group = 0; group < constant_groups.size(); ++group)
            EXPECT_EQ(constant_groups[group], group == cell ? sample_indices : vector<Index>{});
    }
    targets.resize(0, 1);
    outputs.resize(0, 1);
    sample_indices.clear();
    for (const auto& group : groups()) EXPECT_TRUE(group.empty());
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

    targets.resize(4, 3);
    outputs.resize(4, 3);
    targets << 0.5f, 0.5f, 0, 1, 0, 0, 0.2f, 0.2f, 0.1f, 0, 0, 1;
    outputs << 0, 0.7f, 0.7f, 0, 1, 0, 0, 0.5f, 0.5f, 0.9f, 0.9f, 0;
    const vector<Index> original_indices{42, 7, 101, 13};
    const auto tied_rates = evaluation.calculate_multiple_classification_rates(targets, outputs, original_indices);
    ASSERT_EQ(tied_rates(0, 1).size(), 3);
    for (Index i = 0; i < 3; ++i) EXPECT_EQ(tied_rates(0, 1)(i), original_indices[size_t(i)]);
    ASSERT_EQ(tied_rates(2, 0).size(), 1);
    EXPECT_EQ(tied_rates(2, 0)(0), 13);
    EXPECT_EQ(tied_rates(1, 1).size(), 0);
}
