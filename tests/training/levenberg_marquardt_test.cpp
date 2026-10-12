// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "tests/common/pch.h"
#include "tests/common/numerical_derivatives.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/training/loss.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training/levenberg_marquardt.h"
#include "opennn/training/levenberg_marquardt_step.h"
#include "opennn/core/device_backend.h"
#include "gtest/gtest.h"

using namespace opennn;

TEST(LevenbergMarquardtStep, SolvesCoupledSystemWithoutChangingInputs)
{
    MatrixR curvature(2, 2);
    curvature << 3, 1, 1, 2;
    const MatrixR original = curvature;
    VectorR gradient(2), step(2);
    gradient << -9, -8;
    float damping = 1.0f;
    ASSERT_TRUE(detail::find_damped_step(curvature, gradient, damping, 10.0f, 0.001f, 100.0f,
                                         step, [](const auto&) { return true; }));
    EXPECT_NEAR(step(0), 19.0f / 11, 1.0e-6f);
    EXPECT_NEAR(step(1), 23.0f / 11, 1.0e-6f);
    EXPECT_EQ(curvature, original);
    EXPECT_FLOAT_EQ(damping, 0.1f);
}

TEST(LevenbergMarquardtStep, RetriesFromOriginalSystemUntilAcceptance)
{
    MatrixR curvature = MatrixR::Zero(2, 2);
    curvature.diagonal() << 1, 2;
    const MatrixR original = curvature;
    VectorR gradient(2), step(2);
    gradient << -2, -3;
    float damping = 0.1f;
    vector<float> attempts;
    const auto accept = [&](const auto& candidate)
    {
        attempts.push_back(damping);
        return candidate.norm() < 1.0f;
    };
    ASSERT_TRUE(detail::find_damped_step(curvature, gradient, damping, 10.0f, 0.001f, 100.0f,
                                         step, accept));
    EXPECT_EQ(attempts, (vector<float>{0.1f, 1.0f, 10.0f}));
    EXPECT_NEAR(step(0), 2.0f / 11, 1.0e-6f);
    EXPECT_NEAR(step(1), 0.25f, 1.0e-6f);
    EXPECT_EQ(curvature, original);
    EXPECT_FLOAT_EQ(damping, 1.0f);
}

TEST(LevenbergMarquardtStep, ExhaustionPreservesCurvatureAndStopsAtBound)
{
    MatrixR curvature = MatrixR::Identity(1, 1);
    const VectorR gradient = VectorR::Ones(1);
    VectorR step(1);
    float damping = 1.0f;
    int attempts = 0;
    const auto reject = [&](const auto&) { ++attempts; return false; };
    EXPECT_FALSE(detail::find_damped_step(curvature, gradient, damping, 10.0f, 0.001f, 100.0f,
                                          step, reject));
    EXPECT_EQ(attempts, 2);
    EXPECT_FLOAT_EQ(damping, 100.0f);
    EXPECT_FLOAT_EQ(curvature(0, 0), 1.0f);
    attempts = 0;
    EXPECT_FALSE(detail::find_damped_step(curvature, gradient, damping, 10.0f, 0.001f, 100.0f,
                                          step, reject));
    EXPECT_EQ(attempts, 1);
}

TEST(LevenbergMarquardtStep, RejectsInvalidFactorsAndNonfiniteSystem)
{
    LevenbergMarquardt optimizer;
    for (float factor : {0.0f, 1.0f, -2.0f, numeric_limits<float>::infinity(),
                         numeric_limits<float>::quiet_NaN()})
        EXPECT_THROW(optimizer.set_damping_parameter_factor(factor), runtime_error);

    MatrixR curvature = MatrixR::Identity(1, 1);
    VectorR gradient = VectorR::Ones(1), step(1);
    float damping = 1.0f;
    int attempts = 0;
    const auto accept = [&](const auto&) { ++attempts; return true; };
    EXPECT_THROW(detail::find_damped_step(curvature, gradient, damping, 1.0f, 0.001f, 100.0f,
                                          step, accept), invalid_argument);
    gradient(0) = numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(detail::find_damped_step(curvature, gradient, damping, 10.0f, 0.001f, 100.0f,
                                          step, accept));
    EXPECT_EQ(attempts, 0);
}

class LevenbergMarquardtTest : public ::testing::Test
{
protected:
    int previous_threads = 0;

    void SetUp() override
    {
        previous_threads = get_device().numThreads();
    }

    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        set_threads_number(previous_threads);
    }
};

TEST_F(LevenbergMarquardtTest, DefaultConstructor)
{
    LevenbergMarquardt levenberg_marquardt;

    EXPECT_EQ(levenberg_marquardt.get_loss() == nullptr, true);
}

TEST_F(LevenbergMarquardtTest, GeneralConstructor)
{
    Loss loss;
    LevenbergMarquardt levenberg_marquardt(&loss);

    EXPECT_TRUE(levenberg_marquardt.get_loss() != nullptr);
}

TEST_F(LevenbergMarquardtTest, RecordsValidationErrorWithoutJacobianState)
{
    set_seed(32);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.split_samples_sequential(0.75f, 0.25f, 0.0f);

    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");

    LevenbergMarquardt optimizer(&loss);
    optimizer.set_maximum_epochs(1);
    optimizer.set_display(false);

    const TrainingResult result = optimizer.train();

    ASSERT_EQ(result.validation_error_history.size(), 1);
    EXPECT_TRUE(isfinite(result.validation_error_history(0)));
}

TEST_F(LevenbergMarquardtTest, TrainApproximationCPU)
{
    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    loss_short.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt_short(&loss_short);
    levenberg_marquardt_short.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_short.set_maximum_epochs(1);
    levenberg_marquardt_short.set_display(false);
    const type error_short = levenberg_marquardt_short.train().get_training_error();

    set_seed(1);
    TabularDataset dataset_long(16, {2}, {1});
    dataset_long.set_data_random();
    dataset_long.set_sample_roles("Training");
    ApproximationNetwork network_long({2}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    loss_long.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt_long(&loss_long);
    levenberg_marquardt_long.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_long.set_maximum_epochs(100);
    levenberg_marquardt_long.set_display(false);
    const type error_long = levenberg_marquardt_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(LevenbergMarquardtTest, DampingParametersConverge)
{
    set_seed(2);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt(&loss);
    levenberg_marquardt.set_damping_parameter_factor(10.0f);
    levenberg_marquardt.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt.set_display(false);

    levenberg_marquardt.set_maximum_epochs(1);
    const type error_short = levenberg_marquardt.train().get_training_error();
    levenberg_marquardt.set_maximum_epochs(100);
    const type error_long = levenberg_marquardt.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(LevenbergMarquardtTest, StoppingMaximumEpochs)
{
    set_seed(3);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt(&loss);
    levenberg_marquardt.set_maximum_epochs(5);
    levenberg_marquardt.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt.set_display(false);

    const TrainingResult training_results = levenberg_marquardt.train();

    EXPECT_EQ(training_results.get_epochs_number(), 5);
    EXPECT_EQ(training_results.get_epochs_number(), training_results.training_error_history.size());
}

TEST_F(LevenbergMarquardtTest, StoppingLossGoal)
{
    set_seed(4);
    TabularDataset dataset(4, {1}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({1}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt(&loss);

    const type training_loss_goal = type(0.05);
    levenberg_marquardt.set_loss_goal(training_loss_goal);
    levenberg_marquardt.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt.set_maximum_epochs(10000);
    levenberg_marquardt.set_maximum_time(600.0f);
    levenberg_marquardt.set_display(false);

    const TrainingResult training_results = levenberg_marquardt.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}

TEST_F(LevenbergMarquardtTest, StoppingMaximumTime)
{
    set_seed(5);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt(&loss);
    levenberg_marquardt.set_maximum_epochs(1000000);
    levenberg_marquardt.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt.set_maximum_time(0.5);
    levenberg_marquardt.set_display(false);

    const time_t start = time(nullptr);
    const TrainingResult training_results = levenberg_marquardt.train();
    const double elapsed = difftime(time(nullptr), start);

    EXPECT_LT(training_results.get_epochs_number(), 1000000);
    EXPECT_LT(elapsed, 30.0);
}

TEST_F(LevenbergMarquardtTest, Determinism)
{
    set_threads_number(1);

    set_seed(6);
    TabularDataset dataset_first(16, {2}, {1});
    dataset_first.set_data_random();
    dataset_first.set_sample_roles("Training");
    ApproximationNetwork network_first({2}, {6}, {1});
    Loss loss_first(&network_first, &dataset_first);
    loss_first.set_error(Loss::Error::MeanSquaredError);
    loss_first.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt_first(&loss_first);
    levenberg_marquardt_first.set_workers_number(1);
    levenberg_marquardt_first.set_maximum_epochs(20);
    levenberg_marquardt_first.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_first.set_display(false);
    const type error_first = levenberg_marquardt_first.train().get_training_error();

    set_seed(6);
    TabularDataset dataset_second(16, {2}, {1});
    dataset_second.set_data_random();
    dataset_second.set_sample_roles("Training");
    ApproximationNetwork network_second({2}, {6}, {1});
    Loss loss_second(&network_second, &dataset_second);
    loss_second.set_error(Loss::Error::MeanSquaredError);
    loss_second.set_regularization("None");
    LevenbergMarquardt levenberg_marquardt_second(&loss_second);
    levenberg_marquardt_second.set_workers_number(1);
    levenberg_marquardt_second.set_maximum_epochs(20);
    levenberg_marquardt_second.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_second.set_display(false);
    const type error_second = levenberg_marquardt_second.train().get_training_error();

    EXPECT_FLOAT_EQ(error_first, error_second);
}

TEST_F(LevenbergMarquardtTest, RepeatedTrainingResetsState)
{
    set_threads_number(1);
    set_seed(37);

    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({2}, {6}, {1});
    const VectorR initial_parameters = network.get_parameters_map();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");

    LevenbergMarquardt levenberg_marquardt(&loss);
    levenberg_marquardt.set_workers_number(1);
    levenberg_marquardt.set_maximum_epochs(20);
    levenberg_marquardt.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt.set_display(false);

    const float first_error = levenberg_marquardt.train().get_training_error();

    network.set_parameters(initial_parameters);
    const float second_error = levenberg_marquardt.train().get_training_error();

    EXPECT_FLOAT_EQ(first_error, second_error);
}
