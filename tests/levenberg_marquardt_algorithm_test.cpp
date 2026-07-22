#include "pch.h"
#include "numerical_derivatives.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"
#include "opennn/dense_layer.h"
#include "opennn/loss.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/levenberg_marquardt_algorithm.h"
#include "opennn/device_backend.h"
#include "gtest/gtest.h"

using namespace opennn;

class LevenbergMarquardtAlgorithmTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        Backend::instance().set_threads_number(0);
    }
};

TEST_F(LevenbergMarquardtAlgorithmTest, DefaultConstructor)
{
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;

    EXPECT_EQ(levenberg_marquardt_algorithm.get_loss() == nullptr, true);
}


TEST_F(LevenbergMarquardtAlgorithmTest, GeneralConstructor)
{
    Loss loss;
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&loss);

    EXPECT_TRUE(levenberg_marquardt_algorithm.get_loss() != nullptr);
}


TEST_F(LevenbergMarquardtAlgorithmTest, TrainApproximationCPU)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    loss_short.set_regularization("None");
    LevenbergMarquardtAlgorithm levenberg_marquardt_short(&loss_short);
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
    LevenbergMarquardtAlgorithm levenberg_marquardt_long(&loss_long);
    levenberg_marquardt_long.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_long.set_maximum_epochs(100);
    levenberg_marquardt_long.set_display(false);
    const type error_long = levenberg_marquardt_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}


TEST_F(LevenbergMarquardtAlgorithmTest, DampingParametersConverge)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(2);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&loss);
    levenberg_marquardt_algorithm.set_damping_parameter(0.01f);
    levenberg_marquardt_algorithm.set_damping_parameter_factor(10.0f);
    levenberg_marquardt_algorithm.set_minimum_damping_parameter(1.0e-6f);
    levenberg_marquardt_algorithm.set_maximum_damping_parameter(1.0e6f);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_algorithm.set_display(false);

    levenberg_marquardt_algorithm.set_maximum_epochs(1);
    const type error_short = levenberg_marquardt_algorithm.train().get_training_error();
    levenberg_marquardt_algorithm.set_maximum_epochs(100);
    const type error_long = levenberg_marquardt_algorithm.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}


TEST_F(LevenbergMarquardtAlgorithmTest, StoppingMaximumEpochs)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(3);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&loss);
    levenberg_marquardt_algorithm.set_maximum_epochs(5);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_algorithm.set_display(false);

    const TrainingResult training_results = levenberg_marquardt_algorithm.train();

    EXPECT_EQ(training_results.get_epochs_number(), 5);
    EXPECT_EQ(training_results.get_epochs_number(), training_results.training_error_history.size());
}


TEST_F(LevenbergMarquardtAlgorithmTest, StoppingLossGoal)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(4);
    TabularDataset dataset(4, {1}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({1}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&loss);

    const type training_loss_goal = type(0.1);
    levenberg_marquardt_algorithm.set_loss_goal(training_loss_goal);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_algorithm.set_maximum_epochs(10000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);
    levenberg_marquardt_algorithm.set_display(false);

    const TrainingResult training_results = levenberg_marquardt_algorithm.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}


TEST_F(LevenbergMarquardtAlgorithmTest, StoppingMaximumTime)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(5);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&loss);
    levenberg_marquardt_algorithm.set_maximum_epochs(1000000);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_algorithm.set_maximum_time(0.5);
    levenberg_marquardt_algorithm.set_display(false);

    const time_t start = time(nullptr);
    const TrainingResult training_results = levenberg_marquardt_algorithm.train();
    const double elapsed = difftime(time(nullptr), start);

    EXPECT_LT(training_results.get_epochs_number(), 1000000);
    EXPECT_LT(elapsed, 30.0);
}


TEST_F(LevenbergMarquardtAlgorithmTest, Determinism)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    Backend::instance().set_threads_number(1);

    set_seed(6);
    TabularDataset dataset_first(16, {2}, {1});
    dataset_first.set_data_random();
    dataset_first.set_sample_roles("Training");
    ApproximationNetwork network_first({2}, {6}, {1});
    Loss loss_first(&network_first, &dataset_first);
    loss_first.set_error(Loss::Error::MeanSquaredError);
    loss_first.set_regularization("None");
    LevenbergMarquardtAlgorithm levenberg_marquardt_first(&loss_first);
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
    LevenbergMarquardtAlgorithm levenberg_marquardt_second(&loss_second);
    levenberg_marquardt_second.set_workers_number(1);
    levenberg_marquardt_second.set_maximum_epochs(20);
    levenberg_marquardt_second.set_minimum_loss_decrease(0.0f);
    levenberg_marquardt_second.set_display(false);
    const type error_second = levenberg_marquardt_second.train().get_training_error();

    EXPECT_FLOAT_EQ(error_first, error_second);
}
