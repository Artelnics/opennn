#include "pch.h"
#include "numerical_derivatives.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/loss.h"
#include "opennn/quasi_newton_method.h"
#include "opennn/device_backend.h"
#include "gtest/gtest.h"

using namespace opennn;

namespace
{
    MatrixR separable_classification_data(Index samples_number, Index inputs_number)
    {
        MatrixR data(samples_number, inputs_number + 1);
        for (Index i = 0; i < samples_number; ++i)
        {
            float sum = 0.0f;
            for (Index j = 0; j < inputs_number; ++j)
            {
                const float value = ((i * 7 + j * 13) % 100) / 50.0f - 1.0f;
                data(i, j) = value;
                sum += value;
            }
            data(i, inputs_number) = sum > 0.0f ? 1.0f : 0.0f;
        }
        return data;
    }
}

class QuasiNewtonMethodTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        Backend::instance().set_threads_number(0);
    }
};

TEST_F(QuasiNewtonMethodTest, DefaultConstructor)
{
    QuasiNewtonMethod quasi_newton_method;

    EXPECT_EQ(quasi_newton_method.get_loss() == nullptr, true);
}


TEST_F(QuasiNewtonMethodTest, GeneralConstructor)
{
    Loss loss;
    QuasiNewtonMethod quasi_newton_method(&loss);

    EXPECT_TRUE(quasi_newton_method.get_loss() != nullptr);
}


TEST_F(QuasiNewtonMethodTest, BFGS_Update)
{
    const Index inputs_number = 1;
    const Index outputs_number = 1;
    const Index samples_number = 10;

    TabularDataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({ inputs_number }, {}, { outputs_number });

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_method(&loss);

    neural_network.set_parameters_random();

    VectorR gradient_k = calculate_gradient(loss);

    EXPECT_EQ(gradient_k.size(), neural_network.get_parameters_size());

    MatrixR numerical_inverse_hessian = calculate_inverse_hessian(loss);

    EXPECT_EQ(numerical_inverse_hessian.rows(), numerical_inverse_hessian.cols());

    for (Index i = 0; i < numerical_inverse_hessian.size(); ++i)
        EXPECT_FALSE(isnan(numerical_inverse_hessian(i)));
}


TEST_F(QuasiNewtonMethodTest, TrainApproximationCPU)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_short(&loss_short);
    quasi_newton_short.set_maximum_epochs(1);
    quasi_newton_short.set_display(false);
    const type error_short = quasi_newton_short.train().get_training_error();

    set_seed(1);
    TabularDataset dataset_long(16, {2}, {1});
    dataset_long.set_data_random();
    dataset_long.set_sample_roles("Training");
    ApproximationNetwork network_long({2}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_long(&loss_long);
    quasi_newton_long.set_maximum_epochs(100);
    quasi_newton_long.set_display(false);
    const type error_long = quasi_newton_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}


TEST_F(QuasiNewtonMethodTest, TrainClassificationCPU)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const MatrixR classification_data = separable_classification_data(16, 3);

    set_seed(2);
    TabularDataset dataset_short(16, {3}, {1});
    dataset_short.set_data(classification_data);
    dataset_short.set_sample_roles("Training");
    ClassificationNetwork network_short({3}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    QuasiNewtonMethod quasi_newton_short(&loss_short);
    quasi_newton_short.set_maximum_epochs(1);
    quasi_newton_short.set_display(false);
    const type error_short = quasi_newton_short.train().get_training_error();

    set_seed(2);
    TabularDataset dataset_long(16, {3}, {1});
    dataset_long.set_data(classification_data);
    dataset_long.set_sample_roles("Training");
    ClassificationNetwork network_long({3}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    QuasiNewtonMethod quasi_newton_long(&loss_long);
    quasi_newton_long.set_maximum_epochs(100);
    quasi_newton_long.set_display(false);
    const type error_long = quasi_newton_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}


TEST_F(QuasiNewtonMethodTest, MinimumLossDecreaseConverges)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(3);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_method(&loss);
    quasi_newton_method.set_minimum_loss_decrease(0.0f);
    quasi_newton_method.set_display(false);

    quasi_newton_method.set_maximum_epochs(1);
    const type error_short = quasi_newton_method.train().get_training_error();
    quasi_newton_method.set_maximum_epochs(100);
    const type error_long = quasi_newton_method.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}


TEST_F(QuasiNewtonMethodTest, StoppingMaximumEpochs)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(4);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_method(&loss);
    quasi_newton_method.set_maximum_epochs(5);
    quasi_newton_method.set_minimum_loss_decrease(0.0f);
    quasi_newton_method.set_display(false);

    const TrainingResult training_results = quasi_newton_method.train();

    EXPECT_EQ(training_results.get_epochs_number(), 5);
    EXPECT_EQ(training_results.get_epochs_number(), training_results.training_error_history.size());
}


TEST_F(QuasiNewtonMethodTest, StoppingLossGoal)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(5);
    TabularDataset dataset(4, {1}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({1}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_method(&loss);

    const type training_loss_goal = type(0.1);
    quasi_newton_method.set_loss_goal(training_loss_goal);
    quasi_newton_method.set_minimum_loss_decrease(0.0f);
    quasi_newton_method.set_maximum_epochs(10000);
    quasi_newton_method.set_maximum_time(1000.0);
    quasi_newton_method.set_display(false);

    const TrainingResult training_results = quasi_newton_method.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}


TEST_F(QuasiNewtonMethodTest, StoppingMaximumTime)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(6);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_method(&loss);
    quasi_newton_method.set_maximum_epochs(1000000);
    quasi_newton_method.set_minimum_loss_decrease(0.0f);
    quasi_newton_method.set_maximum_time(0.5);
    quasi_newton_method.set_display(false);

    const time_t start = time(nullptr);
    const TrainingResult training_results = quasi_newton_method.train();
    const double elapsed = difftime(time(nullptr), start);

    EXPECT_LT(training_results.get_epochs_number(), 1000000);
    EXPECT_LT(elapsed, 30.0);
}


TEST_F(QuasiNewtonMethodTest, Determinism)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    Backend::instance().set_threads_number(1);

    set_seed(7);
    TabularDataset dataset_first(16, {2}, {1});
    dataset_first.set_data_random();
    dataset_first.set_sample_roles("Training");
    ApproximationNetwork network_first({2}, {6}, {1});
    Loss loss_first(&network_first, &dataset_first);
    loss_first.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_first(&loss_first);
    quasi_newton_first.set_workers_number(1);
    quasi_newton_first.set_maximum_epochs(30);
    quasi_newton_first.set_minimum_loss_decrease(0.0f);
    quasi_newton_first.set_display(false);
    const type error_first = quasi_newton_first.train().get_training_error();

    set_seed(7);
    TabularDataset dataset_second(16, {2}, {1});
    dataset_second.set_data_random();
    dataset_second.set_sample_roles("Training");
    ApproximationNetwork network_second({2}, {6}, {1});
    Loss loss_second(&network_second, &dataset_second);
    loss_second.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_second(&loss_second);
    quasi_newton_second.set_workers_number(1);
    quasi_newton_second.set_maximum_epochs(30);
    quasi_newton_second.set_minimum_loss_decrease(0.0f);
    quasi_newton_second.set_display(false);
    const type error_second = quasi_newton_second.train().get_training_error();

    EXPECT_FLOAT_EQ(error_first, error_second);
}
