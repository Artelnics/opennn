#include "tests/pch.h"
#include "tests/numerical_derivatives.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/batch.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/quasi_newton_method.h"
#include "opennn/core/device_backend.h"
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
        set_threads_number(0);
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

    const VectorR gradient = calculate_gradient(loss);

    EXPECT_EQ(gradient.size(), neural_network.get_parameters_buffer_size());

    for (Index i = 0; i < gradient.size(); ++i)
        EXPECT_FALSE(isnan(gradient(i)));
}

TEST_F(QuasiNewtonMethodTest, TrainApproximationCPU)
{
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
    set_threads_number(1);

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


// Loss::back_propagate deliberately leaves metrics.regularization at zero: on
// GPU the penalty is a cuBLAS reduction with a host result pointer, so paying
// it per batch costs a full sync, and the mini-batch optimizers overwrite it
// once per epoch anyway. Full-batch QuasiNewtonMethod reads loss_value on every
// line-search step, so it adds the term itself. Both halves are pinned here
// because every other test in the suite runs with regularization switched off.
TEST_F(QuasiNewtonMethodTest, BackPropagateLeavesRegularizationToTheOptimizer)
{
    set_seed(1);

    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({2}, {6}, {1});

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization(Loss::Regularization::L2);
    loss.set_regularization_weight(0.5f);

    const TensorView parameters(network.get_parameters_data(),
                                {network.get_parameters_buffer_size()},
                                Type::FP32,
                                network.get_device());

    // The configuration is not degenerate: there is a penalty to be had.
    EXPECT_GT(loss.calculate_regularization(parameters), 0.0f);

    // Through the base class: TabularDataset hides the role-taking overloads.
    Dataset& base_dataset = dataset;
    const Index samples_number = base_dataset.get_samples_number("Training");

    Batch batch(samples_number, &dataset, network.get_config());
    batch.fill(base_dataset.get_sample_indices("Training"),
               base_dataset.get_feature_selection());

    ForwardPropagation forward_propagation(samples_number, &network);
    BackPropagation back_propagation(samples_number, loss);

    network.forward_propagate(batch.get_inputs(), forward_propagation, ForwardPropagationMode::Training);
    loss.back_propagate(batch, forward_propagation, back_propagation);

    EXPECT_FLOAT_EQ(back_propagation.metrics.regularization, 0.0f);
    EXPECT_FLOAT_EQ(back_propagation.metrics.loss_value, back_propagation.metrics.error);
}
