#include "tests/pch.h"
#include "opennn/core/random_utilities.h"
#include "tests/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/loss.h"

using namespace opennn;

TEST(NormalizedSquaredErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_neural_network() == nullptr, true);
    EXPECT_EQ(loss.get_dataset() == nullptr, true);
}

TEST(NormalizedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    TabularDataset dataset;

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::NormalizedSquaredError);

    EXPECT_EQ(loss.get_neural_network() != nullptr, true);
    EXPECT_EQ(loss.get_dataset() != nullptr, true);
}

TEST(NormalizedSquaredErrorTest, BackPropagate)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {targets_number});

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::NormalizedSquaredError);
    loss.set_normalization_coefficient();
    loss.set_regularization_weight(0.0);

    const type error = calculate_numerical_error(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);
    const VectorR gradient = calculate_gradient(loss);

    EXPECT_GE(error, 0);
    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(NormalizedSquaredErrorTest, SetNormalizationCoefficientFromTrainingTargets)
{
    TabularDataset dataset(4, {1}, {1});
    MatrixR data(4, 2);
    data << 0.0f, -2.0f,
            0.0f,  0.0f,
            0.0f,  2.0f,
            0.0f,  4.0f;
    dataset.set_data(data);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::NormalizedSquaredError);
    loss.set_normalization_coefficient();

    EXPECT_NEAR(calculate_numerical_error(loss), 1.2f, 1.0e-6f);
}

// The normalization coefficient is a constant over the whole training set, and
// the optimizer averages the per-batch errors, so a mini-batch error has to
// carry the training/batch sample ratio or the epoch value comes out as the
// true one divided by the batch count. WeightedSquaredError has always applied
// that factor; NormalizedSquaredError had lost it.

TEST(NormalizedSquaredErrorTest, MiniBatchErrorMeanMatchesFullBatch)
{
    constexpr Index samples_number = 8;
    constexpr Index inputs_number = 3;
    constexpr Index targets_number = 1;

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({inputs_number}, {4}, {targets_number});
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::NormalizedSquaredError);
    loss.set_normalization_coefficient();
    loss.set_regularization_weight(0.0f);

    const vector<Index> training_indices        = dataset.get_sample_indices("Training");
    const FeatureSelection features = dataset.get_feature_selection();

    const auto error_over = [&](const vector<Index>& indices)
    {
        const Index count = ssize(indices);

        Batch batch(count, &dataset, neural_network.get_config());
        batch.fill(indices, features);

        ForwardPropagation forward_propagation(count, &neural_network);
        neural_network.forward_propagate(batch.get_inputs(), forward_propagation);

        return loss.calculate_error(batch, forward_propagation).error;
    };

    const float full_batch_error = error_over(training_indices);

    const vector<Index> first_half(training_indices.begin(),
                                   training_indices.begin() + samples_number / 2);
    const vector<Index> second_half(training_indices.begin() + samples_number / 2,
                                    training_indices.end());

    const float mean_of_halves = 0.5f * (error_over(first_half) + error_over(second_half));

    EXPECT_GT(full_batch_error, 0.0f);
    EXPECT_NEAR(mean_of_halves, full_batch_error, 1.0e-4f * max(1.0f, full_batch_error));
}
