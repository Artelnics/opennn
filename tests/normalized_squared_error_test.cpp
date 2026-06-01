#include "pch.h"
#include "numerical_derivatives.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/dataset.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/loss.h"

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

    EXPECT_NEAR(calculate_numerical_error(loss), 0.6f, 1.0e-6f);
}
