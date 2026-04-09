#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/dataset.h"
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
    Dataset dataset;

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

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {targets_number});

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::NormalizedSquaredError);
    loss.set_normalization_coefficient();
    loss.set_regularization_weight(0.0);

    const type error = loss.calculate_numerical_error();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();
    const VectorR gradient = loss.calculate_gradient();

    EXPECT_GE(error, 0);
    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}
