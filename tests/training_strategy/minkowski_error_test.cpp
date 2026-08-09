#include "tests/pch.h"
#include "opennn/core/random_utilities.h"
#include "tests/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/standard_networks.h"

using namespace opennn;

TEST(MinkowskiErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_neural_network() == nullptr, true);
    EXPECT_EQ(loss.get_dataset() == nullptr, true);
}

TEST(MinkowskiErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    TabularDataset dataset;

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MinkowskiError);

    EXPECT_EQ(loss.get_neural_network() != nullptr, true);
    EXPECT_EQ(loss.get_dataset() != nullptr, true);
}

TEST(MinkowskiErrorTest, BackPropagate)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index outputs_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({ inputs_number }, { neurons_number }, { outputs_number });

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MinkowskiError);

    const VectorR gradient = calculate_gradient(loss);

    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-2));
}
