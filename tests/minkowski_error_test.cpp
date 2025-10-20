#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/neural_network.h"
#include "../opennn/dataset.h"
#include "../opennn/minkowski_error.h"
#include "../opennn/standard_networks.h"

using namespace opennn;

TEST(MinkowskiErrorTest, DefaultConstructor)
{
    MinkowskiError minkowski_error;

    EXPECT_EQ(minkowski_error.has_neural_network(), false);
    EXPECT_EQ(minkowski_error.has_dataset(), false);
}


TEST(MinkowskiErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    MinkowskiError minkowski_error(&neural_network, &dataset);

    EXPECT_EQ(minkowski_error.has_neural_network(), true);
    EXPECT_EQ(minkowski_error.has_dataset(), true);
}


TEST(MinkowskiErrorTest, BackPropagate)
{

    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_uses("Training");

    ApproximationNetwork neural_network({ inputs_number }, { neurons_number }, { outputs_number });

    MinkowskiError minkowski_error(&neural_network, &dataset);

    const Tensor<type, 1> gradient = minkowski_error.calculate_gradient();
    const Tensor<type, 1> numerical_gradient = minkowski_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}
