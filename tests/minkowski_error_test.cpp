#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/neural_network.h"
#include "../opennn/dataset.h"
#include "../opennn/minkowski_error.h"

using namespace opennn;

TEST(MinkowskiErrorTest, DefaultConstructor)
{
    MinkowskiError minkowski_error;

    EXPECT_EQ(minkowski_error.has_neural_network(), false);
    EXPECT_EQ(minkowski_error.has_data_set(), false);
}


TEST(MinkowskiErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    MinkowskiError minkowski_error(&neural_network, &dataset);

    EXPECT_EQ(minkowski_error.has_neural_network(), true);
    EXPECT_EQ(minkowski_error.has_data_set(), true);
}


TEST(MinkowskiErrorTest, BackPropagate)
{
/*
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set(Dataset::SampleUse::Training);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
        { inputs_number }, { neurons_number }, { targets_number });

    neural_network.set_parameters_random();

    MinkowskiError minkowski_error(&neural_network, &dataset);

    const Tensor<type, 1> gradient = minkowski_error.calculate_gradient();
    const Tensor<type, 1> numerical_gradient = minkowski_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
*/
}
