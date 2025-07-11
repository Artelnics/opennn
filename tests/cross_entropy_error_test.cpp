#include "pch.h"

#include "../opennn/cross_entropy_error.h"
#include "../opennn/tensors.h"
#include "../opennn/mean_squared_error.h"

using namespace opennn;

TEST(CrossEntropyErrorTest, DefaultConstructor)
{

    CrossEntropyError2d cross_entropy_error;

    EXPECT_TRUE(!cross_entropy_error.has_data_set());
    EXPECT_TRUE(!cross_entropy_error.has_neural_network());
}


TEST(CrossEntropyErrorTest, BackPropagate)
{

    /*
    const Index samples_number = get_random_index(1, 10);

    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = 1;
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });

    dataset.set_data_classification();

    dataset.set("Training");

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
        { inputs_number }, { neurons_number }, { targets_number });

    neural_network.set_parameters_random();

    CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);

    const Tensor<type, 1> gradient = cross_entropy_error.calculate_gradient();

    const Tensor<type, 1> numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
*/
}
