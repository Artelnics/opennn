#include "pch.h"
#include "opennn/tabular_dataset.h"
#include "numerical_derivatives.h"

#include "opennn/loss.h"
#include "opennn/tensor_types.h"
#include "opennn/dense_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/neural_network.h"
#include "opennn/batch.h"
#include "opennn/forward_propagation.h"
#include "opennn/back_propagation.h"


using namespace opennn;

TEST(CrossEntropyError2d, DefaultConstructor)
{
    Loss loss;

    EXPECT_TRUE(loss.get_neural_network() == nullptr);
    EXPECT_TRUE(loss.get_dataset() == nullptr);
}

TEST(CrossEntropyError2d, BackPropagate)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = random_integer(2, 10);

    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = 1;
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });

    dataset.set_data_random();

    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ inputs_number }, Shape{ targets_number }, "Sigmoid"));
    neural_network.compile();

    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    const VectorR gradient = calculate_gradient(loss);

    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));

    type difference = 0;
    for (int i = 0; i < gradient.size(); i++)
    {
        difference += ((gradient[i] - numerical_gradient[i]) * (gradient[i] - numerical_gradient[i]));
    }
    type error = sqrt(difference);

    EXPECT_NEAR(error,0, type(1.0e-1));

    Configuration::instance().set(Device::CPU, Type::FP32);
}


TEST(CrossEntropyError2d, CalculateError)
{
    TabularDataset dataset(5, { 3 }, { 1 });

    MatrixR data(5, 4);
    data << type(0), type(1), type(0), type(1),
            type(1), type(1), type(0), type(0),
            type(0), type(1), type(1), type(1),
            type(0), type(1), type(1), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);

    const vector<Index> input_features_indices = { 0, 1, 2 };
    const vector<Index> target_features_indices = { 3 };

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ 3 }, Shape{ 1 }, "Sigmoid"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    Batch batch(5, &dataset, neural_network.get_config());

    const vector<Index> training_indices = dataset.get_sample_indices("Training");
    batch.fill(training_indices, input_features_indices, {}, target_features_indices);

    ForwardPropagation forward_propagation(5, &neural_network);
    neural_network.forward_propagate(batch.get_inputs(), forward_propagation, false);

    const type error = loss.calculate_error(batch, forward_propagation).error;

    EXPECT_FALSE(std::isnan(error));
    EXPECT_GE(error, type(0));
}


TEST(CrossEntropyError2d, get_name)
{
    MatrixR data;
    TabularDataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data << type(2), type(5), type(0), type(0),
            type(2), type(9), type(1), type(0),
            type(2), type(9), type(1), type(1),
            type(6), type(5), type(0), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);

    vector<Index> input_features_indices(2);
    input_features_indices = { 0,1 };

    vector<Index> target_features_indices(2);
    target_features_indices = { 2,3 };

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ 2 }, Shape{ 2 }, "Sigmoid"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    string name = loss.get_name();

    EXPECT_EQ(name, "CrossEntropy");
}
