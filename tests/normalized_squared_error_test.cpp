#include "pch.h"

#include "../opennn/normalized_squared_error.h"
#include "../opennn/tensors.h"

using namespace opennn;

TEST(NormalizedSquaredErrorTest, DefaultConstructor)
{
    NormalizedSquaredError normalized_squared_error;

    EXPECT_EQ(normalized_squared_error.has_neural_network(), false);
    EXPECT_EQ(normalized_squared_error.has_data_set(), false);
}


TEST(NormalizedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);

    EXPECT_EQ(normalized_squared_error.has_neural_network(), true);
    EXPECT_EQ(normalized_squared_error.has_data_set(), true);
}


TEST(NormalizedSquaredErrorTest, BackPropagate)
{
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set(Dataset::SampleUse::Training);

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices(Dataset::SampleUse::Training),
               dataset.get_variable_indices(Dataset::VariableUse::Input),
               dataset.get_variable_indices(Dataset::VariableUse::Decoder),
               dataset.get_variable_indices(Dataset::VariableUse::Target));

    NeuralNetwork neural_network(
        NeuralNetwork::ModelType::Approximation,
        {inputs_number},
        {neurons_number},
        {targets_number});

    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);

    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);
    normalized_squared_error.set_normalization_coefficient();

    BackPropagation back_propagation(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    const Tensor<type, 1> numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(back_propagation.errors.dimension(0), samples_number);
    EXPECT_EQ(back_propagation.errors.dimension(1), targets_number);
    EXPECT_GE(back_propagation.error(), 0);
    EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), true);
}


TEST(NormalizedSquaredErrorTest, BackPropagateLM)
{

    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);
    bool is_training = true;

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();
    dataset.set(Dataset::SampleUse::Training);

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices(Dataset::SampleUse::Training),
               dataset.get_variable_indices(Dataset::VariableUse::Input),
               dataset.get_variable_indices(Dataset::VariableUse::Decoder),
               dataset.get_variable_indices(Dataset::VariableUse::Target));

    NeuralNetwork neural_network(
        NeuralNetwork::ModelType::Approximation,
        {inputs_number},
        {neurons_number},
        {outputs_number});
    neural_network.set_parameters_random();

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);
    normalized_squared_error.set_normalization_coefficient();

    const Tensor<type, 1> gradient_lm = normalized_squared_error.calculate_numerical_gradient();
    const Tensor<type, 1> numerical_gradient_lm = normalized_squared_error.calculate_numerical_gradient_lm();

    EXPECT_EQ(are_equal(gradient_lm, numerical_gradient_lm, type(1.0e-3)), true);

    ForwardPropagation forwardpropagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forwardpropagation, is_training);

    BackPropagationLM backpropagation_lm(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate_lm(batch, forwardpropagation, backpropagation_lm);

    EXPECT_EQ(backpropagation_lm.errors.dimension(0), samples_number);
    EXPECT_EQ(backpropagation_lm.errors.dimension(1), outputs_number);

    // const Tensor<type, 2> numerical_jacobian = normalized_squared_error.calculate_numerical_jacobian();
    // const Tensor<type, 2> numerical_hessian = normalized_squared_error.calculate_numerical_hessian();

    // EXPECT_EQ(are_equal(backpropagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-3)), true);
    // EXPECT_EQ(are_equal(backpropagation_lm.hessian, numerical_hessian, type(1.0e-3)), true);
}


TEST(NormalizedSquaredErrorTest, NormalizationCoefficient)
{
    const Index samples_number = 4;
    const Index inputs_number = 4;
    const Index neurons_number = 2;
    const Index outputs_number = 4;

    Tensor<string, 1> uses;

    Tensor<type, 1> targets_mean;
    Tensor<type, 2> target_data;

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();

    uses.resize(8);
    uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

    target_data = dataset.get_data(Dataset::VariableUse::Target);

    Eigen::array<int, 1> dimensions({0});
    targets_mean = target_data.mean(dimensions);

    NeuralNetwork neural_network(
        NeuralNetwork::ModelType::Approximation,
        {inputs_number},
        {neurons_number},
        {outputs_number});
    neural_network.set_parameters_random();

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);

    type normalization_coefficient = normalized_squared_error.calculate_normalization_coefficient(target_data, targets_mean);

    EXPECT_GE(normalization_coefficient, 0);
}
