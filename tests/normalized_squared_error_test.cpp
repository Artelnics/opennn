#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/normalized_squared_error.h"

using namespace opennn;

TEST(NormalizedSquaredErrorTest, DefaultConstructor)
{
    NormalizedSquaredError normalized_squared_error;

    EXPECT_EQ(normalized_squared_error.has_neural_network(), false);
    EXPECT_EQ(normalized_squared_error.has_dataset(), false);
}


TEST(NormalizedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);

    EXPECT_EQ(normalized_squared_error.has_neural_network(), true);
    EXPECT_EQ(normalized_squared_error.has_dataset(), true);
}


TEST(NormalizedSquaredErrorTest, BackPropagate)
{
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_uses("Training");

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {targets_number});

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);
    normalized_squared_error.set_normalization_coefficient();

    const type error = normalized_squared_error.calculate_numerical_error();
    const Tensor<type, 1> numerical_gradient = normalized_squared_error.calculate_numerical_gradient();
    const Tensor<type, 1> gradient = normalized_squared_error.calculate_gradient();

    EXPECT_GE(error, 0);
    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
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
    dataset.set_sample_uses("Training");

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices("Training"),
               dataset.get_variable_indices("Input"),
               dataset.get_variable_indices("Target"));

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {outputs_number});

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);
    normalized_squared_error.set_normalization_coefficient();

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    BackPropagation back_propagation(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    BackPropagationLM back_propagation_lm(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

    const Tensor<type, 1> numerical_gradient_lm = normalized_squared_error.calculate_numerical_gradient();
    const Tensor<type, 2> numerical_jacobian_lm = normalized_squared_error.calculate_numerical_jacobian();
    const Tensor<type, 2> numerical_hessian_lm = normalized_squared_error.calculate_numerical_hessian();

    const Tensor<type, 1> gradient_lm = normalized_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient_lm, numerical_gradient_lm, type(1.0e-3)), true);
    EXPECT_NEAR(back_propagation_lm.error(), back_propagation.error(), type(1.0e-3));

    EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient_lm, type(1e-2)), true);
    EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian_lm, type(1.0e-3)), true);
    EXPECT_EQ(are_equal(back_propagation_lm.hessian, numerical_hessian_lm, type(1.0e-3)), true);
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

    target_data = dataset.get_data_variables("Target");

    Eigen::array<int, 1> dimensions({0});
    targets_mean = target_data.mean(dimensions);

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {outputs_number});
    neural_network.set_parameters_random();

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);

    type normalization_coefficient = normalized_squared_error.calculate_normalization_coefficient(target_data, targets_mean);

    EXPECT_GE(normalization_coefficient, 0);
}
