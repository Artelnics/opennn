#include "pch.h"

#include "../opennn/tensor_utilities.h"
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
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {targets_number});

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);
    normalized_squared_error.set_normalization_coefficient();
    normalized_squared_error.set_regularization_weight(0.0);

    const type error = normalized_squared_error.calculate_numerical_error();
    const VectorR numerical_gradient = normalized_squared_error.calculate_numerical_gradient();
    const VectorR gradient = normalized_squared_error.calculate_gradient();

    EXPECT_GE(error, 0);
    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(NormalizedSquaredErrorTest, BackPropagateLM)
{
    const Index samples_number = 10;
    const Index inputs_number = 4;
    const Index outputs_number = 1;
    const Index neurons_number = 3;
    bool is_training = true;

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices("Training"),
               dataset.get_variable_indices("Input"),
               dataset.get_variable_indices("Decoder"),
               dataset.get_variable_indices("Target"));

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {outputs_number});

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);
    normalized_squared_error.set_normalization_coefficient();
    normalized_squared_error.set_regularization_weight(0.0);

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_inputs(), forward_propagation, is_training);

    BackPropagation back_propagation(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    BackPropagationLM back_propagation_lm(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

    const VectorR numerical_gradient = normalized_squared_error.calculate_numerical_gradient();
    const MatrixR numerical_jacobian = normalized_squared_error.calculate_numerical_jacobian();

    EXPECT_NEAR(back_propagation_lm.error, back_propagation.error, type(1.0e-3));

    EXPECT_TRUE(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1e-1)));

    EXPECT_TRUE(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1e-1)));

    const MatrixR expected_hessian_gn = numerical_jacobian.transpose() * numerical_jacobian;
    EXPECT_TRUE(are_equal(back_propagation_lm.hessian, expected_hessian_gn, type(1e-1)));
}


TEST(NormalizedSquaredErrorTest, NormalizationCoefficient)
{
    const Index samples_number = 4;
    const Index inputs_number = 4;
    const Index neurons_number = 2;
    const Index outputs_number = 4;

    VectorR targets_mean;
    MatrixR target_data;

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();

    // vector<string> uses = {"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"};
    // dataset.set_variable_roles(uses);

    target_data = dataset.get_feature_data("Target");

    targets_mean = target_data.colwise().mean().transpose();

    ApproximationNetwork neural_network({inputs_number}, {neurons_number}, {outputs_number});
    neural_network.set_parameters_random();

    NormalizedSquaredError normalized_squared_error(&neural_network, &dataset);

    type normalization_coefficient = normalized_squared_error.calculate_normalization_coefficient(target_data, targets_mean);

    EXPECT_GE(normalization_coefficient, 0);
}
