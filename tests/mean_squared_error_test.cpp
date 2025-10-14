#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/dataset.h"
#include "../opennn/language_dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/standard_networks.h"

using namespace opennn;

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    MeanSquaredError mean_squared_error;

    EXPECT_EQ(mean_squared_error.has_neural_network(), false);
    EXPECT_EQ(mean_squared_error.has_dataset(), false);
}


TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;
    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    EXPECT_EQ(mean_squared_error.has_neural_network(), true);
    EXPECT_EQ(mean_squared_error.has_dataset(), true);
}


TEST(MeanSquaredErrorTest, BackPropagate)
{
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random(); 
    dataset.set_sample_uses("Training"); 

    NeuralNetwork neural_network; 
    neural_network.add_layer(make_unique<Dense2d>(dimensions{ inputs_number }, dimensions{ targets_number })); 
    neural_network.set_parameters_random();

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();

    EXPECT_GE(error, 0);

    const Tensor<type, 1> gradient = mean_squared_error.calculate_gradient();
    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateLm)
{
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    ApproximationNetwork neural_network({ inputs_number }, { neurons_number }, { outputs_number });

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_uses("Training");

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices("Training"),
               dataset.get_variable_indices("Input"),
               dataset.get_variable_indices("Target"));

    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    BackPropagation back_propagation(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    BackPropagationLM back_propagation_lm(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

    const Tensor<type, 2> numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();
    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();
    const Tensor<type, 2> numerical_hessian = mean_squared_error.calculate_numerical_hessian();

    EXPECT_NEAR(back_propagation_lm.error(), back_propagation.error(), type(1.0e-3));
    EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian), true);
    EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1e-2)), true);
    EXPECT_EQ(are_equal(back_propagation_lm.hessian, numerical_hessian, type(1.0e-1)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateMultiheadAttention)
{
    const Index batch_size = get_random_index(1, 4);
    const Index sequence_length = get_random_index(1, 4);
    const Index heads_number = get_random_index(1, 4);
    const Index head_dimension = get_random_index(1, 4);
    const Index embedding_dimension = heads_number * head_dimension;

    const dimensions sample_input_dimensions = { sequence_length, embedding_dimension };

    const dimensions sample_target_dimensions = { sequence_length * embedding_dimension };

    Dataset dataset(batch_size, sample_input_dimensions, sample_target_dimensions);
    dataset.set_data_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(dataset.get_input_dimensions(), heads_number));
    neural_network.add_layer(make_unique<Flatten<3>>(neural_network.get_output_dimensions()));

    Layer* base_layer_ptr = neural_network.get_layer(0).get();
    MultiHeadAttention* mha_layer = dynamic_cast<MultiHeadAttention*>(base_layer_ptr);

    neural_network.set_parameters_random();

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    const Tensor<type, 1> analytical_gradient = mean_squared_error.calculate_gradient();
    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

    EXPECT_GE(error, 0.0) << "MSE must be positive";
    EXPECT_TRUE(isfinite(error)) << "Error must be finite";
    ASSERT_EQ(analytical_gradient.size(), numerical_gradient.size());
    ASSERT_EQ(analytical_gradient.size(), mha_layer->get_parameters_number());
    EXPECT_TRUE(are_equal(analytical_gradient, numerical_gradient, type(1.0e-3)));
}