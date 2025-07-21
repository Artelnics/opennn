#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/weighted_squared_error.h"

using namespace opennn;

TEST(WeightedSquaredErrorTest, DefaultConstructor)
{
    WeightedSquaredError weighted_squared_error;

    EXPECT_EQ(weighted_squared_error.has_neural_network(), false);
    EXPECT_EQ(weighted_squared_error.has_dataset(), false);
}


TEST(WeightedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    WeightedSquaredError weighted_squared_error(&neural_network, &dataset);

    EXPECT_EQ(weighted_squared_error.has_neural_network(), true);
    EXPECT_EQ(weighted_squared_error.has_dataset(), true);
}


TEST(WeightedSquaredErrorTest, BackPropagate)
{
    // const Index samples_number = get_random_index(2, 10);
    // const Index inputs_number = get_random_index(1, 10);
    // const Index neurons_number = get_random_index(1, 10);
    // const Index outputs_number = 1;

    // Dataset data_set(samples_number, {inputs_number}, {outputs_number});

    // data_set.set_data_binary_classification();

    // NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
    //                              { inputs_number }, { neurons_number }, { outputs_number });

    // neural_network.set_parameters_random();

    // WeightedSquaredError weighted_squared_error(&neural_network, &data_set);

    // const Tensor<type, 1> gradient = weighted_squared_error.calculate_gradient();
    // const Tensor<type, 1> numerical_gradient = weighted_squared_error.calculate_numerical_gradient();

    // EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
    // EXPECT_EQ(back_propagation.errors.dimension(0), samples_number);
    // EXPECT_EQ(back_propagation.errors.dimension(1), outputs_number);

    // EXPECT_NEAR((abs(back_propagation.error()) - type(0.25)), type(0), NUMERIC_LIMITS_MIN);
    // EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), true);

    // //Test binary classification random samples, inputs, outputs, neurons

    // const Index samples_number_rand = 1 + rand()%10;
    // const Index inputs_number_rand = 1 + rand()%10;
    // const Index neurons_number = 1 + rand()%10;
    // const Index outputs_number_rand = 1+ rand()%10;

    // //Data set

    // Dataset data_set_rand(samples_number_rand, {inputs_number_rand}, {outputs_number_rand});
    // data_set_rand.set_data_random();

    // const vector<Index> training_samples_indices_rand = data_set_rand.get_sample_indices("Training");
    // const vector<Index> input_variables_indices_rand = data_set_rand.get_variable_indices("Input");
    // const vector<Index> target_variables_indices_rand = data_set_rand.get_variable_indices("Target");

    // Batch batch_rand(samples_number_rand, &data_set_rand);
    // batch_rand.fill(training_samples_indices_rand, input_variables_indices_rand, {}, target_variables_indices_rand);

    // //Neural Network

    // NeuralNetwork neural_network_rand(NeuralNetwork::ModelType::Classification,
    //                              { inputs_number_rand }, { neurons_number }, { outputs_number_rand });
    // neural_network_rand.set_parameters_random();

    // WeightedSquaredError weighted_squared_error_rand(&neural_network_rand, &data_set_rand);

    // ForwardPropagation forward_propagation_rand(samples_number_rand, &neural_network_rand);
    // neural_network_rand.forward_propagate(batch_rand.get_input_pairs(), forward_propagation_rand, is_training);

    // //Loss index

    // weighted_squared_error_rand.set_weights();
    // BackPropagation back_propagation_rand(samples_number_rand, &weighted_squared_error_rand);

    // weighted_squared_error_rand.back_propagate(batch_rand, forward_propagation_rand, back_propagation_rand);

    // const Tensor<type, 1> numerical_gradient_rand = weighted_squared_error_rand.calculate_numerical_gradient();

    // EXPECT_EQ(back_propagation_rand.errors.dimension(0), samples_number_rand);
    // EXPECT_EQ(back_propagation_rand.errors.dimension(1), outputs_number_rand);

    // EXPECT_EQ(are_equal(back_propagation_rand.gradient, numerical_gradient_rand, type(1.0e-2)), true);
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lewser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lewser General Public License for more details.

// You should have received a copy of the GNU Lewser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
