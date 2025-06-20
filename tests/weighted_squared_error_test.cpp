#include "pch.h"

#include "../opennn/weighted_squared_error.h"

using namespace opennn;

TEST(WeightedSquaredErrorTest, DefaultConstructor)
{
    WeightedSquaredError weighted_squared_error;

    EXPECT_EQ(weighted_squared_error.has_neural_network(), false);
    EXPECT_EQ(weighted_squared_error.has_data_set(), false);
}


TEST(WeightedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    WeightedSquaredError weighted_squared_error(&neural_network, &dataset);

    EXPECT_EQ(weighted_squared_error.has_neural_network(), true);
    EXPECT_EQ(weighted_squared_error.has_data_set(), true);
}


TEST(WeightedSquaredErrorTest, BackPropagateOne)
{
    //Test binary classification trivial

    const Index inputs_number = 1;
    const Index outputs_number = 1;
    const Index samples_number = 1;
    bool is_training = true;

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_constant(type(0));

    const vector<Index> training_samples_indices = dataset.get_sample_indices(Dataset::SampleUse::Training);
    const vector<Index> input_variables_indices = dataset.get_variable_indices(Dataset::VariableUse::Input);
    const vector<Index> target_variables_indices = dataset.get_variable_indices(Dataset::VariableUse::Target);

    Batch batch(samples_number, &dataset);
    batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

    //Neural network

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                 { inputs_number }, {  }, { outputs_number });
    neural_network.set_parameters_random();

    WeightedSquaredError weighted_squared_error(&neural_network, &dataset);

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

    //Loss index

    weighted_squared_error.set_weights();
    BackPropagation back_propagation(samples_number, &weighted_squared_error);
    weighted_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    const Tensor<type, 1> numerical_gradient = weighted_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(back_propagation.errors.dimension(0), samples_number);
    EXPECT_EQ(back_propagation.errors.dimension(1), outputs_number);

    EXPECT_NEAR((abs(back_propagation.error()) - type(0.25)), type(0), NUMERIC_LIMITS_MIN);
    EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), true);
}


TEST(WeightedSquaredErrorTest, BackPropagate)
{
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);

    Dataset data_set(samples_number, {inputs_number}, {outputs_number});
    data_set.set_data_random();

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                 { inputs_number }, { neurons_number }, { outputs_number });

    neural_network.set_parameters_random();

    WeightedSquaredError weighted_squared_error(&neural_network, &data_set);

    const Tensor<type, 1> gradient = weighted_squared_error.calculate_numerical_gradient();
    const Tensor<type, 1> numerical_gradient = weighted_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
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
