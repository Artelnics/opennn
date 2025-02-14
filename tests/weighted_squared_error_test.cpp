#include "pch.h"

#include "../opennn/weighted_squared_error.h"

TEST(WeightedSquaredErrorTest, DefaultConstructor)
{
    WeightedSquaredError weighted_squared_error;

    EXPECT_EQ(weighted_squared_error.has_neural_network(), false);
    EXPECT_EQ(weighted_squared_error.has_data_set(), false);
}


TEST(WeightedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    DataSet data_set;

    WeightedSquaredError weighted_squared_error(&neural_network, &data_set);

    EXPECT_EQ(weighted_squared_error.has_neural_network(), true);
    EXPECT_EQ(weighted_squared_error.has_data_set(), true);
}

/*

void WeightedSquaredErrorTest::test_back_propagate()
{
    weighted_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {}, {outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        weighted_squared_error.set_weights();

        back_propagation.set(samples_number, &weighted_squared_error);
        weighted_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = weighted_squared_error.calculate_numerical_gradient();


        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.errors.dimension(0) == 1);
        EXPECT_EQ(back_propagation.errors.dimension(1) == 1);
        EXPECT_EQ(back_propagation.error() - type(0.25) < type(NUMERIC_LIMITS_MIN));

        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)));
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1;
        neurons_number = 1 + rand()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, {} ,target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        weighted_squared_error.set_weights();

        back_propagation.set(samples_number, &weighted_squared_error);
        weighted_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = weighted_squared_error.calculate_numerical_gradient();


        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)));
    }
}

}
*/
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
