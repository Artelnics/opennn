//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error_test.h"
#include "../opennn/back_propagation.h"

SumSquaredErrorTest::SumSquaredErrorTest() : UnitTesting() 
{
    sum_squared_error.set(&neural_network, &data_set);

    sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


SumSquaredErrorTest::~SumSquaredErrorTest() 
{
}


void SumSquaredErrorTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default

    SumSquaredError sum_squared_error_1;

    assert_true(!sum_squared_error_1.has_neural_network(), LOG);
    assert_true(!sum_squared_error_1.has_data_set(), LOG);

    // Neural network and data set

    SumSquaredError sum_squared_error_4(&neural_network, &data_set);

    assert_true(sum_squared_error_4.has_neural_network(), LOG);
    assert_true(sum_squared_error_4.has_data_set(), LOG);
}


void SumSquaredErrorTest::test_destructor()
{
    cout << "test_destructor\n";

    SumSquaredError* sum_squared_error = new SumSquaredError;

    delete sum_squared_error;
}

void SumSquaredErrorTest::test_back_propagate()
{
    cout << "test_back_propagate\n";

    // Test approximation all zero
    {
        samples_number = 1;
        inputs_number = 1;
        outputs_number = 1;
        neurons_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();

        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
        assert_true(back_propagation.gradient.size() == inputs_number+inputs_number*neurons_number+outputs_number+outputs_number*neurons_number, LOG);

        assert_true(is_zero(back_propagation.gradient) , LOG);
    }

    // Test approximation all random
    {
        samples_number = type(1) + rand() % 5;
        inputs_number = type(1) + rand()%5;
        outputs_number = type(1) + rand()%5;
        neurons_number = type(1) + rand()%5;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
    }

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
        assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
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
        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
    }

    // Test forecasting trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Forecasting, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        /*sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error < type(1e-3), LOG);
        assert_true(is_zero(back_propagation.gradient,type(1e-3)), LOG);*/
    }

    // Test forecasting random samples, inputs, outputs, neurons
    {
        samples_number = 1 + arc4random()%10;
        inputs_number = 1 + arc4random()%10;
        outputs_number = 1 + arc4random()%10;
        neurons_number = 1 + arc4random()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();
        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Forecasting, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        //neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        /*back_propagation.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= type(0), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);*/
    }

}


void SumSquaredErrorTest::test_back_propagate_lm()
{
    cout << "test_back_propagate_lm\n";

    // Test approximation random samples, inputs, outputs, neurons
    {
        samples_number = 1 + arc4random()%10;
        inputs_number = 1 + arc4random()%10;
        outputs_number = 1 + arc4random()%10;
        neurons_number = 1 + arc4random()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();
        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        /*sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running

        back_propagation_lm.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();
        numerical_jacobian = sum_squared_error.calculate_numerical_jacobian();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-1), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-1)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-1)), LOG);*/
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = type(1) + arc4random()%10;
        inputs_number = type(1) + arc4random()%10;
        outputs_number = type(1) + arc4random()%10;
        neurons_number = type(1) + arc4random()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        /*sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running

        back_propagation_lm.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();
        numerical_jacobian = sum_squared_error.calculate_numerical_jacobian();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)), LOG);*/
    }

    // Test multiple classification random samples, inputs, outputs, neurons
    {
        samples_number = type(1) + arc4random()%10;
        inputs_number = type(1) + arc4random()%10;
        outputs_number = type(1) + arc4random()%10;
        neurons_number = type(1) + arc4random()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();
        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &sum_squared_error);
        /*sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running

        back_propagation_lm.set(samples_number, &sum_squared_error);
        sum_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = sum_squared_error.calculate_numerical_gradient();
        numerical_jacobian = sum_squared_error.calculate_numerical_jacobian();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)), LOG);*/

    }

}


void SumSquaredErrorTest::run_test_case()
{
    cout << "Running sum squared error test case...\n";

    // Constructor and destructor methods

    test_constructor();

    test_destructor();

    // Back propagate

    test_back_propagate();

    test_back_propagate_lm();

    cout << "End of sum squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

