//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)
//   artelnics@artelnics.com

#include "normalized_squared_error_test.h"

NormalizedSquaredErrorTest::NormalizedSquaredErrorTest() : UnitTesting()
{
    normalized_squared_error.set(&neural_network, &data_set);

    normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest()
{
}


void NormalizedSquaredErrorTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default

    NormalizedSquaredError normalized_squared_error_1;

    assert_true(!normalized_squared_error_1.has_neural_network(), LOG);
    assert_true(!normalized_squared_error_1.has_data_set(), LOG);

    // Neural network and data set

    NormalizedSquaredError normalized_squared_error_2(&neural_network, &data_set);

    assert_true(normalized_squared_error_2.has_neural_network(), LOG);
    assert_true(normalized_squared_error_2.has_data_set(), LOG);
}


void NormalizedSquaredErrorTest::test_destructor()
{
    cout << "test_destructor\n";

    NormalizedSquaredError* nse = new NormalizedSquaredError;

    delete nse;
}


void NormalizedSquaredErrorTest::test_back_propagate()
{
    cout << "test_back_propagate\n";

    // Empty test does not work
    // normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    // Test approximation trivial
    {
        samples_number = 1;
        inputs_number = 1;
        outputs_number = 1;
        neurons_number = 1;

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

        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
        assert_true(back_propagation.gradient.size() == inputs_number+inputs_number*neurons_number+outputs_number+outputs_number*neurons_number, LOG);

        assert_true(is_zero(back_propagation.gradient) , LOG);
    }

    // Test approximation all random
    {
        samples_number = 1 + rand()%5;
        inputs_number = 1 + rand()%5;
        outputs_number = 1 + rand()%5;
        neurons_number = 1 + rand()%5;

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

        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
    }

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
        assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-3)), LOG);

    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
    }


    // Test forecasting trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error < type(1e-1), LOG);
        assert_true(is_zero(back_propagation.gradient,type(1e-1)), LOG);
    }

    // Test forecasting random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= type(0), LOG);
        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
    }

}


void NormalizedSquaredErrorTest::test_back_propagate_lm()
{
    cout << "test_back_propagate_lm\n";

    normalized_squared_error.set_normalization_coefficient(type(1));

    // Test approximation random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running
        /*
        back_propagation_lm.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();
        jacobian_numerical_differentiation = normalized_squared_error.calculate_jacobian_numerical_differentiation();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-1), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-1)), LOG);
        */
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running
        /*
        back_propagation_lm.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();
        jacobian_numerical_differentiation = normalized_squared_error.calculate_jacobian_numerical_differentiation();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
        */
    }

    // Test multiple classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running
        /*
        back_propagation_lm.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        gradient_numerical_differentiation = normalized_squared_error.calculate_gradient_numerical_differentiation();
        jacobian_numerical_differentiation = normalized_squared_error.calculate_jacobian_numerical_differentiation();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
        */
    }

    // Forecasting incompatible with LM

}


void NormalizedSquaredErrorTest::test_calculate_normalization_coefficient()
{
    cout << "test_calculate_normalization_coefficient\n";

    Index samples_number;
    Index inputs_number;
    Index outputs_number;

    Tensor<string, 1> uses;

    Tensor<type, 1> targets_mean;
    Tensor<type, 2> target_data;

//    type normalization_coefficient;

    // Test

    samples_number = 4;
    inputs_number = 4;
    outputs_number = 4;

    data_set.generate_random_data(samples_number, inputs_number + outputs_number);

    uses.resize(8);
    uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

    data_set.set_columns_uses(uses);

    target_data = data_set.get_target_data();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {samples_number, inputs_number, outputs_number});
    neural_network.set_parameters_random();

//    normalization_coefficient = normalized_squared_error.calculate_normalization_coefficient(target_data, targets_mean);
//    assert_true(normalization_coefficient > 0, LOG);
}


void NormalizedSquaredErrorTest::run_test_case()
{
    cout << "Running normalized squared error test case...\n";

    // Constructor and destructor methods

    test_constructor();

    test_destructor();

    test_calculate_normalization_coefficient();

    // Back-propagation methods

    test_back_propagate();

    test_back_propagate_lm();

    cout << "End of normalized squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lenser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lenser General Public License for more details.

// You should have received a copy of the GNU Lenser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
