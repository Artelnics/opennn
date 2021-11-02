//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error_test.h"

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

void SumSquaredErrorTest::test_calculate_error()
{
    cout << "test_calculate_error\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Index neurons_number;

    Tensor<type, 2> data;

    Tensor<type, 1> parameters;

    // Test

    samples_number = 2;
    inputs_number = 3;
    targets_number = 1;
    neurons_number = 2;

    data_set.set(samples_number, inputs_number, targets_number);

    data.resize(samples_number,inputs_number + targets_number);

    data.setValues({{type(1), type(2), type(3),type(4)}, {type(2), type(4), type(6), type(8)}});
    data_set.set_data(data);
    data_set.print();

    batch.set(samples_number, &data_set);

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, targets_number});

    neural_network.set_parameters_constant(type(1));

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);

    sum_squared_error.calculate_errors(batch, forward_propagation, back_propagation);
    sum_squared_error.calculate_error(batch, forward_propagation, back_propagation);

    cout << "forward propagation information" << endl;
    forward_propagation.print();

    cout << "backward propagation information" << endl;
    back_propagation.print();

    cout << abs(back_propagation.error - type(12.7329755)) << endl;
    assert_true(abs(back_propagation.error - type(12.7329755)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test
/*
    samples_number = 2;
    inputs_number = 2;
    targets_number = 2;

    data.resize(samples_number, inputs_number + targets_number);

    data.setValues({{type(1), type(2), type(3), type(4)}, {type(2), type(4), type(6), type(8)}});

    data_set.set_data(data);
    data_set.print_data();

    neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, targets_number});
    neural_network.set_parameters_random();

    parameters = neural_network.get_parameters();

    batch.set(samples_number, &data_set);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);

    sum_squared_error.calculate_error(batch, forward_propagation, back_propagation);

    assert_true(abs(back_propagation.error) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(back_propagation.error == type(1.0), LOG);
*/
}

void SumSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   Index samples_number;

   Index inputs_number;
   Index targets_number;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> input_variables_indices;
   Tensor<Index,1> target_variables_indices;

   // Test

   samples_number = 1;
   inputs_number = 2;
   targets_number = 3;

   data_set.set(1, inputs_number, targets_number);
   data_set.set_data_constant(type(0));
   data_set.set_training();

   batch.set(samples_number, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   // Neural network

   neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, targets_number});
   neural_network.set_parameters_constant(type(0));

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.gradient(0) == 0.0, LOG);
   assert_true(back_propagation.gradient(1) == 0.0, LOG);

}

void SumSquaredErrorTest::test_calculate_error_gradient_lm()
{
   cout << "test_calculate_error_gradient_lm\n";

   Tensor<type, 2> data;

   Tensor<type, 1> parameters;

   Index samples_number;

   Index inputs_number;
   Index targets_number;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> input_variables_indices;
   Tensor<Index,1> target_variables_indices;

   // Test

   samples_number = 1;

   inputs_number = 2;
   targets_number = 3;

   data_set.set(1, inputs_number, targets_number);
   data_set.set_data_constant(type(0));
   data_set.set_training();

   batch.set(samples_number, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   // Neural network

   neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
   neural_network.set_parameters_constant(type(0));

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   back_propagation_lm.set(training_samples_indices.size(), &sum_squared_error);

   sum_squared_error.calculate_squared_errors_jacobian_lm(batch, forward_propagation, back_propagation_lm);
//   sum_squared_error.calculate_gradient(batch, forward_propagation, loss_index_back_propagation_lm);

   assert_true(back_propagation_lm.gradient(0) == 0.0, LOG);
   assert_true(back_propagation_lm.gradient(1) == 0.0, LOG);

}


void SumSquaredErrorTest::test_back_propagate_approximation_zero()
{
    cout << "test_back_propagate_approximation_zero\n";

    // Test

    inputs_number = 1;
    outputs_number = 1;
    samples_number = 1;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_constant(type(0));

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, outputs_number});

    neural_network.set_parameters_constant(type(0));

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation.error < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(is_zero(back_propagation.gradient), LOG);
}


void SumSquaredErrorTest::test_back_propagate_approximation_random()
{
    cout << "test_back_propagate_approximation_random\n";

    // Test

    samples_number = 1 + rand()%10;
    inputs_number = 1 + rand()%5;
    outputs_number = 1 + rand()%5;
    neurons_number = 1 + rand()%5;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
    neural_network.set_parameters_random();

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    gradient_numerical_differentiation = sum_squared_error.calculate_gradient_numerical_differentiation();

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation.error >= type(0), LOG);

    assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);    

    cout << back_propagation.gradient << endl;
    cout << "XXXXXXXX" << endl;
    cout << gradient_numerical_differentiation << endl;
    system("pause");
}


void SumSquaredErrorTest::test_back_propagate_binary_classification_zero()
{
    cout << "test_back_propagate_binary_classification_zero\n";

    // Test

    inputs_number = 1;
    outputs_number = 1;
    samples_number = 1;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_constant(type(0));

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, outputs_number});

    neural_network.set_parameters_constant(type(0));

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation.errors.dimension(0) == 1, LOG);
    assert_true(back_propagation.errors.dimension(1) == 1, LOG);
    assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);
}


void SumSquaredErrorTest::test_back_propagate_binary_classification_random()
{
    cout << "test_back_propagate_binary_classification_random\n";

    // Test

    samples_number = 1 + rand()%10;
    inputs_number = 1 + rand()%10;
    outputs_number = 1;
    neurons_number = 1 + rand()%10;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_binary_random();

    data_set.set_training();

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
    neural_network.set_parameters_random();

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    gradient_numerical_differentiation = sum_squared_error.calculate_gradient_numerical_differentiation();

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation.error >= 0, LOG);

    assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, 1.0e-2), LOG);

}


void SumSquaredErrorTest::test_back_propagate_forecasting_zero()
{
    cout << "test_back_propagate_forecasting_zero\n";

    // Test

    inputs_number = 1;
    outputs_number = 1;
    samples_number = 1;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_constant(type(0));

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, outputs_number});

    neural_network.set_parameters_constant(type(0));

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation.error < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(is_zero(back_propagation.gradient), LOG);
}


void SumSquaredErrorTest::test_back_propagate_forecasting_random()
{
    cout << "test_back_propagate_approximation_random\n";

    // Test

    samples_number = 1 + rand()%10;
    inputs_number = 1 + rand()%10;
    outputs_number = 1 + rand()%10;
    neurons_number = 1 + rand()%10;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, neurons_number, outputs_number});
    neural_network.set_parameters_random();

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    gradient_numerical_differentiation = sum_squared_error.calculate_gradient_numerical_differentiation();

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation.error >= type(0), LOG);

    assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
}


void SumSquaredErrorTest::test_back_propagate_lm_approximation_random()
{
    cout << "test_back_propagate_lm_approximation_random\n";

    sum_squared_error.set_regularization_method(SumSquaredError::RegularizationMethod::NoRegularization);

    // Test

//    samples_number = 1 + rand()%10;
//    inputs_number = 1 + rand()%10;
//    outputs_number = 1 + rand()%10;
//    neurons_number = 1 + rand()%10;

    samples_number = 3;
    inputs_number = 3;
    outputs_number = 4;
    neurons_number = 2;


    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
    neural_network.set_parameters_random();

    batch.set(samples_number, &data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    back_propagation_lm.set(samples_number, &sum_squared_error);
    sum_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

    gradient_numerical_differentiation = sum_squared_error.calculate_gradient_numerical_differentiation();

    assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

    assert_true(back_propagation_lm.error >= type(0), LOG);

    assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-4), LOG);

    assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);

    jacobian_numerical_differentiation = sum_squared_error.calculate_jacobian_numerical_differentiation();

    assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
}


void SumSquaredErrorTest::run_test_case()
{
   cout << "Running sum squared error test case...\n";

   // Constructor and destructor methods

//   test_back_propagate_approximation_random();

//   test_back_propagate_binary_classification_random();

//   test_back_propagate_forecasting_zero();
//   test_back_propagate_forecasting_random();

//   test_back_propagate_lm_approximation_random();

   // Error methods

   test_calculate_error();

   test_calculate_error_gradient(); // Passed

   cout << "End of sum squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
