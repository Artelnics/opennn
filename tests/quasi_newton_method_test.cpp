//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "quasi_newton_method_test.h"


QuasiNewtonMethodTest::QuasiNewtonMethodTest() : UnitTesting() 
{
    sum_squared_error.set(&neural_network, &data_set);

    quasi_newton_method.set_loss_index_pointer(&sum_squared_error);
}


QuasiNewtonMethodTest::~QuasiNewtonMethodTest()
{

}


void QuasiNewtonMethodTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    QuasiNewtonMethod quasi_newton_method_1;

    assert_true(!quasi_newton_method_1.has_loss_index(), LOG);

    // Loss index constructor

    QuasiNewtonMethod quasi_newton_method_2(&sum_squared_error);
    assert_true(quasi_newton_method_2.has_loss_index(), LOG);
}


void QuasiNewtonMethodTest::test_destructor()
{
    cout << "test_destructor\n";

    QuasiNewtonMethod* quasi_newton_method = new QuasiNewtonMethod;
    delete quasi_newton_method;
}


void QuasiNewtonMethodTest::test_set_inverse_hessian_approximation_method()
{
    cout << "test_set_training_direction_method\n";

    quasi_newton_method.set_inverse_hessian_approximation_method(
                QuasiNewtonMethod::InverseHessianApproximationMethod::BFGS);

    assert_true(
                quasi_newton_method.get_inverse_hessian_approximation_method()
                == QuasiNewtonMethod::InverseHessianApproximationMethod::BFGS, LOG);
}


void QuasiNewtonMethodTest::test_calculate_DFP_inverse_hessian_approximation()
{
    cout << "test_calculate_DFP_inverse_hessian_approximation\n";

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;
    neurons_number = 1;

    // Test

    data_set.set(samples_number, inputs_number, outputs_number);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});

    // Test

    neural_network.set_parameters_constant(type(1));

    quasi_newton_method_data.set(&quasi_newton_method);

    quasi_newton_method.calculate_DFP_inverse_hessian(quasi_newton_method_data);

//    assert_true(are_equal(quasi_newton_method_data.inverse_hessian, inverse_hessian, type(1e-4)), LOG);
}


// @todo
void QuasiNewtonMethodTest::test_calculate_BFGS_inverse_hessian_approximation()
{
    cout << "test_calculate_BFGS_inverse_hessian_approximation\n";

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;
    neurons_number = 1;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(type(1));

    sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::L2);

    quasi_newton_method.calculate_BFGS_inverse_hessian(quasi_newton_method_data);

//    assert_true(are_equal(BFGS_inverse_hessian ,inverse_hessian, type(1e-4)), LOG);
}


// @todo
void QuasiNewtonMethodTest::test_calculate_inverse_hessian_approximation()
{
    cout << "test_calculate_inverse_hessian_approximation\n";

    Tensor<type, 1> old_parameters;
    Tensor<type, 1> old_gradient;
    Tensor<type, 2> old_inverse_hessian;

    Tensor<type, 1> parameters;
    Tensor<type, 1> gradient;
    Tensor<type, 2> inverse_hessian;

    Tensor<type, 2> inverse_hessian_approximation;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(samples_number, inputs_number, outputs_number);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});

    quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::InverseHessianApproximationMethod::DFP);

    neural_network.set_parameters_constant(type(1));

    quasi_newton_method.calculate_inverse_hessian_approximation(quasi_newton_method_data);

//    assert_true(inverse_hessian_approximation == inverse_hessian, LOG);

    quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::InverseHessianApproximationMethod::DFP);

    neural_network.set_parameters_constant(type(1));

    neural_network.set_parameters_constant(type(-0.5));

    quasi_newton_method.calculate_inverse_hessian_approximation(quasi_newton_method_data);

//    assert_true(inverse_hessian_approximation == inverse_hessian, LOG);

    // Test

    quasi_newton_method.calculate_inverse_hessian_approximation(quasi_newton_method_data);
}


void QuasiNewtonMethodTest::test_perform_training()
{   
    cout << "test_perform_training\n";

    type old_error = numeric_limits<float>::max();

    type error;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(1,1,1);
    data_set.set_data_constant(type(1));

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(type(1));

    quasi_newton_method.set_maximum_epochs_number(1);
    quasi_newton_method.set_display(false);
    training_results = quasi_newton_method.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);

    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(-1);

    quasi_newton_method.set_maximum_epochs_number(1);

    training_results = quasi_newton_method.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    quasi_newton_method.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);

    training_results = quasi_newton_method.perform_training();
    error = training_results.get_training_error();

    assert_true(error <= old_error, LOG);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    quasi_newton_method.set_loss_goal(training_loss_goal);
    quasi_newton_method.set_minimum_loss_decrease(0.0);
    quasi_newton_method.set_maximum_epochs_number(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.perform_training();

    assert_true(training_results.get_loss() <= training_loss_goal, LOG);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    quasi_newton_method.set_loss_goal(type(0));
    quasi_newton_method.set_minimum_loss_decrease(minimum_loss_decrease);
    quasi_newton_method.set_maximum_epochs_number(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.perform_training();

    assert_true(training_results.get_loss_decrease() <= minimum_loss_decrease, LOG);

}


void QuasiNewtonMethodTest::run_test_case()
{
    cout << "Running quasi-Newton method test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set_inverse_hessian_approximation_method();

    // Training methods

    test_calculate_DFP_inverse_hessian_approximation();

    test_calculate_BFGS_inverse_hessian_approximation();

    test_calculate_inverse_hessian_approximation();

    test_perform_training();

    cout << "End of quasi-Newton method test case.\n\n";
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
