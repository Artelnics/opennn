//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   D E S C E N T   T E S T   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "gradient_descent_test.h"


GradientDescentTest::GradientDescentTest() : UnitTesting()
{
    sum_squared_error.set(&neural_network, &data_set);

    gradient_descent.set_loss_index_pointer(&sum_squared_error);
}


GradientDescentTest::~GradientDescentTest()
{
}


void GradientDescentTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    GradientDescent gradient_descent_1;
    assert_true(!gradient_descent_1.has_loss_index(), LOG);

    // Loss index constructor

    GradientDescent gradient_descent_2(&sum_squared_error);
    assert_true(gradient_descent_2.has_loss_index(), LOG);
}

void GradientDescentTest::test_destructor()
{
    cout << "test_destructor\n";

    GradientDescent* gradient_descent = new GradientDescent;

    delete gradient_descent;
}


void GradientDescentTest::test_perform_training()
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

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(type(1));

    gradient_descent.set_maximum_epochs_number(1);
    gradient_descent.set_display(false);
    training_results = gradient_descent.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);

    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(-1);

    gradient_descent.set_maximum_epochs_number(1);

    training_results = gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    gradient_descent.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);

    training_results = gradient_descent.perform_training();
    error = training_results.get_training_error();

    cout << "error: " << endl << error << endl;
    cout << "old_error: " << endl << old_error << endl;

    assert_true(error < old_error, LOG);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    gradient_descent.set_loss_goal(training_loss_goal);
    gradient_descent.set_minimum_loss_decrease(0.0);
    gradient_descent.set_maximum_epochs_number(1000);
    gradient_descent.set_maximum_time(1000.0);

    training_results = gradient_descent.perform_training();

    assert_true(training_results.get_loss() <= training_loss_goal, LOG);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    gradient_descent.set_loss_goal(type(0));
    gradient_descent.set_minimum_loss_decrease(minimum_loss_decrease);
    gradient_descent.set_maximum_epochs_number(1000);
    gradient_descent.set_maximum_time(1000.0);

    training_results = gradient_descent.perform_training();

    assert_true(training_results.get_loss_decrease() <= minimum_loss_decrease, LOG);
}


void GradientDescentTest::run_test_case()
{
    cout << "Running gradient descent test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Training methods

    test_perform_training();

    cout << "End of gradient descent test case.\n\n";
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
