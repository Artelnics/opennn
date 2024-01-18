//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N      T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "adaptive_moment_estimation_test.h"


AdaptiveMomentEstimationTest::AdaptiveMomentEstimationTest() : UnitTesting()
{
    sum_squared_error.set(&neural_network, &data_set);

    adaptive_moment_estimation.set_loss_index_pointer(&sum_squared_error);
}


AdaptiveMomentEstimationTest::~AdaptiveMomentEstimationTest()
{
}


void AdaptiveMomentEstimationTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    AdaptiveMomentEstimation adaptive_moment_estimation_1;
    assert_true(!adaptive_moment_estimation_1.has_loss_index(), LOG);

    // Loss index constructor

    AdaptiveMomentEstimation adaptive_moment_estimation_2(&sum_squared_error);
    assert_true(adaptive_moment_estimation_2.has_loss_index(), LOG);
}


void AdaptiveMomentEstimationTest::test_destructor()
{
    cout << "test_destructor\n";

    AdaptiveMomentEstimation* adaptive_moment_estimation = new AdaptiveMomentEstimation;

    delete adaptive_moment_estimation;
}


void AdaptiveMomentEstimationTest::test_perform_training()
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

    adaptive_moment_estimation.set_maximum_epochs_number(1);
    adaptive_moment_estimation.set_display(false);
    training_results = adaptive_moment_estimation.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);

    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(-1);

    adaptive_moment_estimation.set_maximum_epochs_number(1);

    training_results = adaptive_moment_estimation.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    adaptive_moment_estimation.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);

    training_results = adaptive_moment_estimation.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.05);

    adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    adaptive_moment_estimation.set_maximum_epochs_number(10000);
    adaptive_moment_estimation.set_maximum_time(1000.0);

    training_results = adaptive_moment_estimation.perform_training();

    assert_true(training_results.get_loss() <= training_loss_goal, LOG);
}


void AdaptiveMomentEstimationTest::run_test_case()
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
