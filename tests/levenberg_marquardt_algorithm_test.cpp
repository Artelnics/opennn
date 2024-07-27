//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G   M A R Q U A R D T   A L G O R I T H M   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "levenberg_marquardt_algorithm_test.h"


LevenbergMarquardtAlgorithmTest::LevenbergMarquardtAlgorithmTest() : UnitTesting() 
{
    sum_squared_error.set(&neural_network, &data_set);

    levenberg_marquardt_algorithm.set_loss_index(&sum_squared_error);

    levenberg_marquardt_algorithm.set_display(false);
}


LevenbergMarquardtAlgorithmTest::~LevenbergMarquardtAlgorithmTest()
{
}


void LevenbergMarquardtAlgorithmTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm_1;

    assert_true(!levenberg_marquardt_algorithm_1.has_loss_index(), LOG);

    // Loss index constructor

    LevenbergMarquardtAlgorithm lma2(&sum_squared_error);

    assert_true(lma2.has_loss_index(), LOG);
}


void LevenbergMarquardtAlgorithmTest::test_destructor()
{
    cout << "test_destructor\n";

    // Test

    LevenbergMarquardtAlgorithm* lma = new LevenbergMarquardtAlgorithm;

    delete lma;

    // Test

    LevenbergMarquardtAlgorithm* lma2 = new LevenbergMarquardtAlgorithm(&sum_squared_error);

    delete lma2;
}


void LevenbergMarquardtAlgorithmTest::test_perform_training()
{
    cout << "test_perform_training\n";

    type old_error = numeric_limits<float>::max();

    TrainingResults training_results;

    Index samples_number;
    Index inputs_number;
    Index outputs_number;

    type error;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(1,1,1);
    data_set.set_data_constant(type(1));

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(type(1));

    levenberg_marquardt_algorithm.set_maximum_epochs_number(1);
    levenberg_marquardt_algorithm.set_display(false);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);
    */
    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, outputs_number});
    neural_network.set_parameters_constant(-1);

    levenberg_marquardt_algorithm.set_maximum_epochs_number(1);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);
    */
    // Test

    old_error = error;

    levenberg_marquardt_algorithm.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();
    error = training_results.get_training_error();

    assert_true(error <= old_error, LOG);
    */
    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(training_loss_goal);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();

    assert_true(training_results.get_training_error() <= training_loss_goal, LOG);
    */
    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(type(0));
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(minimum_loss_decrease);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();

    assert_true(levenberg_marquardt_algorithm.get_minimum_loss_decrease() <= minimum_loss_decrease, LOG);
    */
}


void LevenbergMarquardtAlgorithmTest::run_test_case()
{
    cout << "Running Levenberg-Marquardt algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Training methods

    test_perform_training();

    cout << "End of Levenberg-Marquardt algorithm test case.\n\n";
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
