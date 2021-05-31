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

    levenberg_marquardt_algorithm.set_loss_index_pointer(&sum_squared_error);
}


LevenbergMarquardtAlgorithmTest::~LevenbergMarquardtAlgorithmTest()
{
}


void LevenbergMarquardtAlgorithmTest::test_constructor()
{
   cout << "test_constructor\n"; 

   // Default constructor

   LevenbergMarquardtAlgorithm lma1; 
   assert_true(!lma1.has_loss_index(), LOG);

   // Loss index constructor

   LevenbergMarquardtAlgorithm lma2(&sum_squared_error);
   assert_true(lma2.has_loss_index(), LOG);
}


void LevenbergMarquardtAlgorithmTest::test_perform_training()
{
   cout << "test_perform_training\n";

   Tensor<type, 1> gradient;

   levenberg_marquardt_algorithm.set_display(false);

   type old_loss;
   type loss;
   type minimum_parameters_increment_norm;
   type training_loss_goal;
   type minimum_loss_decrease;
   type gradient_norm_goal;
   type gradient_norm = 0;

   // Test

//   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
//   neural_network.set_parameters_random(0.0, 1.0e-3);

   data_set.set(1, 1, 2);
//   data_set.randomize_data_normal(0.0, 1.0e-3);

//   old_loss = sum_squared_error.calculate_training_loss();

//   levenberg_marquardt_algorithm.perform_training();

//   loss = sum_squared_error.calculate_training_loss();

//   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   minimum_parameters_increment_norm = 100.0;

   levenberg_marquardt_algorithm.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   levenberg_marquardt_algorithm.set_loss_goal(0.0);
   levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
   levenberg_marquardt_algorithm.set_gradient_norm_goal(0.0);
   levenberg_marquardt_algorithm.set_maximum_epochs_number(10);
   levenberg_marquardt_algorithm.set_maximum_time(10.0);

//   levenberg_marquardt_algorithm.perform_training();

   // Performance goal

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   training_loss_goal = 100.0;

   levenberg_marquardt_algorithm.set_minimum_parameters_increment_norm(0.0);
   levenberg_marquardt_algorithm.set_loss_goal(training_loss_goal);
   levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
   levenberg_marquardt_algorithm.set_gradient_norm_goal(0.0);
   levenberg_marquardt_algorithm.set_maximum_epochs_number(10);
   levenberg_marquardt_algorithm.set_maximum_time(10.0);

//   levenberg_marquardt_algorithm.perform_training();

//   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < training_loss_goal, LOG);

   // Minimum loss increas

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   minimum_loss_decrease = 100.0;

   levenberg_marquardt_algorithm.set_minimum_parameters_increment_norm(0.0);
   levenberg_marquardt_algorithm.set_loss_goal(0.0);
   levenberg_marquardt_algorithm.set_minimum_loss_decrease(minimum_loss_decrease);
   levenberg_marquardt_algorithm.set_gradient_norm_goal(0.0);
   levenberg_marquardt_algorithm.set_maximum_epochs_number(10);
   levenberg_marquardt_algorithm.set_maximum_time(10.0);

//   levenberg_marquardt_algorithm.perform_training();

   // Gradient norm goal 

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   gradient_norm_goal = 1.0e6;

   levenberg_marquardt_algorithm.set_minimum_parameters_increment_norm(0.0);
   levenberg_marquardt_algorithm.set_loss_goal(0.0);
   levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
   levenberg_marquardt_algorithm.set_gradient_norm_goal(gradient_norm_goal);
   levenberg_marquardt_algorithm.set_maximum_epochs_number(10);
   levenberg_marquardt_algorithm.set_maximum_time(10.0);

//   levenberg_marquardt_algorithm.perform_training();

//   gradient = sum_squared_error.calculate_training_loss_gradient();
//   gradient_norm = l2_norm(gradient);

   assert_true(gradient_norm < gradient_norm_goal, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_resize_training_error_history()
{
   cout << "test_resize_training_error_history\n";

   TrainingResults training_results;//(&levenberg_marquardt_algorithm);

   training_results.resize_training_error_history(1);

   assert_true(training_results.training_error_history.size() == 1, LOG);
   assert_true(training_results.selection_error_history.size() == 1, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_perform_Householder_QR_decomposition()
{
   cout << "test_perform_Householder_QR_decomposition\n";

   Tensor<type, 2> a;
   Tensor<type, 1> b;

   Tensor<type, 2> inverse;

   // Test

//   a.set(1, 1, 1.0);

//   b.set(1, 0.0);

//   levenberg_marquardt_algorithm.perform_Householder_QR_decomposition(a, b);

   assert_true(is_equal(a, 1.0), LOG);
   assert_true(is_zero(b), LOG);

   // Test

//   a.set(2, 2);
//   a.initialize_identity();

//   b.set(2, 0.0);

//   levenberg_marquardt_algorithm.perform_Householder_QR_decomposition(a, b);

//   inverse.set(2, 2);
//   inverse.initialize_identity();

//   assert_true(a == inverse, LOG);
//   assert_true(b == 0.0, LOG);

   // Test

//   a.set(100, 100);
   a.setRandom();
//   b.set(100);
   b.setRandom();

//   levenberg_marquardt_algorithm.perform_Householder_QR_decomposition(a, b);

   assert_true(a.dimension(0) == 100, LOG);
   assert_true(a.dimension(1) == 100, LOG);
   assert_true(b.size() == 100, LOG);
}


void LevenbergMarquardtAlgorithmTest::run_test_case()
{
   cout << "Running Levenberg-Marquardt algorithm test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Training methods

   test_perform_training();

   // Training history methods

   test_resize_training_error_history();

   // Linear algebraic equations methods

   test_perform_Householder_QR_decomposition();

   cout << "End of Levenberg-Marquardt algorithm test case.\n\n";
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
