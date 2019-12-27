//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   T E S T   C L A S S    
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "stochastic_gradient_descent_test.h"
 

StochasticGradientDescentTest::StochasticGradientDescentTest() : UnitTesting()
{
}


StochasticGradientDescentTest::~StochasticGradientDescentTest()
{
}


void StochasticGradientDescentTest::test_constructor()
{
   cout << "test_constructor\n"; 

   SumSquaredError sum_squared_error;

   // Default constructor

   StochasticGradientDescent sgd1;
   assert_true(sgd1.has_loss_index() == false, LOG);

   // Loss index constructor

   StochasticGradientDescent sgd2(&sum_squared_error);
   assert_true(sgd2.has_loss_index() == true, LOG);
}


void StochasticGradientDescentTest::test_destructor()
{
   cout << "test_destructor\n"; 
}


void StochasticGradientDescentTest::test_set_reserve_all_training_history()
{
   cout << "test_set_reserve_all_training_history\n";

   StochasticGradientDescent sgd;

   sgd.set_reserve_all_training_history(true);

   assert_true(sgd.get_reserve_training_error_history() == true, LOG);
   assert_true(sgd.get_reserve_selection_error_history() == true, LOG);
}


/// @todo

void StochasticGradientDescentTest::test_perform_training()
{
   cout << "test_perform_training\n";
/*
   DataSet data_set(1, 1, 2);
   data_set.randomize_data_normal();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 2});
   neural_network.randomize_parameters_normal();

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   StochasticGradientDescent sgd(&sum_squared_error);

   // Test

   //double old_loss = sum_squared_error.calculate_error({0});

   sgd.set_display(false);
   sgd.set_maximum_epochs_number(1);

   sgd.perform_training();

   //double loss = sum_squared_error.calculate_error({0});

   //assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   neural_network.initialize_parameters(-1.0);

   double minimum_parameters_increment_norm = 0.1;

   sgd.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   sgd.set_loss_goal(0.0);
//   sgd.set_learning_rate(0.01);
   sgd.set_gradient_norm_goal(0.0);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

   // Loss goal

   neural_network.initialize_parameters(-1.0);

   double loss_goal = 0.1;

   sgd.set_minimum_parameters_increment_norm(0.0);
   sgd.set_loss_goal(loss_goal);
   sgd.set_gradient_norm_goal(0.0);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

   //loss = sum_squared_error.calculate_error({0});

   // Minimum loss increase

   neural_network.initialize_parameters(-1.0);

   sgd.set_minimum_parameters_increment_norm(0.0);
   sgd.set_loss_goal(0.0);
   sgd.set_gradient_norm_goal(0.0);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

   // Gradient norm goal 

   neural_network.initialize_parameters(-1.0);

   double gradient_norm_goal = 0.1;

   sgd.set_minimum_parameters_increment_norm(0.0);
   sgd.set_loss_goal(0.0);
   sgd.set_gradient_norm_goal(gradient_norm_goal);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

//   double gradient_norm = sum_squared_error.calculate_error_gradient({0}).l2_norm();
//   assert_true(gradient_norm < gradient_norm_goal, LOG);
*/
}


void StochasticGradientDescentTest::test_resize_training_history()
{
   cout << "test_resize_training_history\n";

   StochasticGradientDescent sgd;

   sgd.set_reserve_all_training_history(true);

   OptimizationAlgorithm::Results sgdtr;//(&sgd);

   sgdtr.resize_training_history(1);

   assert_true(sgdtr.training_error_history.size() == 1, LOG);
   assert_true(sgdtr.selection_error_history.size() == 1, LOG);
}


/// @todo

void StochasticGradientDescentTest::test_to_XML()
{
   cout << "test_to_XML\n";
/*
   StochasticGradientDescent sgd;

   tinyxml2::XMLDocument* document;

   // Test

   document = sgd.to_XML();
   assert_true(document != nullptr, LOG);

   delete document;
*/
}


void StochasticGradientDescentTest::test_from_XML()
{
   cout << "test_from_XML\n";
}


void StochasticGradientDescentTest::run_test_case()
{
   cout << "Running stochastic gradient descent test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Set methods

   test_set_reserve_all_training_history();

   // Training methods

   test_perform_training();

   // Training history methods

   test_resize_training_history();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of stochastic gradient descent test case.\n";
}


 // OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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

