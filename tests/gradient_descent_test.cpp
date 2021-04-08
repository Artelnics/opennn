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
}


GradientDescentTest::~GradientDescentTest()
{
}


void GradientDescentTest::test_constructor()
{
   cout << "test_constructor\n"; 

   SumSquaredError sum_squared_error;

   // Default constructor

   GradientDescent gd1;
   assert_true(!gd1.has_loss_index(), LOG);

   // Loss index constructor

   GradientDescent gd2(&sum_squared_error);
   assert_true(gd2.has_loss_index(), LOG);
}


void GradientDescentTest::test_destructor()
{
   cout << "test_destructor\n"; 
}


void GradientDescentTest::test_set_reserve_all_training_history()
{
   cout << "test_set_reserve_all_training_history\n";

   GradientDescent gradient_descent;

   gradient_descent.set_reserve_all_training_history(true);

   assert_true(gradient_descent.get_reserve_training_error_history(), LOG);
   assert_true(gradient_descent.get_reserve_selection_error_history(), LOG);
}


void GradientDescentTest::test_perform_training()
{
   cout << "test_perform_training\n";

   DataSet data_set(1, 1, 2);
   data_set.set_data_random();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 2});
   neural_network.set_parameters_random();

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   GradientDescent gradient_descent(&sum_squared_error);

   // Test

//   type old_loss = sum_squared_error.calculate_error({0});

   gradient_descent.set_display(false);
   gradient_descent.set_maximum_epochs_number(1);

   gradient_descent.perform_training();

//   type loss = sum_squared_error.calculate_error({0});

   //assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   neural_network.set_parameters_constant(-1.0);

   type minimum_parameters_increment_norm = 0.1;

   gradient_descent.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   gradient_descent.set_loss_goal(0.0);
   gradient_descent.set_minimum_loss_decrease(0.0);
   gradient_descent.set_gradient_norm_goal(0.0);
   gradient_descent.set_maximum_epochs_number(1000);
   gradient_descent.set_maximum_time(1000.0);

   gradient_descent.perform_training();

   // Performance goal

   neural_network.set_parameters_constant(-1.0);

   type training_loss_goal = 0.1;

   gradient_descent.set_minimum_parameters_increment_norm(0.0);
   gradient_descent.set_loss_goal(training_loss_goal);
   gradient_descent.set_minimum_loss_decrease(0.0);
   gradient_descent.set_gradient_norm_goal(0.0);
   gradient_descent.set_maximum_epochs_number(1000);
   gradient_descent.set_maximum_time(1000.0);

   gradient_descent.perform_training();

   //loss = sum_squared_error.calculate_error({0});

   // Minimum loss increase

   neural_network.set_parameters_constant(-1.0);

   type minimum_loss_decrease = 0.1;

   gradient_descent.set_minimum_parameters_increment_norm(0.0);
   gradient_descent.set_loss_goal(0.0);
   gradient_descent.set_minimum_loss_decrease(minimum_loss_decrease);
   gradient_descent.set_gradient_norm_goal(0.0);
   gradient_descent.set_maximum_epochs_number(1000);
   gradient_descent.set_maximum_time(1000.0);

   gradient_descent.perform_training();

   // Gradient norm goal

   neural_network.set_parameters_constant(-1.0);

   type gradient_norm_goal = 0.1;

   gradient_descent.set_minimum_parameters_increment_norm(0.0);
   gradient_descent.set_loss_goal(0.0);
   gradient_descent.set_minimum_loss_decrease(0.0);
   gradient_descent.set_gradient_norm_goal(gradient_norm_goal);
   gradient_descent.set_maximum_epochs_number(1000);
   gradient_descent.set_maximum_time(1000.0);

   gradient_descent.perform_training();

//   type gradient_norm = sum_squared_error.calculate_error_gradient({0}).l2_norm();
//   assert_true(gradient_norm < gradient_norm_goal, LOG);

}


void GradientDescentTest::test_resize_training_history()
{
   cout << "test_resize_training_history\n";

   GradientDescent gradient_descent;

   gradient_descent.set_reserve_all_training_history(true);

   TrainingResults training_results;

   training_results.resize_training_history(1);

   assert_true(training_results.training_error_history.size() == 1, LOG);
   assert_true(training_results.selection_error_history.size() == 1, LOG);
}


/// @todo

void GradientDescentTest::test_to_XML()
{
   cout << "test_to_XML\n";

//   GradientDescent gradient_descent;

//   tinyxml2::XMLDocument* document;

   // Test

//   document = gradient_descent.to_XML();
//   assert_true(document != nullptr, LOG);

//   delete document;

}


void GradientDescentTest::test_from_XML()
{
   cout << "test_from_XML\n";
}


void GradientDescentTest::run_test_case()
{
   cout << "Running gradient descent test case...\n";

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

