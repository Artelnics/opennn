/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S T O C H A S T I C   G R A D I E N T   D E S C E N T   T E S T   C L A S S                                */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "stochastic_gradient_descent_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR 

StochasticGradientDescentTest::StochasticGradientDescentTest() : UnitTesting()
{
}


// DESTRUCTOR

StochasticGradientDescentTest::~StochasticGradientDescentTest()
{
}


// METHODS

void StochasticGradientDescentTest::test_constructor()
{
   message += "test_constructor\n"; 

   SumSquaredError sse;

   // Default constructor

   StochasticGradientDescent sgd1;
   assert_true(sgd1.has_loss_index() == false, LOG);

   // Loss index constructor

   StochasticGradientDescent sgd2(&sse);
   assert_true(sgd2.has_loss_index() == true, LOG);
}


void StochasticGradientDescentTest::test_destructor()
{
   message += "test_destructor\n"; 
}


void StochasticGradientDescentTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   StochasticGradientDescent sgd;

   sgd.set_reserve_all_training_history(true);

   assert_true(sgd.get_reserve_elapsed_time_history() == true, LOG);
   assert_true(sgd.get_reserve_parameters_history() == true, LOG);
   assert_true(sgd.get_reserve_parameters_norm_history() == true, LOG);
   assert_true(sgd.get_reserve_error_history() == true, LOG);
   assert_true(sgd.get_reserve_gradient_history() == true, LOG);
   assert_true(sgd.get_reserve_gradient_norm_history() == true, LOG);
//   assert_true(sgd.get_reserve_training_direction_history() == true, LOG);
//   assert_true(sgd.get_reserve_training_rate_history() == true, LOG);
   assert_true(sgd.get_reserve_selection_error_history() == true, LOG);
}


void StochasticGradientDescentTest::test_perform_training()
{
   message += "test_perform_training\n";

   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();

   NeuralNetwork nn(1, 2);
   nn.randomize_parameters_normal();

   SumSquaredError sse(&nn, &ds);

   StochasticGradientDescent sgd(&sse);

   // Test

   //double old_loss = sse.calculate_error({0});

   sgd.set_display(false);
   sgd.set_maximum_epochs_number(1);

   sgd.perform_training();

   //double loss = sse.calculate_error({0});

   //assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   nn.initialize_parameters(-1.0);

   double minimum_parameters_increment_norm = 0.1;

   sgd.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   sgd.set_loss_goal(0.0);
//   sgd.set_learning_rate(0.01);
   sgd.set_gradient_norm_goal(0.0);
   sgd.set_training_batch_size(100);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

   // Loss goal
/*
   nn.initialize_parameters(-1.0);

   double loss_goal = 0.1;

   sgd.set_minimum_parameters_increment_norm(0.0);
   sgd.set_loss_goal(loss_goal);
   sgd.set_minimum_loss_decrease(0.0);
   sgd.set_gradient_norm_goal(0.0);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

   //loss = sse.calculate_error({0});

   // Minimum loss increase

   nn.initialize_parameters(-1.0);

   double minimum_loss_increase = 0.1;

   sgd.set_minimum_parameters_increment_norm(0.0);
   sgd.set_loss_goal(0.0);
   sgd.set_minimum_loss_decrease(minimum_loss_increase);
   sgd.set_gradient_norm_goal(0.0);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();

   // Gradient norm goal 

   nn.initialize_parameters(-1.0);

   double gradient_norm_goal = 0.1;

   sgd.set_minimum_parameters_increment_norm(0.0);
   sgd.set_loss_goal(0.0);
   sgd.set_minimum_loss_decrease(0.0);
   sgd.set_gradient_norm_goal(gradient_norm_goal);
   sgd.set_maximum_epochs_number(1000);
   sgd.set_maximum_time(1000.0);

   sgd.perform_training();



   double gradient_norm = sse.calculate_error_gradient({0}).calculate_L2_norm();
   assert_true(gradient_norm < gradient_norm_goal, LOG);
*/
}


void StochasticGradientDescentTest::test_resize_training_history()
{
   message += "test_resize_training_history\n";

   StochasticGradientDescent sgd;

   sgd.set_reserve_all_training_history(true);

   StochasticGradientDescent::StochasticGradientDescentResults sgdtr(&sgd);

   sgdtr.resize_training_history(1);

   assert_true(sgdtr.parameters_history.size() == 1, LOG);
   assert_true(sgdtr.parameters_norm_history.size() == 1, LOG);

   assert_true(sgdtr.loss_history.size() == 1, LOG);
   assert_true(sgdtr.gradient_history.size() == 1, LOG);
   assert_true(sgdtr.gradient_norm_history.size() == 1, LOG);
   assert_true(sgdtr.selection_error_history.size() == 1, LOG);

//   assert_true(sgdtr.training_direction_history.size() == 1, LOG);
//   assert_true(sgdtr.training_rate_history.size() == 1, LOG);
   assert_true(sgdtr.elapsed_time_history.size() == 1, LOG);
}


void StochasticGradientDescentTest::test_to_XML()
{
   message += "test_to_XML\n";

   StochasticGradientDescent sgd;

   tinyxml2::XMLDocument* document;

   // Test

   document = sgd.to_XML();
   assert_true(document != nullptr, LOG);

   delete document;
}


void StochasticGradientDescentTest::test_from_XML()
{
   message += "test_from_XML\n";

   StochasticGradientDescent sgd1;
   StochasticGradientDescent sgd2;

   tinyxml2::XMLDocument* document;

   // Test

   sgd1.initialize_random();

   document = sgd1.to_XML();

   sgd2.from_XML(*document);

   delete document;

   assert_true(sgd2 == sgd1, LOG);

}


void StochasticGradientDescentTest::run_test_case()
{
   message += "Running gradient descent test case...\n";

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

   message += "End of stochastic gradient descent test case.\n";
}


 // OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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

