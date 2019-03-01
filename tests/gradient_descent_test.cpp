/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R A D I E N T   D E S C E N T   T E S T   C L A S S                                                      */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "gradient_descent_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR 

GradientDescentTest::GradientDescentTest() : UnitTesting()
{
}


// DESTRUCTOR

GradientDescentTest::~GradientDescentTest()
{
}


// METHODS

void GradientDescentTest::test_constructor()
{
   message += "test_constructor\n"; 

   SumSquaredError sse;

   // Default constructor

   GradientDescent gd1; 
   assert_true(gd1.has_loss_index() == false, LOG);

   // Loss index constructor

   GradientDescent gd2(&sse);
   assert_true(gd2.has_loss_index() == true, LOG);
}


void GradientDescentTest::test_destructor()
{
   message += "test_destructor\n"; 
}


void GradientDescentTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   GradientDescent gd;

   gd.set_reserve_all_training_history(true);

   assert_true(gd.get_reserve_elapsed_time_history() == true, LOG);
   assert_true(gd.get_reserve_parameters_history() == true, LOG);
   assert_true(gd.get_reserve_parameters_norm_history() == true, LOG);
   assert_true(gd.get_reserve_error_history() == true, LOG);
   assert_true(gd.get_reserve_gradient_history() == true, LOG);
   assert_true(gd.get_reserve_gradient_norm_history() == true, LOG);
   assert_true(gd.get_reserve_training_direction_history() == true, LOG);
   assert_true(gd.get_reserve_training_rate_history() == true, LOG);
   assert_true(gd.get_reserve_selection_error_history() == true, LOG);
}


void GradientDescentTest::test_perform_training()
{
   message += "test_perform_training\n";
/*
   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();

   NeuralNetwork nn(1, 2);
   nn.randomize_parameters_normal();

   SumSquaredError sse(&nn, &ds);

   GradientDescent gd(&sse);

   // Test

   //double old_loss = sse.calculate_error({0});

   gd.set_display(false);
   gd.set_maximum_epochs_number(1);

   gd.perform_training();

   //double loss = sse.calculate_error({0});

   //assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   nn.initialize_parameters(-1.0);

   double minimum_parameters_increment_norm = 0.1;

   gd.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   gd.set_loss_goal(0.0);
   gd.set_minimum_loss_decrease(0.0);
   gd.set_gradient_norm_goal(0.0);
   gd.set_maximum_epochs_number(1000);
   gd.set_maximum_time(1000.0);

   gd.perform_training();

   // Performance goal

   nn.initialize_parameters(-1.0);

   double loss_goal = 0.1;

   gd.set_minimum_parameters_increment_norm(0.0);
   gd.set_loss_goal(loss_goal);
   gd.set_minimum_loss_decrease(0.0);
   gd.set_gradient_norm_goal(0.0);
   gd.set_maximum_epochs_number(1000);
   gd.set_maximum_time(1000.0);

   gd.perform_training();

   //loss = sse.calculate_error({0});

   // Minimum loss increase

   nn.initialize_parameters(-1.0);

   double minimum_loss_increase = 0.1;

   gd.set_minimum_parameters_increment_norm(0.0);
   gd.set_loss_goal(0.0);
   gd.set_minimum_loss_decrease(minimum_loss_increase);
   gd.set_gradient_norm_goal(0.0);
   gd.set_maximum_epochs_number(1000);
   gd.set_maximum_time(1000.0);

   gd.perform_training();

   // Gradient norm goal 

   nn.initialize_parameters(-1.0);

   double gradient_norm_goal = 0.1;

   gd.set_minimum_parameters_increment_norm(0.0);
   gd.set_loss_goal(0.0);
   gd.set_minimum_loss_decrease(0.0);
   gd.set_gradient_norm_goal(gradient_norm_goal);
   gd.set_maximum_epochs_number(1000);
   gd.set_maximum_time(1000.0);

   gd.perform_training();

   double gradient_norm = sse.calculate_error_gradient({0}).calculate_L2_norm();
   assert_true(gradient_norm < gradient_norm_goal, LOG);
*/
}


void GradientDescentTest::test_resize_training_history()
{
   message += "test_resize_training_history\n";

   GradientDescent gd;

   gd.set_reserve_all_training_history(true);

   GradientDescent::GradientDescentResults gdtr(&gd);

   gdtr.resize_training_history(1);

   assert_true(gdtr.parameters_history.size() == 1, LOG);
   assert_true(gdtr.parameters_norm_history.size() == 1, LOG);

   assert_true(gdtr.loss_history.size() == 1, LOG);
   assert_true(gdtr.gradient_history.size() == 1, LOG);
   assert_true(gdtr.gradient_norm_history.size() == 1, LOG);
   assert_true(gdtr.selection_error_history.size() == 1, LOG);

   assert_true(gdtr.training_direction_history.size() == 1, LOG);
   assert_true(gdtr.training_rate_history.size() == 1, LOG);
   assert_true(gdtr.elapsed_time_history.size() == 1, LOG);
}


void GradientDescentTest::test_to_XML()
{
   message += "test_to_XML\n";

   GradientDescent gd;

   tinyxml2::XMLDocument* document;

   // Test

   document = gd.to_XML();
   assert_true(document != nullptr, LOG);

   delete document;
}


void GradientDescentTest::test_from_XML()
{
   message += "test_from_XML\n";

   GradientDescent gd1;
   GradientDescent gd2;

   tinyxml2::XMLDocument* document;

   // Test

   gd1.initialize_random();

   document = gd1.to_XML();

   gd2.from_XML(*document);

   delete document;

   assert_true(gd2 == gd1, LOG);

}


void GradientDescentTest::run_test_case()
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

   message += "End of gradient descent test case.\n";
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

