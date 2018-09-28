/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N E W T O N   M E T H O D   T E S T   C L A S S                                                            */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "newton_method_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

NewtonMethodTest::NewtonMethodTest() : UnitTesting() 
{   
}


// DESTRUCTOR

NewtonMethodTest::~NewtonMethodTest()
{
}


void NewtonMethodTest::test_constructor()
{
   message += "test_constructor\n"; 

   LossIndex mof;

   // Default constructor

   NewtonMethod nm1; 
   assert_true(nm1.has_loss_index() == false, LOG);

   // Objective functional constructor

   NewtonMethod nm2(&mof); 
   assert_true(nm2.has_loss_index() == true, LOG);
}


void NewtonMethodTest::test_destructor()
{
   message += "test_destructor\n";
}


void NewtonMethodTest::test_calculate_gradient_descent_training_direction()
{
   message += "test_calculate_gradient_descent_training_direction\n";

   DataSet ds(1, 1, 2);

   NeuralNetwork nn(1, 1);
   
   LossIndex pf(&nn, &ds);
   
   NewtonMethod nm(&pf);

   Vector<double> gradient(2, 1.0);

   Vector<double> gradient_descent_training_direction = nm.calculate_gradient_descent_training_direction(gradient);

   assert_true(gradient_descent_training_direction.size() == 2, LOG);
   assert_true((gradient_descent_training_direction.calculate_norm() - 1.0) < 1.0e-3, LOG);
}


void NewtonMethodTest::test_calculate_training_direction()
{
   message += "test_calculate_training_direction\n";

}


// @todo

void NewtonMethodTest::test_perform_training()
{
   message += "test_perform_training\n";

   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();

   NeuralNetwork nn(1, 1);
   LossIndex pf(&nn, &ds);
   NewtonMethod nm(&pf);

   nn.initialize_parameters(0.1);

//   double old_loss = pf.calculate_loss();

   nm.set_display(false);
   nm.set_maximum_iterations_number(1);
//   nm.perform_training();

//   double loss = pf.calculate_loss();
   
//   assert_true(evaluation < old_loss, LOG);

   // Minimum parameters increment norm

   nn.initialize_parameters(1.0);

   double minimum_parameters_increment_norm = 0.1;

   nm.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   nm.set_loss_goal(0.0);
   nm.set_minimum_loss_increase(0.0);
   nm.set_gradient_norm_goal(0.0);
   nm.set_maximum_iterations_number(1000);
   nm.set_maximum_time(1000.0);

//   nm.perform_training();

   // Performance goal

   nn.initialize_parameters(1.0);

   double loss_goal = 0.1;

   nm.set_minimum_parameters_increment_norm(0.0);
   nm.set_loss_goal(loss_goal);
   nm.set_minimum_loss_increase(0.0);
   nm.set_gradient_norm_goal(0.0);
   nm.set_maximum_iterations_number(1000);
   nm.set_maximum_time(1000.0);

//   nm.perform_training();

//   loss = pf.calculate_loss();

//   assert_true(loss < loss_goal, LOG);

   // Minimum evaluation improvement

   nn.initialize_parameters(1.0);

   double minimum_loss_increase = 0.1;

   nm.set_minimum_parameters_increment_norm(0.0);
   nm.set_loss_goal(0.0);
   nm.set_minimum_loss_increase(minimum_loss_increase);
   nm.set_gradient_norm_goal(0.0);
   nm.set_maximum_iterations_number(1000);
   nm.set_maximum_time(1000.0);

//   nm.perform_training();

   // Gradient norm goal 

   nn.initialize_parameters(1.0);

   double gradient_norm_goal = 0.1;

   nm.set_minimum_parameters_increment_norm(0.0);
   nm.set_loss_goal(0.0);
   nm.set_minimum_loss_increase(0.0);
   nm.set_gradient_norm_goal(gradient_norm_goal);
   nm.set_maximum_iterations_number(1000);
   nm.set_maximum_time(1000.0);

//   nm.perform_training();

//   double gradient_norm = pf.calculate_gradient_norm();

//   assert_true(gradient_norm < gradient_norm_goal, LOG);

}


void NewtonMethodTest::test_to_XML()
{
   message += "test_to_XML\n";

   NewtonMethod nm;

   tinyxml2::XMLDocument* document;

   // Test

   document = nm.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void NewtonMethodTest::test_from_XML()
{
   message += "test_from_XML\n";

   NewtonMethod nm;

   tinyxml2::XMLDocument* document;

   // Test

   document = nm.to_XML();

   nm.from_XML(*document);

   delete document;
}


void NewtonMethodTest::test_resize_training_history()
{
   message += "test_resize_training_history\n";

   NewtonMethod nm;

   nm.set_reserve_all_training_history(true);

   NewtonMethod::NewtonMethodResults nmtr(&nm);

   nmtr.resize_training_history(1);

   assert_true(nmtr.parameters_history.size() == 1, LOG);
   assert_true(nmtr.parameters_norm_history.size() == 1, LOG);

   assert_true(nmtr.loss_history.size() == 1, LOG);
   assert_true(nmtr.gradient_history.size() == 1, LOG);
   assert_true(nmtr.gradient_norm_history.size() == 1, LOG);
   assert_true(nmtr.inverse_Hessian_history.size() == 1, LOG);
   assert_true(nmtr.selection_loss_history.size() == 1, LOG);  

   assert_true(nmtr.training_direction_history.size() == 1, LOG);
   assert_true(nmtr.training_rate_history.size() == 1, LOG);
   assert_true(nmtr.elapsed_time_history.size() == 1, LOG);
}


void NewtonMethodTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   NewtonMethod nm;
   nm.set_reserve_all_training_history(true);

   assert_true(nm.get_reserve_parameters_history() == true, LOG);
   assert_true(nm.get_reserve_parameters_norm_history() == true, LOG);

   assert_true(nm.get_reserve_loss_history() == true, LOG);
   assert_true(nm.get_reserve_gradient_history() == true, LOG);
   assert_true(nm.get_reserve_gradient_norm_history() == true, LOG);
   assert_true(nm.get_reserve_inverse_Hessian_history() == true, LOG);

   assert_true(nm.get_reserve_training_direction_history() == true, LOG);
   assert_true(nm.get_reserve_training_rate_history() == true, LOG);
   assert_true(nm.get_reserve_elapsed_time_history() == true, LOG);
   assert_true(nm.get_reserve_selection_loss_history() == true, LOG);
}


void NewtonMethodTest::run_test_case()
{
   message += "Running Newton method test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Training methods

   test_calculate_gradient_descent_training_direction();
   test_calculate_training_direction();

   test_perform_training();

   // Training history methods

   test_resize_training_history();
   test_set_reserve_all_training_history();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of Newton method test case.\n";
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
