/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O N J U G A T E   G R A D I E N T   T E S T   C L A S S                                                  */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "conjugate_gradient_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

ConjugateGradientTest::ConjugateGradientTest() : UnitTesting() 
{
}


// DESTRUCTOR

ConjugateGradientTest::~ConjugateGradientTest()
{
}


// METHODS

void ConjugateGradientTest::test_constructor()
{
   message += "test_constructor\n"; 

   SumSquaredError sse;

   // Default constructor

   ConjugateGradient cg1; 
   assert_true(cg1.has_loss_index() == false, LOG);

   // Loss index constructor

   ConjugateGradient cg2(&sse);
   assert_true(cg2.has_loss_index() == true, LOG);
}


void ConjugateGradientTest::test_destructor()
{
   message += "test_destructor\n";
}


void ConjugateGradientTest::test_get_training_direction_method()
{
   message += "test_get_training_direction_method\n";

   ConjugateGradient cg;

   cg.set_training_direction_method(ConjugateGradient::PR);

   ConjugateGradient::TrainingDirectionMethod training_direction_method = cg.get_training_direction_method();

   assert_true(training_direction_method == ConjugateGradient::PR, LOG);
}


void ConjugateGradientTest::test_get_training_direction_method_name()
{
   message += "test_get_training_direction_method_name\n";
}


void ConjugateGradientTest::test_set_training_direction_method()
{
   message += "test_set_training_direction_method\n";

   ConjugateGradient cg;

   cg.set_training_direction_method(ConjugateGradient::FR);
   assert_true(cg.get_training_direction_method() == ConjugateGradient::FR, LOG);

   cg.set_training_direction_method(ConjugateGradient::PR);
   assert_true(cg.get_training_direction_method() == ConjugateGradient::PR, LOG);
}


void ConjugateGradientTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   ConjugateGradient cg;
   cg.set_reserve_all_training_history(true);

   assert_true(cg.get_reserve_parameters_history() == true, LOG);
   assert_true(cg.get_reserve_parameters_norm_history() == true, LOG);

   assert_true(cg.get_reserve_error_history() == true, LOG);
   assert_true(cg.get_reserve_gradient_history() == true, LOG);
   assert_true(cg.get_reserve_gradient_norm_history() == true, LOG);

   assert_true(cg.get_reserve_training_direction_history() == true, LOG);
   assert_true(cg.get_reserve_training_rate_history() == true, LOG);
   assert_true(cg.get_reserve_elapsed_time_history() == true, LOG);
   assert_true(cg.get_reserve_selection_error_history() == true, LOG);
}


void ConjugateGradientTest::test_calculate_PR_parameter()
{
   message += "test_calculate_PR_parameter\n";

   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();

   NeuralNetwork nn(1 ,1);
   SumSquaredError sse(&nn, &ds);
   ConjugateGradient cg(&sse);

   nn.initialize_parameters(2.0);
//   Vector<double> old_gradient = sse.calculate_gradient();

   nn.initialize_parameters(1.0);
//   Vector<double> gradient = sse.calculate_gradient();

//   double PR_parameter = cg.calculate_PR_parameter(old_gradient, gradient);

//   assert_true(PR_parameter >= 0.0, LOG);
//   assert_true(PR_parameter <= 1.0, LOG);
}


void ConjugateGradientTest::test_calculate_FR_parameter()
{
   message += "test_calculate_FR_parameter\n";

   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();
   NeuralNetwork nn(1 ,1);
   SumSquaredError sse(&nn, &ds);
   ConjugateGradient cg(&sse);
/*
   sse.set_loss_method("SUM_SQUARED_ERROR");

   nn.initialize_parameters(2.0);
   Vector<double> old_gradient = sse.calculate_gradient();

   nn.initialize_parameters(1.0);
   Vector<double> gradient = sse.calculate_gradient();

   double FR_parameter = cg.calculate_FR_parameter(old_gradient, gradient);

   assert_true(FR_parameter >= 0.0, LOG);
   assert_true(FR_parameter <= 1.0, LOG);
*/
}


void ConjugateGradientTest::test_calculate_PR_training_direction()
{
   message += "test_calculate_PR_training_direction\n";

   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();
   NeuralNetwork nn(1 ,1);
   SumSquaredError sse(&nn, &ds);
   ConjugateGradient cg(&sse);
/*
   sse.set_loss_method("SUM_SQUARED_ERROR");

   nn.initialize_parameters(2.0);
   Vector<double> old_gradient = sse.calculate_gradient();
   Vector<double> old_training_direction = old_gradient;   

   nn.initialize_parameters(1.0);
   Vector<double> gradient = sse.calculate_gradient();

   Vector<double> PR_training_direction 
   = cg.calculate_PR_training_direction(old_gradient, gradient, old_training_direction);

   size_t parameters_number = nn.get_parameters_number();

   assert_true(PR_training_direction.size() == parameters_number, LOG);
*/
}


void ConjugateGradientTest::test_calculate_FR_training_direction()
{
   message += "test_calculate_FR_training_direction\n";

   DataSet ds(1, 1, 2);
   ds.randomize_data_normal();
   NeuralNetwork nn(1 ,1);
   SumSquaredError sse(&nn, &ds);
   ConjugateGradient cg(&sse);
/*
   sse.set_loss_method("SUM_SQUARED_ERROR");

   nn.initialize_parameters(2.0);
   Vector<double> old_gradient = sse.calculate_gradient();
   Vector<double> old_training_direction = old_gradient;   

   nn.initialize_parameters(1.0);
   Vector<double> gradient = sse.calculate_gradient();
	
   Vector<double> FR_training_direction 
   = cg.calculate_FR_training_direction(old_gradient, gradient, old_training_direction);

   size_t parameters_number = nn.get_parameters_number();

   assert_true(FR_training_direction.size() == parameters_number, LOG);
*/
}


void ConjugateGradientTest::test_calculate_training_direction()
{
   message += "test_calculate_training_direction\n";

}


void ConjugateGradientTest::test_perform_training()
{
   message += "test_perform_training\n";

   DataSet ds(1, 1, 1);
   ds.randomize_data_normal();

   NeuralNetwork nn(1, 1, 1);

   SumSquaredError sse(&nn, &ds);
/*
   sse.set_loss_method(LossIndex::SUM_SQUARED_ERROR);

   double old_loss;
   double loss;

   double loss_goal;

   ConjugateGradient cg(&sse);

   double minimum_parameters_increment_norm;

   // Test

   nn.randomize_parameters_normal();

   old_loss = sse.calculate_loss();

   cg.set_display(false);
   cg.set_maximum_iterations_number(1);

   cg.perform_training();

   loss = sse.calculate_loss();

   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   nn.initialize_parameters(-1.0);

   minimum_parameters_increment_norm = 0.1;

   cg.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   cg.set_loss_goal(0.0);
   cg.set_minimum_loss_decrease(0.0);
   cg.set_gradient_norm_goal(0.0);
   cg.set_maximum_iterations_number(1000);
   cg.set_maximum_time(1000.0);

   cg.perform_training();

   // Performance goal

   nn.initialize_parameters(-1.0);

   loss_goal = 0.1;

   cg.set_minimum_parameters_increment_norm(0.0);
   cg.set_loss_goal(loss_goal);
   cg.set_minimum_loss_decrease(0.0);
   cg.set_gradient_norm_goal(0.0);
   cg.set_maximum_iterations_number(1000);
   cg.set_maximum_time(1000.0);

   cg.perform_training();

   loss = sse.calculate_loss();

   assert_true(loss < loss_goal, LOG);

   // Minimum evaluation improvement

   nn.initialize_parameters(-1.0);

   double minimum_loss_increase = 0.1;

   cg.set_minimum_parameters_increment_norm(0.0);
   cg.set_loss_goal(0.0);
   cg.set_minimum_loss_decrease(minimum_loss_increase);
   cg.set_gradient_norm_goal(0.0);
   cg.set_maximum_iterations_number(1000);
   cg.set_maximum_time(1000.0);

   cg.perform_training();

   // Gradient norm goal 

   nn.initialize_parameters(-1.0);

   double gradient_norm_goal = 0.1;

   cg.set_minimum_parameters_increment_norm(0.0);
   cg.set_loss_goal(0.0);
   cg.set_minimum_loss_decrease(0.0);
   cg.set_gradient_norm_goal(gradient_norm_goal);
   cg.set_maximum_iterations_number(1000);
   cg.set_maximum_time(1000.0);

   cg.perform_training();

   double gradient_norm = sse.calculate_gradient().calculate_norm();

   assert_true(gradient_norm < gradient_norm_goal, LOG);
*/
}


void ConjugateGradientTest::test_to_XML()   
{
   message += "test_to_XML\n";

   ConjugateGradient cg;

   tinyxml2::XMLDocument* cgd = cg.to_XML();
   assert_true(cgd != nullptr, LOG);
}


void ConjugateGradientTest::test_from_XML()
{
   message += "test_from_XML\n";

   ConjugateGradient cg1;
   ConjugateGradient cg2;

   tinyxml2::XMLDocument* document;

   // Test

   cg1.set_display(true);

   document = cg1.to_XML();

   cg2.from_XML(*document);

   delete document;

   assert_true(cg1 == cg2, LOG);

}


void ConjugateGradientTest::run_test_case()
{
   message += "Running conjugate gradient test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_training_direction_method();
   test_get_training_direction_method_name();

   // Set methods

   test_set_training_direction_method();

   // Training methods

   test_calculate_PR_parameter();
   test_calculate_FR_parameter();

   test_calculate_FR_training_direction();
   test_calculate_PR_training_direction();

   test_calculate_training_direction();

   test_perform_training();

   // Training history methods

   test_set_reserve_all_training_history();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   message += "End of conjugate gradient test case.\n";
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
