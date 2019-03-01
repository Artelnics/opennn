/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */ 
/*   Q U A S I - N E W T O N   M E T H O D   T E S T   C L A S S                                                */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "quasi_newton_method_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

QuasiNewtonMethodTest::QuasiNewtonMethodTest() : UnitTesting() 
{
}


// DESTRUCTOR

/// Destructor.

QuasiNewtonMethodTest::~QuasiNewtonMethodTest()
{

}


// METHODS

void QuasiNewtonMethodTest::test_constructor()
{
   message += "test_constructor\n"; 

   SumSquaredError sse;

   // Default constructor

   QuasiNewtonMethod qnm1; 
   assert_true(qnm1.has_loss_index() == false, LOG);

   // Loss index constructor

   QuasiNewtonMethod qnm2(&sse); 
   assert_true(qnm2.has_loss_index() == true, LOG);
}


void QuasiNewtonMethodTest::test_destructor()
{
}


void QuasiNewtonMethodTest::test_get_inverse_Hessian_approximation_method()
{
   message += "test_get_inverse_Hessian_approximation_method\n";

   QuasiNewtonMethod qnm;

   qnm.set_inverse_Hessian_approximation_method(QuasiNewtonMethod::DFP);
   assert_true(qnm.get_inverse_Hessian_approximation_method() == QuasiNewtonMethod::DFP, LOG);

   qnm.set_inverse_Hessian_approximation_method(QuasiNewtonMethod::BFGS);
   assert_true(qnm.get_inverse_Hessian_approximation_method() == QuasiNewtonMethod::BFGS, LOG);
}


void QuasiNewtonMethodTest::test_get_inverse_Hessian_approximation_method_name()
{
   message += "test_get_inverse_Hessian_approximation_method_name\n";
}


void QuasiNewtonMethodTest::test_set_inverse_Hessian_approximation_method()
{
   message += "test_set_training_direction_method\n";

   QuasiNewtonMethod qnm;

   qnm.set_inverse_Hessian_approximation_method(QuasiNewtonMethod::BFGS);
   assert_true(qnm.get_inverse_Hessian_approximation_method() == QuasiNewtonMethod::BFGS, LOG);
}


// @todo

void QuasiNewtonMethodTest::test_calculate_DFP_inverse_Hessian_approximation()
{
   message += "test_calculate_DFP_inverse_Hessian_approximation\n";

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();
   NeuralNetwork nn(1, 1);
   SumSquaredError sse(&nn, &ds);
   QuasiNewtonMethod qnm(&sse);

   // Test 

   nn.initialize_parameters(1.0);
/*
   Vector<double> old_parameters = nn.get_parameters();
   Vector<double> old_gradient = sse.calculate_gradient();
//   Matrix<double> old_inverse_Hessian = sse.calculate_inverse_Hessian();

   nn.initialize_parameters(-0.5);

   Vector<double> parameters = nn.get_parameters();
   Vector<double> gradient = sse.calculate_gradient();
//   Matrix<double> inverse_Hessian = sse.calculate_inverse_Hessian();

//   Matrix<double> DFP_inverse_Hessian
//   = qnm.calculate_DFP_inverse_Hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);

//   assert_true(DFP_inverse_Hessian == inverse_Hessian, LOG);

   // Test 

   nn.initialize_parameters(1.0e-3);

   old_parameters = nn.get_parameters();
   old_gradient = sse.calculate_gradient();
//   old_inverse_Hessian = sse.calculate_inverse_Hessian();

   nn.initialize_parameters(1.0e-6);

   parameters = nn.get_parameters();
   gradient = sse.calculate_gradient();
//   inverse_Hessian = sse.calculate_inverse_Hessian();

//   DFP_inverse_Hessian = qnm.calculate_DFP_inverse_Hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);

//   assert_true(DFP_inverse_Hessian == inverse_Hessian, LOG);

   // Test 

   nn.initialize_parameters(1.0e-6);

   old_parameters = nn.get_parameters();
   old_gradient = sse.calculate_gradient();
//   old_inverse_Hessian = sse.calculate_inverse_Hessian();

   nn.initialize_parameters(1.0e-9);

   parameters = nn.get_parameters();
   gradient = sse.calculate_gradient();
//   inverse_Hessian = sse.calculate_inverse_Hessian();

//   DFP_inverse_Hessian = qnm.calculate_DFP_inverse_Hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);

//   assert_true(DFP_inverse_Hessian == inverse_Hessian, LOG);

   // Test 

   old_parameters.initialize(1.0e-3);
   parameters.initialize(1.0e-6);

   old_gradient.initialize(1.0e-3);
   gradient.initialize(1.0e-6);

//   old_inverse_Hessian(0,0) = 0.75;
//   old_inverse_Hessian(0,1) = -0.25;
//   old_inverse_Hessian(1,0) = -0.25;
//   old_inverse_Hessian(1,1) = 0.75;

//   DFP_inverse_Hessian = qnm.calculate_DFP_inverse_Hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);
*/
}


// @todo

void QuasiNewtonMethodTest::test_calculate_BFGS_inverse_Hessian_approximation()
{
   message += "test_calculate_BFGS_inverse_Hessian_approximation\n";

   DataSet ds;
   NeuralNetwork nn(1, 1);
/*
   SumSquaredError sse(&nn, &ds);

   sse.set_regularization_method(LossIndex::L2);

   QuasiNewtonMethod qnm(&sse);

   nn.initialize_parameters(1.0);

   Vector<double> old_parameters = nn.get_parameters();
   Vector<double> old_gradient = sse.calculate_gradient();
//   Matrix<double> old_inverse_Hessian = sse.calculate_inverse_Hessian();

   nn.initialize_parameters(-0.5);

   Vector<double> parameters = nn.get_parameters();
   Vector<double> gradient = sse.calculate_gradient();
//   Matrix<double> inverse_Hessian = sse.calculate_inverse_Hessian();

//   Matrix<double> BFGS_inverse_Hessian
//   = qnm.calculate_BFGS_inverse_Hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);

//   assert_true(BFGS_inverse_Hessian == inverse_Hessian, LOG);
*/
}

// @todo

void QuasiNewtonMethodTest::test_calculate_inverse_Hessian_approximation()
{
   message += "test_calculate_inverse_Hessian_approximation\n";

   NeuralNetwork nn(1, 1);
   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();
   SumSquaredError sse(&nn, &ds);
   QuasiNewtonMethod qnm(&sse);

   qnm.set_inverse_Hessian_approximation_method(QuasiNewtonMethod::DFP);

   nn.initialize_parameters(1.0);
/*
   Vector<double> old_parameters = nn.get_parameters();
   Vector<double> old_gradient = sse.calculate_gradient();
//   Matrix<double> old_inverse_Hessian = sse.calculate_inverse_Hessian();

   nn.initialize_parameters(-0.5);

   Vector<double> parameters = nn.get_parameters();
   Vector<double> gradient = sse.calculate_gradient();
//   Matrix<double> inverse_Hessian = sse.calculate_inverse_Hessian();

//   Matrix<double> inverse_Hessian_approximation
//   = qnm.calculate_inverse_Hessian_approximation(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);

//   assert_true(inverse_Hessian_approximation == inverse_Hessian, LOG);

   qnm.set_inverse_Hessian_approximation_method(QuasiNewtonMethod::DFP);

   nn.initialize_parameters(1.0);

   old_parameters = nn.get_parameters();
   old_gradient = sse.calculate_gradient();
//   old_inverse_Hessian = sse.calculate_inverse_Hessian();

   nn.initialize_parameters(-0.5);

   parameters = nn.get_parameters();
   gradient = sse.calculate_gradient();
//   inverse_Hessian = sse.calculate_inverse_Hessian();

//   inverse_Hessian_approximation
//   = qnm.calculate_inverse_Hessian_approximation(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);

//   assert_true(inverse_Hessian_approximation == inverse_Hessian, LOG);

   // Test 

   old_parameters.initialize(1.0e-3);
   parameters.initialize(1.0e-6);

   old_gradient.initialize(1.0e-3);
   gradient.initialize(1.0e-6);

//   old_inverse_Hessian(0,0) = 0.75;
//   old_inverse_Hessian(0,1) = -0.25;
//   old_inverse_Hessian(1,0) = -0.25;
//   old_inverse_Hessian(1,1) = 0.75;

//   inverse_Hessian_approximation
//   = qnm.calculate_inverse_Hessian_approximation(old_parameters, parameters, old_gradient, gradient, old_inverse_Hessian);
*/
}


void QuasiNewtonMethodTest::test_calculate_training_direction()
{
   message += "test_calculate_training_direction\n";
}


void QuasiNewtonMethodTest::test_perform_training()
{
   message += "test_perform_training\n";

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();
   NeuralNetwork nn(1, 1, 1);
   SumSquaredError sse(&nn, &ds);
   QuasiNewtonMethod qnm(&sse);
   qnm.set_inverse_Hessian_approximation_method(QuasiNewtonMethod::DFP);

   qnm.set_reserve_all_training_history(true);

    // Test

   nn.initialize_parameters(3.1415927);
/*
   double old_loss = sse.calculate_loss();

   qnm.set_maximum_iterations_number(2),
   qnm.set_display(false);

   qnm.perform_training();

   double loss = sse.calculate_loss();

   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   nn.initialize_parameters(3.1415927);

   double minimum_parameters_increment_norm = 0.1;

   qnm.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   qnm.set_loss_goal(0.0);
   qnm.set_minimum_loss_decrease(0.0);
   qnm.set_gradient_norm_goal(0.0);
   qnm.set_maximum_iterations_number(10);
   qnm.set_maximum_time(1000.0);

   qnm.perform_training();

   // Performance goal

   nn.initialize_parameters(3.1415927);

   double loss_goal = 100.0;

   qnm.set_minimum_parameters_increment_norm(0.0);
   qnm.set_loss_goal(loss_goal);
   qnm.set_minimum_loss_decrease(0.0);
   qnm.set_gradient_norm_goal(0.0);
   qnm.set_maximum_iterations_number(10);
   qnm.set_maximum_time(1000.0);

   qnm.perform_training();

   loss = sse.calculate_loss();

   assert_true(loss < loss_goal, LOG);

   // Minimum evaluation improvement

   nn.initialize_parameters(3.1415927);

   double minimum_loss_increase = 100.0;

   qnm.set_minimum_parameters_increment_norm(0.0);
   qnm.set_loss_goal(0.0);
   qnm.set_minimum_loss_decrease(minimum_loss_increase);
   qnm.set_gradient_norm_goal(0.0);
   qnm.set_maximum_iterations_number(10);
   qnm.set_maximum_time(1000.0);

   qnm.perform_training();

   // Gradient norm goal 

   nn.initialize_parameters(3.1415927);

   double gradient_norm_goal = 100.0;

   qnm.set_minimum_parameters_increment_norm(0.0);
   qnm.set_loss_goal(0.0);
   qnm.set_minimum_loss_decrease(0.0);
   qnm.set_gradient_norm_goal(gradient_norm_goal);
   qnm.set_maximum_iterations_number(10);
   qnm.set_maximum_time(1000.0);

   qnm.perform_training();

   double gradient_norm = sse.calculate_gradient().calculate_norm();
   assert_true(gradient_norm < gradient_norm_goal, LOG);
*/
}


void QuasiNewtonMethodTest::test_to_XML()   
{
   message += "test_to_XML\n";

   QuasiNewtonMethod qnm;

   tinyxml2::XMLDocument* document = qnm.to_XML();
   assert_true(document != nullptr, LOG);

   delete document;
}


void QuasiNewtonMethodTest::test_resize_training_history()
{
    message += "test_resize_training_history\n";

    DataSet ds;

    NeuralNetwork nn;

    SumSquaredError sse(&nn, &ds);

    QuasiNewtonMethod qnm(&sse);

    QuasiNewtonMethod::QuasiNewtonMethodResults results(&qnm);

    qnm.set_reserve_all_training_history(true);

    // Test

    results.resize_training_history(2);

    assert_true(results.elapsed_time_history.size() == 2, LOG);

}


void QuasiNewtonMethodTest::test_load()
{
   message += "test_load\n";

   QuasiNewtonMethod qnm;

   tinyxml2::XMLDocument* document = qnm.to_XML();
   qnm.from_XML(*document);

   delete document;
}


void QuasiNewtonMethodTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   QuasiNewtonMethod qnm;
   qnm.set_reserve_all_training_history(true);
}


void QuasiNewtonMethodTest::run_test_case()
{
   message += "Running quasi-Newton method test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor(); 

   // Get methods

   test_get_inverse_Hessian_approximation_method();
   test_get_inverse_Hessian_approximation_method_name();

   // Set methods

   test_set_inverse_Hessian_approximation_method();

   // Training methods

   test_calculate_DFP_inverse_Hessian_approximation();
   test_calculate_BFGS_inverse_Hessian_approximation();

   test_calculate_inverse_Hessian_approximation();
   test_calculate_training_direction();

   test_perform_training();

   // Training history methods

   test_resize_training_history();
   test_set_reserve_all_training_history();

   // Serialization methods

   test_to_XML();   
   test_load();

   message += "End of quasi-Newton method test case.\n";
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
