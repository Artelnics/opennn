/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   L E V E N B E R G   M A R Q U A R D T   A L G O R I T H M   T E S T   C L A S S                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "levenberg_marquardt_algorithm_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

LevenbergMarquardtAlgorithmTest::LevenbergMarquardtAlgorithmTest() : UnitTesting() 
{
}


// DESTRUCTOR

LevenbergMarquardtAlgorithmTest::~LevenbergMarquardtAlgorithmTest()
{
}


// METHODS

void LevenbergMarquardtAlgorithmTest::test_constructor()
{
   message += "test_constructor\n"; 

   LossIndex pf;

   // Default constructor

   LevenbergMarquardtAlgorithm lma1; 
   assert_true(lma1.has_loss_index() == false, LOG);

   // Loss index constructor

   LevenbergMarquardtAlgorithm lma2(&pf); 
   assert_true(lma2.has_loss_index() == true, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_destructor()
{
   message += "test_destructor\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_damping_parameter()
{
   message += "test_get_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_damping_parameter_factor()
{
   message += "test_get_damping_parameter_factor\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_minimum_damping_parameter()
{
   message += "test_get_minimum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_maximum_damping_parameter()
{
   message += "test_get_maximum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_damping_parameter()
{
   message += "test_set_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_damping_parameter_factor()
{
   message += "test_set_damping_parameter_factor\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_minimum_damping_parameter()
{
   message += "test_set_minimum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_maximum_damping_parameter()
{
   message += "test_set_maximum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_calculate_loss()
{
    message += "test_calculate_loss\n";

    DataSet ds;

    NeuralNetwork nn;

    LossIndex pf(&nn, &ds);

    Vector<double> terms;

    double loss;

    LevenbergMarquardtAlgorithm lma(&pf);

    // Test

    ds.set(2, 2, 2);
    ds.randomize_data_normal();

    nn.set(2, 2);
    nn.randomize_parameters_normal();

    terms = pf.calculate_terms();

    loss = lma.calculate_loss(terms);

    assert_true(fabs(loss-pf.calculate_loss()) < 1.0e-3, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";

   DataSet ds;

   NeuralNetwork nn;

   LossIndex pf(&nn, &ds);

   Vector<double> terms;
   Matrix<double> terms_Jacobian;

   Vector<double> gradient;

   LevenbergMarquardtAlgorithm lma(&pf);

   pf.set_error_type("SUM_SQUARED_ERROR");

   // Test

   ds.set(1, 1, 2);
   ds.randomize_data_normal();

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   terms = pf.calculate_terms();

   terms_Jacobian = pf.calculate_terms_Jacobian();

   gradient = lma.calculate_gradient(terms, terms_Jacobian);

   assert_true((gradient-pf.calculate_gradient()).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(1, 1);

   nn.randomize_parameters_normal();

   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   pf.set_user_error_pointer(mptp);

   terms= pf.calculate_terms();

   terms_Jacobian = pf.calculate_terms_Jacobian();

   gradient = lma.calculate_gradient(terms, terms_Jacobian);

   assert_true(gradient == pf.calculate_gradient(), LOG);

}


void LevenbergMarquardtAlgorithmTest::test_calculate_Hessian_approximation()
{
   message += "test_calculate_Hessian_approximation\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;

   size_t parameters_number;

   Vector<double> parameters;

   DataSet ds;

   LossIndex pf(&nn, &ds);

   pf.set_error_type(LossIndex::SUM_SQUARED_ERROR);

   Matrix<double> terms_Jacobian;
   Matrix<double> Hessian;
   Matrix<double> numerical_Hessian;
   Matrix<double> Hessian_approximation;

   LevenbergMarquardtAlgorithm lma(&pf);
   
   // Test

   nn.set(1, 2);
   nn.initialize_parameters(0.0);

   parameters_number = nn.count_parameters_number();

   ds.set(1,2,2);
   ds.initialize_data(0.0);

   terms_Jacobian = pf.calculate_terms_Jacobian();

   Hessian_approximation = lma.calculate_Hessian_approximation(terms_Jacobian);

   assert_true(Hessian_approximation.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian_approximation.get_columns_number() == parameters_number, LOG);
   assert_true(Hessian_approximation.is_symmetric(), LOG);

   // Test

   pf.set_error_type(LossIndex::MEAN_SQUARED_ERROR);

   nn.set(1,1,2);
   nn.randomize_parameters_normal();

   parameters_number = nn.count_parameters_number();

   ds.set(1,2,3);
   ds.randomize_data_normal();

   terms_Jacobian = pf.calculate_terms_Jacobian();

   Hessian_approximation = lma.calculate_Hessian_approximation(terms_Jacobian);

   assert_true(Hessian_approximation.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian_approximation.get_columns_number() == parameters_number, LOG);
   assert_true(Hessian_approximation.is_symmetric(), LOG);

   // Test

   nn.set(2);

   nn.randomize_parameters_normal();

   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   pf.set_user_error_pointer(mptp);

   terms_Jacobian = pf.calculate_terms_Jacobian();

   Hessian = pf.calculate_Hessian();

   lma.set_damping_parameter(0.0);

   assert_true((lma.calculate_Hessian_approximation(terms_Jacobian) - Hessian).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   pf.set_error_type(LossIndex::SUM_SQUARED_ERROR);

   ds.set(1, 1, 1);

   ds.randomize_data_normal();

   nn.set(1, 1);

   parameters = nn.arrange_parameters();

   nn.randomize_parameters_normal();

   numerical_Hessian = nd.calculate_Hessian(pf, &LossIndex::calculate_loss, parameters);

   terms_Jacobian = pf.calculate_terms_Jacobian();

   Hessian_approximation = lma.calculate_Hessian_approximation(terms_Jacobian);

   assert_true((numerical_Hessian - Hessian_approximation).calculate_absolute_value() >= 0.0, LOG);

}


void LevenbergMarquardtAlgorithmTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   LevenbergMarquardtAlgorithm lma;
   lma.set_reserve_all_training_history(true);

   assert_true(lma.get_reserve_parameters_history() == true, LOG);
   assert_true(lma.get_reserve_parameters_norm_history() == true, LOG);

   assert_true(lma.get_reserve_loss_history() == true, LOG);
   assert_true(lma.get_reserve_selection_loss_history() == true, LOG);
   assert_true(lma.get_reserve_gradient_history() == true, LOG);
   assert_true(lma.get_reserve_gradient_norm_history() == true, LOG);
   assert_true(lma.get_reserve_Hessian_approximation_history() == true, LOG);

   assert_true(lma.get_reserve_damping_parameter_history() == true, LOG);
   assert_true(lma.get_reserve_elapsed_time_history() == true, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_perform_training()
{
   message += "test_perform_training\n";

   NeuralNetwork nn;
   
   DataSet ds;
   
   LossIndex pf(&nn, &ds);
   Vector<double> gradient;

   pf.set_error_type(LossIndex::MEAN_SQUARED_ERROR);

   LevenbergMarquardtAlgorithm lma(&pf);
   lma.set_display(false);

   double old_loss;
   double loss;
   double minimum_parameters_increment_norm;
   double loss_goal;
   double minimum_loss_increase;
   double gradient_norm_goal;
   double gradient_norm;

   // Test

   nn.set(1, 1, 1);
   nn.randomize_parameters_normal(0.0, 1.0e-3);

   ds.set(1, 1, 2);
   ds.randomize_data_normal(0.0, 1.0e-3);

   old_loss = pf.calculate_loss();

   lma.perform_training();

   loss = pf.calculate_loss();

   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   nn.randomize_parameters_normal(0.0, 1.0e-3);

   minimum_parameters_increment_norm = 100.0;

   lma.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   lma.set_loss_goal(0.0);
   lma.set_minimum_loss_increase(0.0);
   lma.set_gradient_norm_goal(0.0);
   lma.set_maximum_iterations_number(10);
   lma.set_maximum_time(10.0);

   lma.perform_training();

   // Performance goal

   nn.randomize_parameters_normal(0.0, 1.0e-3);

   loss_goal = 100.0;

   lma.set_minimum_parameters_increment_norm(0.0);
   lma.set_loss_goal(loss_goal);
   lma.set_minimum_loss_increase(0.0);
   lma.set_gradient_norm_goal(0.0);
   lma.set_maximum_iterations_number(10);
   lma.set_maximum_time(10.0);

   lma.perform_training();

   loss = pf.calculate_loss();

   assert_true(loss < loss_goal, LOG);

   // Minimum loss increas

   nn.randomize_parameters_normal(0.0, 1.0e-3);

   minimum_loss_increase = 100.0;

   lma.set_minimum_parameters_increment_norm(0.0);
   lma.set_loss_goal(0.0);
   lma.set_minimum_loss_increase(minimum_loss_increase);
   lma.set_gradient_norm_goal(0.0);
   lma.set_maximum_iterations_number(10);
   lma.set_maximum_time(10.0);

   lma.perform_training();

   // Gradient norm goal 

   nn.randomize_parameters_normal(0.0, 1.0e-3);

   gradient_norm_goal = 1.0e6;

   lma.set_minimum_parameters_increment_norm(0.0);
   lma.set_loss_goal(0.0);
   lma.set_minimum_loss_increase(0.0);
   lma.set_gradient_norm_goal(gradient_norm_goal);
   lma.set_maximum_iterations_number(10);
   lma.set_maximum_time(10.0);

   lma.perform_training();

   gradient = pf.calculate_gradient();
   gradient_norm = gradient.calculate_norm();

   assert_true(gradient_norm < gradient_norm_goal, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_resize_training_history()
{
   message += "test_resize_training_history\n";

   LevenbergMarquardtAlgorithm lma;

   lma.set_reserve_all_training_history(true);

   LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults lmatr(&lma);

   lmatr.resize_training_history(1);

   assert_true(lmatr.parameters_history.size() == 1, LOG);
   assert_true(lmatr.parameters_norm_history.size() == 1, LOG);

   assert_true(lmatr.loss_history.size() == 1, LOG);
   assert_true(lmatr.selection_loss_history.size() == 1, LOG);
   assert_true(lmatr.gradient_history.size() == 1, LOG);
   assert_true(lmatr.gradient_norm_history.size() == 1, LOG);
   assert_true(lmatr.Hessian_approximation_history.size() == 1, LOG);

   assert_true(lmatr.damping_parameter_history.size() == 1, LOG);
   assert_true(lmatr.elapsed_time_history.size() == 1, LOG);

}


void LevenbergMarquardtAlgorithmTest::test_to_XML()   
{
   message += "test_to_XML\n";

   LevenbergMarquardtAlgorithm lma;

   tinyxml2::XMLDocument* lmad = lma.to_XML();
   
   assert_true(lmad != NULL, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_from_XML()
{
   message += "test_from_XML\n";

   LevenbergMarquardtAlgorithm lma;
}


void LevenbergMarquardtAlgorithmTest::test_perform_Householder_QR_decomposition()
{
   message += "test_perform_Householder_QR_decomposition\n";

   LevenbergMarquardtAlgorithm lma;

   Matrix<double> a;
   Vector<double> b;

   Matrix<double> inverse;

   // Test

   a.set(1, 1, 1.0);

   b.set(1, 0.0);

   lma.perform_Householder_QR_decomposition(a, b);

   assert_true(a == 1.0, LOG);
   assert_true(b == 0.0, LOG);

   // Test

   a.set(2, 2);
   a.initialize_identity();

   b.set(2, 0.0);

   lma.perform_Householder_QR_decomposition(a, b);

   inverse.set(2, 2);
   inverse.initialize_identity();

   assert_true(a == inverse, LOG);
   assert_true(b == 0.0, LOG);

   // Test

   a.set(100, 100);
   a.randomize_normal();
   b.set(100);
   b.randomize_normal();

   lma.perform_Householder_QR_decomposition(a, b);

   assert_true(a.get_rows_number() == 100, LOG);
   assert_true(a.get_columns_number() == 100, LOG);
   assert_true(b.size() == 100, LOG);
}


void LevenbergMarquardtAlgorithmTest::run_test_case()
{
   message += "Running Levenberg-Marquardt algorithm test case...\n";

   // Constructor and destructor methods
/*
   test_constructor();
   test_destructor();

   // Get methods

   test_get_damping_parameter();

   test_get_damping_parameter_factor();

   test_get_minimum_damping_parameter();
   test_get_maximum_damping_parameter();

   // Set methods

   test_set_damping_parameter();

   test_set_damping_parameter_factor();

   test_set_minimum_damping_parameter();
   test_set_maximum_damping_parameter();

   // Training methods
*/
//   test_calculate_loss();
//   test_calculate_gradient();
//   test_calculate_Hessian_approximation();
/*
   test_perform_training();

   // Training history methods

   test_set_reserve_all_training_history();
   test_resize_training_history();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   // Linear algebraic equations methods

   test_perform_Householder_QR_decomposition();
*/
   message += "End of Levenberg-Marquardt algorithm test case.\n";
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
