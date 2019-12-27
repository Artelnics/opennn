//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "quasi_newton_method_test.h"


QuasiNewtonMethodTest::QuasiNewtonMethodTest() : UnitTesting() 
{
}


QuasiNewtonMethodTest::~QuasiNewtonMethodTest()
{

}


void QuasiNewtonMethodTest::test_constructor()
{
   cout << "test_constructor\n"; 

   SumSquaredError sum_squared_error;

   // Default constructor

   QuasiNewtonMethod qnm1; 
   assert_true(qnm1.has_loss_index() == false, LOG);

   // Loss index constructor

   QuasiNewtonMethod qnm2(&sum_squared_error);
   assert_true(qnm2.has_loss_index() == true, LOG);
}


void QuasiNewtonMethodTest::test_destructor()
{
}


void QuasiNewtonMethodTest::test_get_inverse_hessian_approximation_method()
{
   cout << "test_get_inverse_hessian_approximation_method\n";

   QuasiNewtonMethod quasi_newton_method;

   quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::DFP);
   assert_true(quasi_newton_method.get_inverse_hessian_approximation_method() == QuasiNewtonMethod::DFP, LOG);

   quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::BFGS);
   assert_true(quasi_newton_method.get_inverse_hessian_approximation_method() == QuasiNewtonMethod::BFGS, LOG);
}


void QuasiNewtonMethodTest::test_get_inverse_hessian_approximation_method_name()
{
   cout << "test_get_inverse_hessian_approximation_method_name\n";
}


void QuasiNewtonMethodTest::test_set_inverse_hessian_approximation_method()
{
   cout << "test_set_training_direction_method\n";

   QuasiNewtonMethod quasi_newton_method;

   quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::BFGS);
   assert_true(quasi_newton_method.get_inverse_hessian_approximation_method() == QuasiNewtonMethod::BFGS, LOG);
}


/// @todo

void QuasiNewtonMethodTest::test_calculate_DFP_inverse_hessian_approximation()
{
   cout << "test_calculate_DFP_inverse_hessian_approximation\n";

   DataSet data_set(2, 1, 1);
   data_set.randomize_data_normal();
   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1});
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   QuasiNewtonMethod quasi_newton_method(&sum_squared_error);

   // Test 

   neural_network.initialize_parameters(1.0);

   Vector<double> old_parameters = neural_network.get_parameters();
   Vector<double> old_gradient = sum_squared_error.calculate_training_loss_gradient();
//   Matrix<double> old_inverse_hessian = sum_squared_error.calculate_inverse_hessian();

   neural_network.initialize_parameters(-0.5);

   Vector<double> parameters = neural_network.get_parameters();
   Vector<double> gradient = sum_squared_error.calculate_training_loss_gradient();
//   Matrix<double> inverse_hessian = sum_squared_error.calculate_inverse_hessian();

//   Matrix<double> DFP_inverse_hessian
//   = quasi_newton_method.calculate_DFP_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

//   assert_true(DFP_inverse_hessian == inverse_hessian, LOG);

   // Test 

   neural_network.initialize_parameters(1.0e-3);

   old_parameters = neural_network.get_parameters();
   old_gradient = sum_squared_error.calculate_training_loss_gradient();
//   old_inverse_hessian = sum_squared_error.calculate_inverse_hessian();

   neural_network.initialize_parameters(1.0e-6);

   parameters = neural_network.get_parameters();
   gradient = sum_squared_error.calculate_training_loss_gradient();
//   inverse_hessian = sum_squared_error.calculate_inverse_hessian();

//   DFP_inverse_hessian = quasi_newton_method.calculate_DFP_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

//   assert_true(DFP_inverse_hessian == inverse_hessian, LOG);

   // Test 

   neural_network.initialize_parameters(1.0e-6);

   old_parameters = neural_network.get_parameters();
   old_gradient = sum_squared_error.calculate_training_loss_gradient();
//   old_inverse_hessian = sum_squared_error.calculate_inverse_hessian();

   neural_network.initialize_parameters(1.0e-9);

   parameters = neural_network.get_parameters();
   gradient = sum_squared_error.calculate_training_loss_gradient();
//   inverse_hessian = sum_squared_error.calculate_inverse_hessian();

//   DFP_inverse_hessian = quasi_newton_method.calculate_DFP_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

//   assert_true(DFP_inverse_hessian == inverse_hessian, LOG);

   // Test 

   old_parameters.initialize(1.0e-3);
   parameters.initialize(1.0e-6);

   old_gradient.initialize(1.0e-3);
   gradient.initialize(1.0e-6);

//   old_inverse_hessian(0,0) = 0.75;
//   old_inverse_hessian(0,1) = -0.25;
//   old_inverse_hessian(1,0) = -0.25;
//   old_inverse_hessian(1,1) = 0.75;

//   DFP_inverse_hessian = quasi_newton_method.calculate_DFP_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);
}

/// @todo

void QuasiNewtonMethodTest::test_calculate_BFGS_inverse_hessian_approximation()
{
   cout << "test_calculate_BFGS_inverse_hessian_approximation\n";
/*
   DataSet data_set;
   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1});

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   sum_squared_error.set_regularization_method(LossIndex::L2);

   QuasiNewtonMethod quasi_newton_method(&sum_squared_error);

   neural_network.initialize_parameters(1.0);

   Vector<double> old_parameters = neural_network.get_parameters();
   Vector<double> old_gradient = sum_squared_error.calculate_training_loss_gradient();
//   Matrix<double> old_inverse_hessian = sum_squared_error.calculate_inverse_hessian();

   neural_network.initialize_parameters(-0.5);

   Vector<double> parameters = neural_network.get_parameters();
   Vector<double> gradient = sum_squared_error.calculate_training_loss_gradient();
//   Matrix<double> inverse_hessian = sum_squared_error.calculate_inverse_hessian();

//   Matrix<double> BFGS_inverse_hessian
//   = quasi_newton_method.calculate_BFGS_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

//   assert_true(BFGS_inverse_hessian == inverse_hessian, LOG);
*/
}


/// @todo

void QuasiNewtonMethodTest::test_calculate_inverse_hessian_approximation()
{
   cout << "test_calculate_inverse_hessian_approximation\n";

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1});
   DataSet data_set(2, 1, 1);
   data_set.randomize_data_normal();
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   QuasiNewtonMethod quasi_newton_method(&sum_squared_error);

   quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::DFP);

   neural_network.initialize_parameters(1.0);

   Vector<double> old_parameters = neural_network.get_parameters();
   Vector<double> old_gradient = sum_squared_error.calculate_training_loss_gradient();
//   Matrix<double> old_inverse_hessian = sum_squared_error.calculate_inverse_hessian();

   neural_network.initialize_parameters(-0.5);

   Vector<double> parameters = neural_network.get_parameters();
   Vector<double> gradient = sum_squared_error.calculate_training_loss_gradient();
//   Matrix<double> inverse_hessian = sum_squared_error.calculate_inverse_hessian();

//   Matrix<double> inverse_hessian_approximation
//   = quasi_newton_method.calculate_inverse_hessian_approximation(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

//   assert_true(inverse_hessian_approximation == inverse_hessian, LOG);

   quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::DFP);

   neural_network.initialize_parameters(1.0);

   old_parameters = neural_network.get_parameters();
   old_gradient = sum_squared_error.calculate_training_loss_gradient();
//   old_inverse_hessian = sum_squared_error.calculate_inverse_hessian();

   neural_network.initialize_parameters(-0.5);

   parameters = neural_network.get_parameters();
   gradient = sum_squared_error.calculate_training_loss_gradient();
//   inverse_hessian = sum_squared_error.calculate_inverse_hessian();

//   inverse_hessian_approximation
//   = quasi_newton_method.calculate_inverse_hessian_approximation(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

//   assert_true(inverse_hessian_approximation == inverse_hessian, LOG);

   // Test 

   old_parameters.initialize(1.0e-3);
   parameters.initialize(1.0e-6);

   old_gradient.initialize(1.0e-3);
   gradient.initialize(1.0e-6);

//   old_inverse_hessian(0,0) = 0.75;
//   old_inverse_hessian(0,1) = -0.25;
//   old_inverse_hessian(1,0) = -0.25;
//   old_inverse_hessian(1,1) = 0.75;

//   inverse_hessian_approximation
//   = quasi_newton_method.calculate_inverse_hessian_approximation(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);
}


void QuasiNewtonMethodTest::test_calculate_training_direction()
{
   cout << "test_calculate_training_direction\n";
}


void QuasiNewtonMethodTest::test_perform_training()
{
   cout << "test_perform_training\n";

   DataSet data_set(2, 1, 1);
   data_set.randomize_data_normal();
   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1, 1});
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   QuasiNewtonMethod quasi_newton_method(&sum_squared_error);
   quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::DFP);

   quasi_newton_method.set_reserve_all_training_history(true);

    // Test

   neural_network.initialize_parameters(3.1415927);

   double old_loss = sum_squared_error.calculate_training_loss();

   quasi_newton_method.set_maximum_epochs_number(2),
   quasi_newton_method.set_display(false);

   quasi_newton_method.perform_training();

   double loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   neural_network.initialize_parameters(3.1415927);

   double minimum_parameters_increment_norm = 0.1;

   quasi_newton_method.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   quasi_newton_method.set_loss_goal(0.0);
   quasi_newton_method.set_minimum_loss_decrease(0.0);
   quasi_newton_method.set_gradient_norm_goal(0.0);
   quasi_newton_method.set_maximum_epochs_number(10);
   quasi_newton_method.set_maximum_time(1000.0);

   quasi_newton_method.perform_training();

   // Performance goal

   neural_network.initialize_parameters(3.1415927);

   double loss_goal = 100.0;

   quasi_newton_method.set_minimum_parameters_increment_norm(0.0);
   quasi_newton_method.set_loss_goal(loss_goal);
   quasi_newton_method.set_minimum_loss_decrease(0.0);
   quasi_newton_method.set_gradient_norm_goal(0.0);
   quasi_newton_method.set_maximum_epochs_number(10);
   quasi_newton_method.set_maximum_time(1000.0);

   quasi_newton_method.perform_training();

   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < loss_goal, LOG);

   // Minimum evaluation improvement

   neural_network.initialize_parameters(3.1415927);

   double minimum_loss_increase = 100.0;

   quasi_newton_method.set_minimum_parameters_increment_norm(0.0);
   quasi_newton_method.set_loss_goal(0.0);
   quasi_newton_method.set_minimum_loss_decrease(minimum_loss_increase);
   quasi_newton_method.set_gradient_norm_goal(0.0);
   quasi_newton_method.set_maximum_epochs_number(10);
   quasi_newton_method.set_maximum_time(1000.0);

   quasi_newton_method.perform_training();

   // Gradient norm goal 

   neural_network.initialize_parameters(3.1415927);

   double gradient_norm_goal = 100.0;

   quasi_newton_method.set_minimum_parameters_increment_norm(0.0);
   quasi_newton_method.set_loss_goal(0.0);
   quasi_newton_method.set_minimum_loss_decrease(0.0);
   quasi_newton_method.set_gradient_norm_goal(gradient_norm_goal);
   quasi_newton_method.set_maximum_epochs_number(10);
   quasi_newton_method.set_maximum_time(1000.0);

   quasi_newton_method.perform_training();

//   double gradient_norm = sum_squared_error.calculate_training_loss_gradient().calculate_norm();
//   assert_true(gradient_norm < gradient_norm_goal, LOG);

}


void QuasiNewtonMethodTest::test_to_XML()   
{
   cout << "test_to_XML\n";

   QuasiNewtonMethod quasi_newton_method;

   tinyxml2::XMLDocument* document = quasi_newton_method.to_XML();
   assert_true(document != nullptr, LOG);

   delete document;
}


void QuasiNewtonMethodTest::test_resize_training_history()
{
    cout << "test_resize_training_history\n";

    DataSet data_set;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    QuasiNewtonMethod quasi_newton_method(&sum_squared_error);

    OptimizationAlgorithm::Results results;

    quasi_newton_method.set_reserve_all_training_history(true);

    // Test

    results.resize_training_history(2);

//    assert_true(results.elapsed_time_history.size() == 2, LOG);

}


void QuasiNewtonMethodTest::test_load()
{
   cout << "test_load\n";

   QuasiNewtonMethod quasi_newton_method;

   tinyxml2::XMLDocument* document = quasi_newton_method.to_XML();
   quasi_newton_method.from_XML(*document);

   delete document;
}


void QuasiNewtonMethodTest::test_set_reserve_all_training_history()
{
   cout << "test_set_reserve_all_training_history\n";

   QuasiNewtonMethod quasi_newton_method;
   quasi_newton_method.set_reserve_all_training_history(true);
}


void QuasiNewtonMethodTest::run_test_case()
{
   cout << "Running quasi-Newton method test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor(); 

   // Get methods

   test_get_inverse_hessian_approximation_method();
   test_get_inverse_hessian_approximation_method_name();

   // Set methods

   test_set_inverse_hessian_approximation_method();

   // Training methods

   test_calculate_DFP_inverse_hessian_approximation();
   test_calculate_BFGS_inverse_hessian_approximation();

   test_calculate_inverse_hessian_approximation();
   test_calculate_training_direction();

   test_perform_training();

   // Training history methods

   test_resize_training_history();
   test_set_reserve_all_training_history();

   // Serialization methods

   test_to_XML();   
   test_load();

   cout << "End of quasi-Newton method test case.\n";
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
