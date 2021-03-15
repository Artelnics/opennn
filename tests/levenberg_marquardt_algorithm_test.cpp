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
}


LevenbergMarquardtAlgorithmTest::~LevenbergMarquardtAlgorithmTest()
{
}


void LevenbergMarquardtAlgorithmTest::test_constructor()
{
   cout << "test_constructor\n"; 

   SumSquaredError sum_squared_error;

   // Default constructor

   LevenbergMarquardtAlgorithm lma1; 
   assert_true(!lma1.has_loss_index(), LOG);

   // Loss index constructor

   LevenbergMarquardtAlgorithm lma2(&sum_squared_error);
   assert_true(lma2.has_loss_index(), LOG);
}


void LevenbergMarquardtAlgorithmTest::test_destructor()
{
   cout << "test_destructor\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_damping_parameter()
{
   cout << "test_get_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_damping_parameter_factor()
{
   cout << "test_get_damping_parameter_factor\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_minimum_damping_parameter()
{
   cout << "test_get_minimum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_get_maximum_damping_parameter()
{
   cout << "test_get_maximum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_damping_parameter()
{
   cout << "test_set_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_damping_parameter_factor()
{
   cout << "test_set_damping_parameter_factor\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_minimum_damping_parameter()
{
   cout << "test_set_minimum_damping_parameter\n";
}


void LevenbergMarquardtAlgorithmTest::test_set_maximum_damping_parameter()
{
   cout << "test_set_maximum_damping_parameter\n";
}


/// @todo

void LevenbergMarquardtAlgorithmTest::test_calculate_training_loss()
{
    cout << "test_calculate_training_loss\n";

    DataSet data_set;
    Tensor<Index, 1> architecture;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    Tensor<type, 1> terms;

    type loss;

    LevenbergMarquardtAlgorithm lma(&sum_squared_error);

  // Test

//    data_set.set(2, 2, 2);
//    data_set.set_data_random();

//    architecture.setValues({2,2});

//    neural_network.set(NeuralNetwork::Approximation, architecture);
//    neural_network.set_parameters_random();

//    terms = sum_squared_error.calculate_training_error_terms();

//    loss = sum_squared_error.calculate_training_loss(terms);

//    assert_true(abs(loss-sum_squared_error.calculate_training_loss()) < 1.0e-3, LOG);
}


/// @todo

void LevenbergMarquardtAlgorithmTest::test_calculate_training_loss_gradient()
{
   cout << "test_calculate_training_loss_gradient\n";
//   DataSet data_set;

//   NeuralNetwork neural_network;

//   Tensor<Index, 1> architecture;

//   SumSquaredError sum_squared_error(&neural_network, &data_set);

//   Tensor<type, 1> terms;
//   Tensor<type, 2> terms_Jacobian;

//   Tensor<type, 1> gradient;
//   Tensor<type, 1> mse_gradient;

//   LevenbergMarquardtAlgorithm lma(&sum_squared_error);

   // Test

//   MeanSquaredError mean_squared_error(&neural_network, &data_set);

//   data_set.set(1, 1, 2);
//   data_set.set_data_random();

//   Tensor<type, 2> inputs = data_set.get_training_input_data();
//   Tensor<type, 2> targets = data_set.get_training_target_data();

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();

//   Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

//   terms = mean_squared_error.calculate_training_error_terms(outputs, targets);

//   terms_Jacobian = mse.calculate_error_terms_Jacobian(inputs,
//                                                       neural_network.forward_propagate(inputs),
//                                                       mean_squared_error.calculate_output_delta(outputs,targets)));

//   gradient = dot(terms_Jacobian.calculate_transpose(), terms);
//   mse_gradient = mean_squared_error.calculate_error_gradient();
// levenberg_marquardt_algorithm
//   cout << "columns sum: " << terms_Jacobian.calculate_columns_sum()*2.0 << endl;
//   cout << "gradient: " << gradient << endl;
//   cout << "mse_gradient: " << mse_gradient*2.0 << endl;

//    assert_true(absolute_value(gradient-mse_gradient) < 1.0e-3, LOG);

   // Test

//   data_set.set(1, 1, 2);
//   data_set.set_data_random();

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();

//   terms = sum_squared_error.calculate_training_error_terms();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

//   gradient = lma.calculate_gradient(terms, terms_Jacobian);

//   assert_true(absolute_value(gradient-sum_squared_error.calculate_gradient()) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);

//   neural_network.set_parameters_random();

//   terms= sum_squared_error.calculate_training_error_terms();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

//   gradient = lma.calculate_gradient(terms, terms_Jacobian);

//   assert_true(gradient == sum_squared_error.calculate_gradient(), LOG);

}


void LevenbergMarquardtAlgorithmTest::test_calculate_hessian_approximation()
{
   cout << "test_calculate_hessian_approximation\n";

//   NumericalDifferentiation nd;

//   NeuralNetwork neural_network;

//   Index parameters_number;

//   Tensor<Index, 1> architecture;

//   Tensor<type, 1> parameters;

//   DataSet data_set;

//   SumSquaredError sum_squared_error(&neural_network, &data_set);

//   Tensor<type, 2> terms_Jacobian;
//   Tensor<type, 2> hessian;
//   Tensor<type, 2> numerical_hessian;
//   Tensor<type, 2> hessian_approximation;

//   LevenbergMarquardtAlgorithm lma(&sum_squared_error);
   
   // Test

//   neural_network.set(NeuralNetwork::Approximation, {1, 2});
//   neural_network.set_parameters_constant(0.0);

//   parameters_number = neural_network.get_parameters_number();

//   data_set.set(1,2,2);
//   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

//   hessian_approximation = lma.calculate_hessian_approximation(terms_Jacobian);

//   assert_true(hessian_approximation.dimension(0) == parameters_number, LOG);
//   assert_true(hessian_approximation.dimension(1) == parameters_number, LOG);
//   assert_true(hessian_approximation.is_symmetric(), LOG);

   // Test

//   neural_network.set(NeuralNetwork::Approximation, {1,1,2});
//   neural_network.set_parameters_random();

//   parameters_number = neural_network.get_parameters_number();

//   data_set.set(1,2,3);
//   data_set.set_data_random();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

//   hessian_approximation = lma.calculate_hessian_approximation(terms_Jacobian);

//   assert_true(hessian_approximation.dimension(0) == parameters_number, LOG);
//   assert_true(hessian_approximation.dimension(1) == parameters_number, LOG);
//   assert_true(hessian_approximation.is_symmetric(), LOG);

   // Test

//   data_set.set(1, 1, 1);

//   data_set.set_data_random();

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);

//   parameters = neural_network.get_parameters();

//   neural_network.set_parameters_random();

//   numerical_hessian = nd.calculate_hessian(pf, &LossIndex::calculate_training_loss, parameters);

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

//   hessian_approximation = lma.calculate_hessian_approximation(terms_Jacobian);

//   assert_true(absolute_value(numerical_hessian - hessian_approximation) >= 0.0, LOG);

}


void LevenbergMarquardtAlgorithmTest::test_set_reserve_all_training_history()
{
   cout << "test_set_reserve_all_training_history\n";

   LevenbergMarquardtAlgorithm lma;
   lma.set_reserve_all_training_history(true);


   assert_true(lma.get_reserve_training_error_history(), LOG);
   assert_true(lma.get_reserve_selection_error_history(), LOG);
}


/// @todo

void LevenbergMarquardtAlgorithmTest::test_perform_training()
{
   cout << "test_perform_training\n";

   NeuralNetwork neural_network;
   
   DataSet data_set;
   
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   Tensor<type, 1> gradient;

   LevenbergMarquardtAlgorithm lma(&sum_squared_error);
   lma.set_display(false);

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

//   lma.perform_training();

//   loss = sum_squared_error.calculate_training_loss();

//   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   minimum_parameters_increment_norm = 100.0;

   lma.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   lma.set_loss_goal(0.0);
   lma.set_minimum_loss_decrease(0.0);
   lma.set_gradient_norm_goal(0.0);
   lma.set_maximum_epochs_number(10);
   lma.set_maximum_time(10.0);

//   lma.perform_training();

   // Performance goal

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   training_loss_goal = 100.0;

   lma.set_minimum_parameters_increment_norm(0.0);
   lma.set_loss_goal(training_loss_goal);
   lma.set_minimum_loss_decrease(0.0);
   lma.set_gradient_norm_goal(0.0);
   lma.set_maximum_epochs_number(10);
   lma.set_maximum_time(10.0);

//   lma.perform_training();

//   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < training_loss_goal, LOG);

   // Minimum loss increas

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   minimum_loss_decrease = 100.0;

   lma.set_minimum_parameters_increment_norm(0.0);
   lma.set_loss_goal(0.0);
   lma.set_minimum_loss_decrease(minimum_loss_decrease);
   lma.set_gradient_norm_goal(0.0);
   lma.set_maximum_epochs_number(10);
   lma.set_maximum_time(10.0);

//   lma.perform_training();

   // Gradient norm goal 

//   neural_network.set_parameters_random(0.0, 1.0e-3);

   gradient_norm_goal = 1.0e6;

   lma.set_minimum_parameters_increment_norm(0.0);
   lma.set_loss_goal(0.0);
   lma.set_minimum_loss_decrease(0.0);
   lma.set_gradient_norm_goal(gradient_norm_goal);
   lma.set_maximum_epochs_number(10);
   lma.set_maximum_time(10.0);

//   lma.perform_training();

//   gradient = sum_squared_error.calculate_training_loss_gradient();
//   gradient_norm = l2_norm(gradient);

   assert_true(gradient_norm < gradient_norm_goal, LOG);

}


void LevenbergMarquardtAlgorithmTest::test_resize_training_history()
{
   cout << "test_resize_training_history\n";

   LevenbergMarquardtAlgorithm lma;

   lma.set_reserve_all_training_history(true);

   TrainingResults lmatr;//(&lma);

   lmatr.resize_training_history(1);

   assert_true(lmatr.training_error_history.size() == 1, LOG);
   assert_true(lmatr.selection_error_history.size() == 1, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_to_XML()   
{
   cout << "test_to_XML\n";

   LevenbergMarquardtAlgorithm lma;

//   tinyxml2::XMLDocument* lmad = lma.to_XML();
   
//   assert_true(lmad != nullptr, LOG);
}


void LevenbergMarquardtAlgorithmTest::test_from_XML()
{
   cout << "test_from_XML\n";

   LevenbergMarquardtAlgorithm lma;
}


void LevenbergMarquardtAlgorithmTest::test_perform_Householder_QR_decomposition()
{
   cout << "test_perform_Householder_QR_decomposition\n";

   LevenbergMarquardtAlgorithm lma;

   Tensor<type, 2> a;
   Tensor<type, 1> b;

   Tensor<type, 2> inverse;

   // Test

//   a.set(1, 1, 1.0);

//   b.set(1, 0.0);

//   lma.perform_Householder_QR_decomposition(a, b);

//   assert_true(a == 1.0, LOG);
//   assert_true(b == 0.0, LOG);

   // Test

//   a.set(2, 2);
//   a.initialize_identity();

//   b.set(2, 0.0);

//   lma.perform_Householder_QR_decomposition(a, b);

//   inverse.set(2, 2);
//   inverse.initialize_identity();

//   assert_true(a == inverse, LOG);
//   assert_true(b == 0.0, LOG);

   // Test

//   a.set(100, 100);
   a.setRandom();
//   b.set(100);
   b.setRandom();

//   lma.perform_Householder_QR_decomposition(a, b);

   assert_true(a.dimension(0) == 100, LOG);
   assert_true(a.dimension(1) == 100, LOG);
   assert_true(b.size() == 100, LOG);
}


void LevenbergMarquardtAlgorithmTest::run_test_case()
{
   cout << "Running Levenberg-Marquardt algorithm test case...\n";

   // Constructor and destructor methods

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

   test_calculate_training_loss();
   test_calculate_training_loss_gradient();
   test_calculate_hessian_approximation();
   test_perform_training();

   // Training history methods

   test_set_reserve_all_training_history();
   test_resize_training_history();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

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
