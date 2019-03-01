/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   W E I G H T E D   S Q U A R E D   E R R O R   T E S T   C L A S S                                          */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "weighted_squared_error_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

WeightedSquaredErrorTest::WeightedSquaredErrorTest() : UnitTesting()
{
}


// DESTRUCTOR

WeightedSquaredErrorTest::~WeightedSquaredErrorTest()
{
}


// METHODS


void WeightedSquaredErrorTest::test_constructor()
{
   message += "test_constructor\n";

   // Default

   WeightedSquaredError wse1;

   assert_true(wse1.has_neural_network() == false, LOG);
   assert_true(wse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   WeightedSquaredError wse2(&nn2);

   assert_true(wse2.has_neural_network() == true, LOG);
   assert_true(wse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   WeightedSquaredError wse3(&nn3, &ds3);

   assert_true(wse3.has_neural_network() == true, LOG);
   assert_true(wse3.has_data_set() == true, LOG);
}


void WeightedSquaredErrorTest::test_destructor()
{
}


void WeightedSquaredErrorTest::test_calculate_loss()
{
   message += "test_calculate_loss\n";

   Vector<double> parameters;

   NeuralNetwork nn(1, 1, 1);
   nn.initialize_parameters(0.0);

   DataSet ds(1, 1, 1);
   ds.initialize_data(0.0);

   WeightedSquaredError wse(&nn, &ds);
/*
   assert_true(wse.calculate_error() == 0.0, LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.get_parameters();

   ds.set(2, 1, 1);
   ds.generate_data_binary_classification(2, 1);

   assert_true(wse.calculate_error() == wse.calculate_error(parameters), LOG);

   // Test

   nn.set(3, 1);

   ds.set(2, 3, 1);
   ds.generate_data_binary_classification(2, 3);

   wse.set_weights();

   parameters = nn.get_parameters();

   assert_true(wse.calculate_error() == wse.calculate_error(parameters), LOG);

   // Test

   nn.set(3, 1);

   nn.initialize_parameters(0.0);

   ds.set(2, 3, 1);
   ds.generate_data_binary_classification(2, 3);

   wse.set_weights();

   assert_true(wse.get_positives_weight() == wse.get_negatives_weight(), LOG);

   // Test

   nn.set(3, 1);

   nn.initialize_parameters(0.0);

   ds.set(3, 3, 1);
   ds.generate_data_binary_classification(3, 3);

   wse.set_weights();

   assert_true(wse.get_positives_weight() != wse.get_negatives_weight(), LOG);
*/
}


void WeightedSquaredErrorTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";
/*
   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;

   Vector<double> parameters;

   DataSet ds;

   WeightedSquaredError wse(&nn, &ds);

   Vector<double> gradient;
   Vector<double> numerical_gradient;
   Vector<double> error;

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   wse.set_weights();

   gradient = wse.calculate_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(3, 4, 1);
   nn.initialize_parameters(0.0);

   ds.set(5, 3, 1);
   ds.initialize_data(0.0);

   wse.set(&nn, &ds);

   wse.set_weights();

   gradient = wse.calculate_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   multilayer_perceptron_architecture.set(3);
   multilayer_perceptron_architecture[0] = 2;
   multilayer_perceptron_architecture[1] = 1;
   multilayer_perceptron_architecture[2] = 1;

   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);

   ds.set(5, 2, 1);
   wse.set(&nn, &ds);
   ds.initialize_data(0.0);

   wse.set_weights();

   gradient = wse.calculate_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   wse.set_weights();

   gradient = wse.calculate_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(3, 4, 1);
   nn.initialize_parameters(0.0);

   ds.set(5, 3, 1);
   wse.set(&nn, &ds);
   ds.initialize_data(0.0);

   wse.set_weights();

   gradient = wse.calculate_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(1.0);
   parameters = nn.get_parameters();

   ds.set(2, 1, 1);
   ds.initialize_data(1.0);

   gradient = wse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(wse, &WeightedSquaredError::calculate_error, parameters);
   assert_true((gradient - numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   ds.initialize_data(1.0);

   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   wse.set_weights();

   gradient = wse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(wse, &WeightedSquaredError::calculate_error, parameters);
   error = (gradient - numerical_gradient).calculate_absolute_value();

   // Test

   nn.set(2, 1);
   nn.initialize_parameters(1.0);
   parameters = nn.get_parameters();

   ds.set(3, 2, 1);
   ds.generate_data_binary_classification(3, 2);

   wse.set_weights();

   gradient = wse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(wse, &WeightedSquaredError::calculate_error, parameters);

   assert_true((gradient - numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


void WeightedSquaredErrorTest::test_calculate_selection_error()
{
   message += "test_calculate_selection_error\n";

   NeuralNetwork nn(1, 1, 1);

   nn.initialize_parameters(0.0);

   DataSet ds(1, 1, 1);

   ds.get_instances_pointer()->set_selection();

   ds.initialize_data(0.0);

   WeightedSquaredError wse(&nn, &ds);
/*
   double selection_error = wse.calculate_selection_error();

   assert_true(selection_error == 0.0, LOG);
*/
}


void WeightedSquaredErrorTest::test_calculate_error_terms()
{
   message += "test_calculate_error_terms\n";
/*
   NeuralNetwork nn;
   Vector<size_t> hidden_layers_size;
   Vector<double> parameters;

   DataSet ds;
   
   WeightedSquaredError wse(&nn, &ds);

   double error;

   Vector<double> error_terms;

   // Test

   nn.set(2, 1);
   nn.randomize_parameters_normal();

   ds.set(3, 2, 2);
   ds.generate_data_binary_classification(3, 2);

   error = wse.calculate_all_instances_error();

   error_terms = wse.calculate_error_terms();

   assert_true(fabs((error_terms*error_terms).calculate_sum() - error) < 1.0e-3, LOG);

   // Test

   nn.set(3, 1);
   nn.randomize_parameters_normal();

   ds.set(9, 3, 1);
   ds.generate_data_binary_classification(9, 3);

   error = wse.calculate_all_instances_error();

   error_terms = wse.calculate_error_terms();

   assert_true(fabs((error_terms*error_terms).calculate_sum() - error) < 1.0e-3, LOG);
*/
}


// @todo Last test

void WeightedSquaredErrorTest::test_calculate_error_terms_Jacobian()
{
   message += "test_calculate_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;
   Vector<double> parameters;

   DataSet ds;

   WeightedSquaredError wse(&nn, &ds);

   Vector<double> error_gradient;

   Vector<double> error_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   // Test
/*
   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.generate_data_binary_classification(3, 1);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   wse.set(&nn, &ds);
   ds.generate_data_binary_classification(3, 3);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   multilayer_perceptron_architecture.set(3);
   multilayer_perceptron_architecture[0] = 2;
   multilayer_perceptron_architecture[1] = 1;
   multilayer_perceptron_architecture[2] = 2;

   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);

   ds.set(2, 2, 5);
   wse.set(&nn, &ds);
   ds.generate_data_binary_classification(3, 2);

   terms_Jacobian = wse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   ds.set(3, 1, 1);
   ds.generate_data_binary_classification(3, 1);

   terms_Jacobian = wse.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(wse, &WeightedSquaredError::calculate_error_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 1);
   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   ds.set(2, 2, 1);
   ds.generate_data_binary_classification(2, 2);

   terms_Jacobian = wse.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(wse, &WeightedSquaredError::calculate_error_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);
*/
   // Test

//   nn.set(2, 2, 2);
//   nn.randomize_parameters_normal();

//   ds.set(2, 2, 2);
//   ds.generate_data_binary_classification(4, 2);
   
//   error_gradient = wse.calculate_gradient();

//   error_terms = wse.calculate_error_terms();
//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

//   cout << (terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 << endl;
//   cout << error_gradient << endl;

//   assert_true(((terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 - error_gradient).calculate_absolute_value() < 1.0e-3, LOG);
}


void WeightedSquaredErrorTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
}


void WeightedSquaredErrorTest::test_to_XML()
{
   message += "test_to_XML\n";
}


void WeightedSquaredErrorTest::test_from_XML()
{
   message += "test_from_XML\n";
}


void WeightedSquaredErrorTest::run_test_case()
{
   message += "Running weighted squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_loss();

   test_calculate_selection_error();

   test_calculate_gradient();

   // Error terms methods

//   test_calculate_error_terms();

//   test_calculate_error_terms_Jacobian();

   // Loss Hessian methods

//   test_calculate_Hessian();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of weighted squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lewser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lewser General Public License for more details.

// You should have received a copy of the GNU Lewser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
