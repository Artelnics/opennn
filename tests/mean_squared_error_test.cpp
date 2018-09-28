/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M E A N   S Q U A R E D   E R R O R   T E S T   C L A S S                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "mean_squared_error_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

MeanSquaredErrorTest::MeanSquaredErrorTest() : UnitTesting() 
{
}


// DESTRUCTOR

MeanSquaredErrorTest::~MeanSquaredErrorTest()
{
}


// METHODS


void MeanSquaredErrorTest::test_constructor()
{
   message += "test_constructor\n";

   // Default

   MeanSquaredError mse1;

   assert_true(mse1.has_neural_network() == false, LOG);
   assert_true(mse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   MeanSquaredError mse2(&nn2);

   assert_true(mse2.has_neural_network() == true, LOG);
   assert_true(mse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   MeanSquaredError mse3(&nn3, &ds3);

   assert_true(mse3.has_neural_network() == true, LOG);
   assert_true(mse3.has_data_set() == true, LOG);

}


void MeanSquaredErrorTest::test_destructor()
{
}


void MeanSquaredErrorTest::test_calculate_loss()   
{
   message += "test_calculate_loss\n";

   Vector<double> parameters;

   NeuralNetwork nn(1, 1, 1);
   nn.initialize_parameters(0.0);

   DataSet ds(1, 1, 1);
   ds.initialize_data(0.0);

   MeanSquaredError mse(&nn, &ds);

   assert_true(mse.calculate_error() == 0.0, LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   assert_true(mse.calculate_error() == mse.calculate_error(parameters), LOG);

}


void MeanSquaredErrorTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;

   Vector<double> parameters;

   DataSet ds;

   MeanSquaredError mse(&nn, &ds);

   Vector<double> gradient;
   Vector<double> numerical_gradient;
   Vector<double> error;

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   gradient = mse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient = mse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   multilayer_perceptron_architecture.set(3);
   multilayer_perceptron_architecture[0] = 2;
   multilayer_perceptron_architecture[1] = 1;
   multilayer_perceptron_architecture[2] = 3;

   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);

   ds.set(2, 3, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient = mse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   gradient = mse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient = mse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(1.0);
   parameters = nn.arrange_parameters();

   ds.set(1, 1, 2);
   ds.initialize_data(1.0);

   gradient = mse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(mse, &MeanSquaredError::calculate_error, parameters);
   assert_true((gradient - numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   ds.initialize_data(1.0);

   nn.randomize_parameters_normal();
   parameters = nn.arrange_parameters();

   gradient = mse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(mse, &MeanSquaredError::calculate_error, parameters);
   error = (gradient - numerical_gradient).calculate_absolute_value();
}


void MeanSquaredErrorTest::test_calculate_selection_loss()   
{
   message += "test_calculate_selection_loss\n";

   NeuralNetwork nn(1, 1, 1);

   nn.initialize_parameters(0.0);

   DataSet ds(1, 1, 1);

   ds.get_instances_pointer()->set_selection();

   ds.initialize_data(0.0);

   MeanSquaredError mse(&nn, &ds);  

   double selection_error = mse.calculate_selection_error();

   assert_true(selection_error == 0.0, LOG);
}


void MeanSquaredErrorTest::test_calculate_terms()
{
   message += "test_calculate_terms\n";

   NeuralNetwork nn;
   Vector<size_t> hidden_layers_size;
   Vector<double> parameters;

   DataSet ds;
   
   MeanSquaredError mse(&nn, &ds);

   double objective;

   Vector<double> evaluation_terms;

   // Test

   nn.set(2, 2);
   nn.randomize_parameters_normal();

   ds.set(2, 2, 3);
   ds.randomize_data_normal();

   objective = mse.calculate_error();

   evaluation_terms = mse.calculate_terms();

   assert_true(fabs((evaluation_terms*evaluation_terms).calculate_sum() - objective) < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_calculate_terms_Jacobian()
{
   message += "test_calculate_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;
   Vector<double> parameters;

   DataSet ds;

   MeanSquaredError mse(&nn, &ds);

   Vector<double> objective_gradient;

   Vector<double> evaluation_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   terms_Jacobian = mse.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().count_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.count_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   terms_Jacobian = mse.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().count_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.count_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   multilayer_perceptron_architecture.set(3);
   multilayer_perceptron_architecture[0] = 2;
   multilayer_perceptron_architecture[1] = 1;
   multilayer_perceptron_architecture[2] = 2;

   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);

   ds.set(2, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   terms_Jacobian = mse.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().count_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.count_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.randomize_parameters_normal();
   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   terms_Jacobian = mse.calculate_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(mse, &MeanSquaredError::calculate_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();
   parameters = nn.arrange_parameters();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   terms_Jacobian = mse.calculate_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(mse, &MeanSquaredError::calculate_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();
   
   objective_gradient = mse.calculate_gradient();

   evaluation_terms = mse.calculate_terms();
   terms_Jacobian = mse.calculate_terms_Jacobian();

   assert_true(((terms_Jacobian.calculate_transpose()).dot(evaluation_terms)*2.0 - objective_gradient).calculate_absolute_value() < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
}


void MeanSquaredErrorTest::test_to_XML()
{
   message += "test_to_XML\n";
}


void MeanSquaredErrorTest::test_from_XML()
{
   message += "test_from_XML\n";
}


void MeanSquaredErrorTest::run_test_case()
{
   message += "Running mean squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Objective methods

   test_calculate_loss();   
   test_calculate_selection_loss();

   test_calculate_gradient();

   // Objective terms methods

   test_calculate_terms();
   test_calculate_terms_Jacobian();

   // Objective Hessian methods

   test_calculate_Hessian();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of mean squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lemser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lemser General Public License for more details.

// You should have received a copy of the GNU Lemser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
