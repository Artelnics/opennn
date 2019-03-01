/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S                                      */
/*                                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "normalized_squared_error_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR


NormalizedSquaredErrorTest::NormalizedSquaredErrorTest(void) : UnitTesting() 
{
}


// DESTRUCTOR

NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest(void)
{
}


// METHODS

void NormalizedSquaredErrorTest::test_constructor(void)
{
   message += "test_constructor\n";

   // Default

   NormalizedSquaredError nse1;

   assert_true(nse1.has_neural_network() == false, LOG);
   assert_true(nse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork mlp2;
   NormalizedSquaredError nse2(&mlp2);

   assert_true(nse2.has_neural_network() == true, LOG);
   assert_true(nse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork mlp3;
   DataSet ds3;
   NormalizedSquaredError nse3(&mlp3, &ds3);

   assert_true(nse3.has_neural_network() == true, LOG);
   assert_true(nse3.has_data_set() == true, LOG);
}


void NormalizedSquaredErrorTest::test_destructor(void)
{
   message += "test_destructor\n";
}


void NormalizedSquaredErrorTest::test_calculate_error(void)
{
   message += "test_calculate_error\n";
/*
   Vector<double> parameters;

   NeuralNetwork nn(1,1);

   DataSet ds(1,1,1);

   MultilayerPerceptron* mlpp = nn.get_multilayer_perceptron_pointer();

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

   mlpp->get_layer_pointer(0)->set_activation_function(PerceptronLayer::Linear);

   mlpp->initialize_biases(0.0);
   mlpp->initialize_synaptic_weights(1.0);

   Matrix<double> new_data(2, 2);
   new_data(0,0) = -1.0;
   new_data(0,1) = -1.0;
   new_data(1,0) = 1.0;
   new_data(1,1) = 1.0;

   ds.set_data(new_data);

   NormalizedSquaredError nse(&nn, &ds);

   assert_true(nse.calculate_all_instances_error() == 0.0, LOG);

   // Test

   instances_number = 7;
   inputs_number = 8;
   outputs_number = 5;
   hidden_neurons = 3;

   nn.set(inputs_number, hidden_neurons, outputs_number);
   nn.randomize_parameters_normal();

   parameters = nn.get_parameters();

   ds.set(instances_number, inputs_number, outputs_number);
   ds.randomize_data_normal();

   nse.set_normalization_coefficient();

   assert_true(fabs(nse.calculate_all_instances_error() - nse.calculate_all_instances_error(parameters)) < std::numeric_limits<double>::min(), LOG);
*/
}


void NormalizedSquaredErrorTest::test_calculate_error_gradient(void)
{
   message += "test_calculate_error_gradient\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;

   Vector<double> parameters;

   DataSet ds;
   Matrix<double> data;

   NormalizedSquaredError nse(&nn, &ds);

   Vector<double> error_gradient;
   Vector<double> numerical_error_gradient;

   size_t instances_number = 10;
   size_t inputs_number = 10;
   size_t outputs_number = 9;
   size_t hidden_neurons = 5;

   Vector<size_t> indices;

   // Test 

   nn.set(1,1,1);

   nn.initialize_parameters(0.0);

   ds.set(2, 1, 1);

   data.set(2, 2);

   data(0,0) = -1.0;
   data(0,1) = -1.0;
   data(1,0) = 1.0;
   data(1,1) = 1.0;

   ds.set_data(data);

   error_gradient = nse.calculate_training_error_gradient();

   assert_true(error_gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(error_gradient == 0.0, LOG);

   // Test

   instances_number = 10;
   inputs_number = 7;
   outputs_number = 7;
   hidden_neurons = 5;

   nn.set(inputs_number, hidden_neurons, outputs_number);
   nn.randomize_parameters_normal();

   parameters = nn.get_parameters();

   ds.set(instances_number, inputs_number, outputs_number);
   ds.randomize_data_normal();

   nse.set_normalization_coefficient();

   indices.set(0, 1, instances_number-1);

   error_gradient = nse.calculate_training_error_gradient();
cout << "error gradient: " << error_gradient << endl;
system("pause");
   numerical_error_gradient = nd.calculate_gradient(nse, &NormalizedSquaredError::calculate_error, indices, parameters);

   assert_true((error_gradient - numerical_error_gradient).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   instances_number = 10;
   inputs_number = 7;
   outputs_number = 7;
   hidden_neurons = 5;

   nn.set(inputs_number, hidden_neurons, outputs_number);
   nn.randomize_parameters_normal();

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, PerceptronLayer::Logistic);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, PerceptronLayer::Logistic);

   parameters = nn.get_parameters();

   ds.set(instances_number, inputs_number, outputs_number);
   ds.randomize_data_normal();

   nse.set_normalization_coefficient();

   indices.set(0,1,instances_number-1);

   error_gradient = nse.calculate_training_error_gradient();
   numerical_error_gradient = nd.calculate_gradient(nse, &NormalizedSquaredError::calculate_error, indices, parameters);

   assert_true((error_gradient - numerical_error_gradient).calculate_absolute_value() < 1.0e-3, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_Hessian(void)
{
   message += "test_calculate_Hessian\n";
}


void NormalizedSquaredErrorTest::test_calculate_terms(void)
{
   message += "test_calculate_terms\n";
/*
   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;
   Vector<double> network_parameters;

   DataSet ds;

   NormalizedSquaredError nse(&nn, &ds);

   double error;

   Vector<double> evaluation_terms;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;

   // Test

   instances_number = 7;
   inputs_number = 6;
   outputs_number = 7;

   nn.set(inputs_number, outputs_number);
   nn.randomize_parameters_normal();

   ds.set(instances_number, inputs_number, outputs_number);
   ds.randomize_data_normal();

   nse.set_normalization_coefficient();

   error = nse.calculate_all_instances_error();

   evaluation_terms = nse.calculate_error_terms();

//   assert_true(fabs((evaluation_terms*evaluation_terms).calculate_sum() - error) < 1.0e-3, LOG);
*/
}


void NormalizedSquaredErrorTest::test_calculate_terms_Jacobian(void)
{
   message += "test_calculate_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<int> hidden_layers_size;
   Vector<double> network_parameters;

   DataSet ds;

   NormalizedSquaredError nse(&nn, &ds);

   Vector<double> error_gradient;

   Vector<double> evaluation_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   // Test
/*
   nn.set(1, 1);
   nn.randomize_parameters_normal();
   network_parameters = nn.get_parameters();

   ds.set(2, 1, 1);
   ds.randomize_data_normal();

   terms_Jacobian = nse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_error_terms, network_parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();
   network_parameters = nn.get_parameters();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   terms_Jacobian = nse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_error_terms, network_parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2,2,2);
   nn.randomize_parameters_normal();

   ds.set(2,2,2);
   ds.randomize_data_normal();
   
   error_gradient = nse.calculate_error_gradient();

   evaluation_terms = nse.calculate_error_terms();
   terms_Jacobian = nse.calculate_error_terms_Jacobian();

   assert_true(((terms_Jacobian.calculate_transpose()).dot(evaluation_terms)*2.0 - error_gradient).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


void NormalizedSquaredErrorTest::test_calculate_squared_errors(void)
{
    message += "test_calculate_squared_errors\n";

    NeuralNetwork nn;

    DataSet ds;

    NormalizedSquaredError nse(&nn, &ds);

    Vector<double> squared_errors;

    // Test

    nn.set(1, 1);
    nn.randomize_parameters_normal();

    ds.set(2, 1, 1);
    ds.randomize_data_normal();

    squared_errors = nse.calculate_squared_errors();

    assert_true(squared_errors.size() == 2, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_maximal_errors(void)
{
    message += "test_calculate_maximal_errors\n";

    NeuralNetwork nn;

    DataSet ds;

    NormalizedSquaredError nse(&nn, &ds);

    Vector<double> squared_errors;
    Vector<size_t> maximal_errors;

    // Test

    nn.set(1, 1);
    nn.randomize_parameters_normal();

    ds.set(3, 1, 1);
    ds.randomize_data_normal();

    squared_errors = nse.calculate_squared_errors();
    maximal_errors = nse.calculate_maximal_errors(3);

    assert_true(maximal_errors.size() == 3, LOG);

    assert_true(squared_errors.get_subvector(maximal_errors).is_decrescent(), LOG);
}


void NormalizedSquaredErrorTest::test_to_XML(void)
{
   message += "test_to_XML\n";
}


void NormalizedSquaredErrorTest::test_from_XML(void)
{
   message += "test_from_XML\n";
}


void NormalizedSquaredErrorTest::run_test_case(void)
{
   message += "Running normalized squared error test case...\n";

   // Constructor and destructor methods
/*
   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();
*/
   test_calculate_error_gradient();
/*
   test_calculate_Hessian();

   // Error terms methods

   test_calculate_terms();

   test_calculate_terms_Jacobian();

   // Squared errors methods

   test_calculate_squared_errors();

   test_calculate_maximal_errors();

   // Serialization methods

   test_to_XML();
   test_from_XML();
*/
   message += "End of normalized squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lenser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lenser General Public License for more details.

// You should have received a copy of the GNU Lenser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
