/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "normalized_squared_error_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR


NormalizedSquaredErrorTest::NormalizedSquaredErrorTest() : UnitTesting() 
{
}


// DESTRUCTOR

NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest()
{
}


// METHODS

void NormalizedSquaredErrorTest::test_constructor()
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


void NormalizedSquaredErrorTest::test_destructor()
{
   message += "test_destructor\n";
}


void NormalizedSquaredErrorTest::test_calculate_error()
{
   message += "test_calculate_error\n";

   Vector<double> parameters;

   NeuralNetwork nn(1,1);

   DataSet ds(1,1,1);

   MultilayerPerceptron* mlpp = nn.get_multilayer_perceptron_pointer();

   mlpp->get_layer_pointer(0)->set_activation_function(Perceptron::Linear);

   mlpp->initialize_biases(0.0);
   mlpp->initialize_synaptic_weights(1.0);

   Matrix<double> new_data(2, 2);
   new_data(0,0) = -1.0;
   new_data(0,1) = -1.0;
   new_data(1,0) = 1.0;
   new_data(1,1) = 1.0;

   ds.set_data(new_data);

   NormalizedSquaredError nse(&nn, &ds);

   assert_true(nse.calculate_error() == 0.0, LOG);

   // Test

   nn.set(1, 2);
   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   ds.set(2, 1, 2);
   ds.randomize_data_normal();

   assert_true(nse.calculate_error() == nse.calculate_error(parameters), LOG);
}


void NormalizedSquaredErrorTest::test_calculate_selection_error()
{
   message += "test_calculate_selection_error\n";

   NeuralNetwork nn;

   DataSet ds;

   NormalizedSquaredError nse(&nn, &ds);

   double selection_loss;

   // Test

   nn.set(2,2,2);

   ds.set(2,2,2);

   ds.get_instances_pointer()->set_selection();

   ds.randomize_data_normal();

   selection_loss = nse.calculate_selection_error();

   assert_true(selection_loss != 0.0, LOG);

}


// @todo

void NormalizedSquaredErrorTest::test_calculate_gradient()
{
//   message += "test_calculate_gradient\n";

//   NumericalDifferentiation nd;

//   NeuralNetwork nn;

//   Vector<double> network_parameters;

//   DataSet ds;
//   Matrix<double> data;

//   NormalizedSquaredError nse(&nn, &ds);

//   Vector<double> objective_gradient;
//   Vector<double> numerical_objective_gradient;

//   // Test

//   nn.set(1,1,1);

//   nn.initialize_parameters(0.0);

//   ds.set(2, 1, 1);

//   data.set(2, 2);
//   data(0,0) = -1.0;
//   data(0,1) = -1.0;
//   data(1,0) = 1.0;
//   data(1,1) = 1.0;

//   ds.set_data(data);

//   objective_gradient = nse.calculate_gradient();

//   assert_true(objective_gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(objective_gradient == 0.0, LOG);

//   // Test

//   nn.set(5, 4, 2);
//   nn.randomize_parameters_normal();

//   network_parameters = nn.arrange_parameters();

//   ds.set(3, 5, 2);
//   ds.randomize_data_normal();

//   objective_gradient = nse.calculate_gradient();
//   numerical_objective_gradient = nd.calculate_gradient(nse, &NormalizedSquaredError::calculate_error, network_parameters);

//   assert_true((objective_gradient - numerical_objective_gradient).calculate_absolute_value() < 1.0e-3, LOG);

//   // Test

//   nn.set(5, 4, 2);
//   nn.randomize_parameters_normal();

//   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);
//   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Logistic);

//   network_parameters = nn.arrange_parameters();

//   ds.set(3, 5, 2);
//   ds.randomize_data_normal();

//   objective_gradient = nse.calculate_gradient();
//   numerical_objective_gradient = nd.calculate_gradient(nse, &NormalizedSquaredError::calculate_error, network_parameters);

//   assert_true((objective_gradient - numerical_objective_gradient).calculate_absolute_value() < 1.0e-3, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
}


// @todo

void NormalizedSquaredErrorTest::test_calculate_terms()
{
//   message += "test_calculate_terms\n";

//   NeuralNetwork nn;
//   Vector<size_t> multilayer_perceptron_architecture;
//   Vector<double> network_parameters;

//   DataSet ds;

//   NormalizedSquaredError nse(&nn, &ds);

//   double objective;

//   Vector<double> evaluation_terms;

//   // Test

//   nn.set(2, 2);
//   nn.randomize_parameters_normal();

//   ds.set(3, 2, 2);
//   ds.randomize_data_normal();

//   objective = nse.calculate_error();

//   evaluation_terms = nse.calculate_terms();

//   assert_true(fabs((evaluation_terms*evaluation_terms).calculate_sum() - objective) < 1.0e-3, LOG);

}


// @todo

void NormalizedSquaredErrorTest::test_calculate_terms_Jacobian()
{
//   message += "test_calculate_terms_Jacobian\n";

//   NumericalDifferentiation nd;

//   NeuralNetwork nn;
//   Vector<int> hidden_layers_size;
//   Vector<double> network_parameters;

//   DataSet ds;

//   NormalizedSquaredError nse(&nn, &ds);

//   Vector<double> objective_gradient;

//   Vector<double> evaluation_terms;
//   Matrix<double> terms_Jacobian;
//   Matrix<double> numerical_Jacobian_terms;

//   // Test

//   nn.set(1, 1);
//   nn.randomize_parameters_normal();
//   network_parameters = nn.arrange_parameters();

//   ds.set(2, 1, 1);
//   ds.randomize_data_normal();

//   terms_Jacobian = nse.calculate_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_terms, network_parameters);

//   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

//   // Test

//   nn.set(2, 2, 2);
//   nn.randomize_parameters_normal();
//   network_parameters = nn.arrange_parameters();

//   ds.set(2, 2, 2);
//   ds.randomize_data_normal();

//   terms_Jacobian = nse.calculate_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_terms, network_parameters);

//   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

//   // Test

//   nn.set(2,2,2);
//   nn.randomize_parameters_normal();

//   ds.set(2,2,2);
//   ds.randomize_data_normal();
   
//   objective_gradient = nse.calculate_gradient();

//   evaluation_terms = nse.calculate_terms();
//   terms_Jacobian = nse.calculate_terms_Jacobian();

//   assert_true(((terms_Jacobian.calculate_transpose()).dot(evaluation_terms)*2.0 - objective_gradient).calculate_absolute_value() < 1.0e-3, LOG);


}


void NormalizedSquaredErrorTest::test_calculate_squared_errors()
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


void NormalizedSquaredErrorTest::test_calculate_maximal_errors()
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


void NormalizedSquaredErrorTest::test_to_XML()
{
   message += "test_to_XML\n";
}


void NormalizedSquaredErrorTest::test_from_XML()
{
   message += "test_from_XML\n";
}


void NormalizedSquaredErrorTest::run_test_case()
{
   message += "Running normalized squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Objective methods

   test_calculate_error();
   test_calculate_selection_error();

   test_calculate_gradient();

   test_calculate_Hessian();

   // Objective terms methods

   test_calculate_terms();

   test_calculate_terms_Jacobian();

   // Squared errors methods

   test_calculate_squared_errors();

   test_calculate_maximal_errors();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of normalized squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
