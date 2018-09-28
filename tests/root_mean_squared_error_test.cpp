/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R O O T   M E A N   S Q U A R E D   E R R O R   T E S T   C L A S S                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "root_mean_squared_error_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

RootMeanSquaredErrorTest::RootMeanSquaredErrorTest() : UnitTesting() 
{
}


// DESTRUCTOR

/// Destructor.

RootMeanSquaredErrorTest::~RootMeanSquaredErrorTest()
{

}


// METHODS

void RootMeanSquaredErrorTest::test_constructor()
{
   message += "test_constructor\n";

   // Default

   RootMeanSquaredError rmse1;

   assert_true(rmse1.has_neural_network() == false, LOG);
   assert_true(rmse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   RootMeanSquaredError rmse2(&nn2);

   assert_true(rmse2.has_neural_network() == true, LOG);
   assert_true(rmse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   RootMeanSquaredError rmse3(&nn3, &ds3);

   assert_true(rmse3.has_neural_network() == true, LOG);
   assert_true(rmse3.has_data_set() == true, LOG);
}


void RootMeanSquaredErrorTest::test_destructor()
{
   message += "test_destructor\n";
}


void RootMeanSquaredErrorTest::test_calculate_loss()
{
   message += "test_calculate_loss\n";

   Vector<double> parameters;

   NeuralNetwork nn(1,1,1);
   nn.initialize_parameters(0.0);

   DataSet ds(1,1,1);
   ds.initialize_data(0.0);

   RootMeanSquaredError rmse(&nn, &ds);

   assert_true(rmse.calculate_error() == 0.0, LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   assert_true(rmse.calculate_error() == rmse.calculate_error(parameters), LOG);

}


void RootMeanSquaredErrorTest::test_calculate_gradient()
{NumericalDifferentiation nd;

    NeuralNetwork nn;

    Vector<double> network_parameters;

    DataSet ds;
    Matrix<double> data;

    RootMeanSquaredError nse(&nn, &ds);

    Vector<double> objective_gradient;
    Vector<double> numerical_objective_gradient;

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

    objective_gradient = nse.calculate_gradient();

    assert_true(objective_gradient.size() == nn.count_parameters_number(), LOG);
    assert_true(objective_gradient == 0.0, LOG);

    // Test

    nn.set(5, 4, 2);
    nn.randomize_parameters_normal();

    network_parameters = nn.arrange_parameters();

    ds.set(3, 5, 2);
    ds.randomize_data_normal();

    objective_gradient = nse.calculate_gradient();
    numerical_objective_gradient = nd.calculate_gradient(nse, &RootMeanSquaredError::calculate_error, network_parameters);

    assert_true((objective_gradient - numerical_objective_gradient).calculate_absolute_value() < 1.0e-3, LOG);

    // Test

    nn.set(5, 4, 2);
    nn.randomize_parameters_normal();

    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);
    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Logistic);

    network_parameters = nn.arrange_parameters();

    ds.set(3, 5, 2);
    ds.randomize_data_normal();

    objective_gradient = nse.calculate_gradient();
    numerical_objective_gradient = nd.calculate_gradient(nse, &RootMeanSquaredError::calculate_error, network_parameters);

    assert_true((objective_gradient - numerical_objective_gradient).calculate_absolute_value() < 1.0e-3, LOG);
}


void RootMeanSquaredErrorTest::test_calculate_selection_loss()
{
   message += "test_calculate_selection_loss\n";

   NeuralNetwork nn(1,1,1);

   nn.initialize_parameters(0.0);

   DataSet ds(1,1,1);

   ds.get_instances_pointer()->set_selection();

   ds.initialize_data(0.0);

   RootMeanSquaredError rmse(&nn, &ds);
}


void RootMeanSquaredErrorTest::test_to_XML()   
{
   message += "test_to_XML\n";
}


void RootMeanSquaredErrorTest::test_from_XML()   
{
   message += "test_from_XML\n";
}


void RootMeanSquaredErrorTest::run_test_case()
{
   message += "Running root mean squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Objective methods

   test_calculate_loss();
   test_calculate_selection_loss();

   test_calculate_gradient();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of root mean squared error test case.\n";
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
