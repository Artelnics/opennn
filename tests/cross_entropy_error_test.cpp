/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   C R O S S   E N T R O P Y   E R R O R   T E S T   C L A S S                                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "cross_entropy_error_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

CrossEntropyErrorTest::CrossEntropyErrorTest(void) : UnitTesting() 
{
}


// DESTRUCTOR

CrossEntropyErrorTest::~CrossEntropyErrorTest(void)
{
}


// METHODS


// @todo

void CrossEntropyErrorTest::test_calculate_performance(void)
{
   message += "test_calculate_performance\n";

   NeuralNetwork nn;

   MultilayerPerceptron* mlpp;

   DataSet ds;
   
   CrossEntropyError cee(&nn, &ds);
//   double evaluation;

   // Test

   nn.set(1, 1);

   mlpp = nn.get_multilayer_perceptron_pointer();

   mlpp->get_layer_pointer(0)->set_activation_function(Perceptron::Logistic);
   nn.initialize_parameters(0.0);
   
   ds.set(1,1,1);
   ds.initialize_data(0.0);

//   objective = cee.calculate_performance();

//   assert_true(objective == 0.0, LOG);
}


// @todo

void CrossEntropyErrorTest::test_calculate_generalization_performance(void)   
{
   message += "test_calculate_generalization_performance\n";

   NeuralNetwork nn;
   DataSet ds;
   
   CrossEntropyError cee(&nn, &ds);

//   double generalization_objective;

   // Test

   nn.set(1, 1);
   nn.get_multilayer_perceptron_pointer()->get_layer_pointer(0)->set_activation_function(Perceptron::Logistic);
   nn.initialize_parameters(0.0);
   
   ds.set(1,1,1);
   ds.get_instances_pointer()->set_generalization();
   ds.initialize_data(0.0);
   
//   generalization_objective = cee.calculate_generalization_performance();

//   assert_true(generalization_objective == 0.0, LOG);

}

// @todo

void CrossEntropyErrorTest::test_calculate_gradient(void)
{
   message += "test_calculate_gradient\n";
/*
   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> architecture;

   Vector<double> parameters;

   DataSet ds;

   CrossEntropyError cee(&nn, &ds);

   Vector<double> gradient;
   Vector<double> numerical_gradient;
   Vector<double> numerical_differentiation_error;

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1,1,1);

   ds.initialize_data(0.0);

//   gradient = cee.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);

   nn.initialize_parameters(0.0);

   ds.set(5,3,2);
   cee.set(&nn, &ds);
   ds.initialize_data(0.0);

//   gradient = cee.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 2;
   architecture[1] = 1;
   architecture[2] = 3;

   nn.set(architecture);

   nn.initialize_parameters(0.0);

   ds.set(5,2,3);
   cee.set(&nn, &ds);
   ds.initialize_data(0.0);

//   gradient = cee.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1,1,1);

   ds.initialize_data(0.0);

//   gradient = cee.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3,4,2);

   nn.initialize_parameters(0.0);

   ds.set(5,3,2);
   cee.set(&nn, &ds);
   ds.initialize_data(0.0);

//   gradient = cee.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 2;
   architecture[1] = 1;
   architecture[2] = 3;

   nn.set(architecture);
   nn.initialize_parameters(0.0);

   ds.set(5,2,3);
   cee.set(&nn, &ds);
   ds.initialize_data(0.0);

//   gradient = cee.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(1.0);
   parameters = nn.arrange_parameters();

   ds.set(2,1,1);
   ds.initialize_data(0.5);

//   gradient = cee.calculate_gradient();
//   numerical_gradient = nd.calculate_gradient(cee, &CrossEntropyError::calculate_performance, parameters);   
//   assert_true((gradient - numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


void CrossEntropyErrorTest::test_calculate_Hessian(void)
{
   message += "test_calculate_Hessian\n";
}


void CrossEntropyErrorTest::test_to_XML(void)   
{
	message += "test_to_XML\n"; 
}


void CrossEntropyErrorTest::test_from_XML(void)
{
	message += "test_from_XML\n"; 
}


void CrossEntropyErrorTest::run_test_case(void)
{
   // Get methods

   // Set methods

   // Objective methods

   test_calculate_performance();   
   test_calculate_generalization_performance();

   test_calculate_gradient();
   test_calculate_Hessian();

   // Serialization methods

   test_to_XML();
   test_from_XML();

}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez.
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
