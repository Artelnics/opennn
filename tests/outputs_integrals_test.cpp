/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O U T P U T S   I N T E G R A L S   T E S T   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "outputs_integrals_test.h"

using namespace OpenNN;


OutputsIntegralsTest::OutputsIntegralsTest() : UnitTesting() 
{
}


OutputsIntegralsTest::~OutputsIntegralsTest() 
{
}


void OutputsIntegralsTest::test_constructor()
{
   message += "test_constructor\n";

   // Default

   OutputsIntegrals oi1;

   assert_true(oi1.has_neural_network() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   OutputsIntegrals oi2(&nn2);

   assert_true(oi2.get_neural_network_pointer() != NULL, LOG);

}


void OutputsIntegralsTest::test_destructor()
{
   message += "test_destructor\n";
}

// @todo

void OutputsIntegralsTest::test_calculate_loss()   
{
   message += "test_calculate_loss\n";
/*
   NeuralNetwork nn;
   Vector<double> network_parameters;

   OutputsIntegrals oi(&nn);

   double loss;

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);

   loss = oi.calculate_loss();

   assert_true(loss == 0.0, LOG);
*/
}

// @todo

void OutputsIntegralsTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";

//   NumericalDifferentiation nd;
//   NeuralNetwork nn;
//   OutputsIntegrals nnoi(&nn);

//   Vector<size_t> architecture;

//   Vector<double> parameters;
//   Vector<double> gradient;
//   Vector<double> numerical_gradient;
//   Vector<double> error;

   // Test 

//   nn.set(1, 1);
//   nn.initialize_parameters(0.0);

//   gradient = nnoi.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test 

//   nn.set(1, 2);
//   nn.initialize_parameters(0.0);

//   nnoi.set_neural_network_pointer(&nn);

//   gradient = nnoi.calculate_gradient();

//   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
//   assert_true(gradient == 0.0, LOG);

   // Test

//   nn.set(1, 2);
//   nn.initialize_parameters(1.0);
//   parameters = nn.arrange_parameters();

//   gradient = nnoi.calculate_gradient();
//   numerical_gradient = nd.calculate_gradient(nnoi, &OutputsIntegrals::calculate_loss, parameters);
//   error = (gradient - numerical_gradient).calculate_absolute_value();
//   assert_true(error < 1.0e-3, LOG);
}

// @todo

void OutputsIntegralsTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
}


void OutputsIntegralsTest::test_to_XML()   
{
	message += "test_to_XML\n"; 
}


void OutputsIntegralsTest::test_from_XML()
{
	message += "test_from_XML\n"; 
}


void OutputsIntegralsTest::run_test_case()
{
   message += "Running outputs integrals test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Evaluation methods

   test_calculate_loss();   

   test_calculate_gradient();

   test_calculate_Hessian();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   message += "End of outputs integrals test case.\n";
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
