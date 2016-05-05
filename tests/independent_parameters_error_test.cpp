/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "independent_parameters_error_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

IndependentParametersErrorTest::IndependentParametersErrorTest(void) : UnitTesting() 
{
}


// DESTRUCTOR

IndependentParametersErrorTest::~IndependentParametersErrorTest(void) 
{
}


// METHODS

void IndependentParametersErrorTest::test_constructor(void)
{
   message += "test_constructor\n";

   // Default

   IndependentParametersError ipe1;

   assert_true(ipe1.has_neural_network() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   IndependentParametersError ipe2(&nn2);

   assert_true(ipe2.get_neural_network_pointer() != NULL, LOG);

}


void IndependentParametersErrorTest::test_destructor(void)
{
   message += "test_destructor\n";
}


// @todo

void IndependentParametersErrorTest::test_calculate_performance(void)   
{
   message += "test_calculate_performance\n";

//   NeuralNetwork nn;
//   Vector<double> parameters; 

//   IndependentParametersError ipe(&nn);

//   double performance;

   // Test

//   nn.set(1);
//   nn.initialize_parameters(0.0);

//   performance = ipe.calculate_performance();

//   assert_true(performance == 0.0, LOG);

   // Test

//   nn.set(1, 1, 1);
//   nn.initialize_parameters(0.0);

//   objective = ipe.calculate_performance();

//   assert_true(objective == 1.0, LOG);
}


void IndependentParametersErrorTest::test_to_XML(void)   
{
	message += "test_to_XML\n"; 

    IndependentParametersError ipe;

    tinyxml2::XMLDocument* document;

    // Test

    document = ipe.to_XML();

    assert_true(document != NULL, LOG);

    delete document;

}


void IndependentParametersErrorTest::test_from_XML(void)
{
//   IndependentParametersError ipe;

//   assert_true(ipe.get_display() == false, LOG);
}


void IndependentParametersErrorTest::run_test_case(void)
{
   message += "Running independent parameters error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Objective methods

   test_calculate_performance();   

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   message += "End of independent parameters error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
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
