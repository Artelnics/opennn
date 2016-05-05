/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   F I N A L   S O L U T I O N S   E R R O R   T E S T   C L A S S                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "final_solutions_error_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

FinalSolutionsErrorTest::FinalSolutionsErrorTest(void) : UnitTesting() 
{
}


// DESTRUCTOR

FinalSolutionsErrorTest::~FinalSolutionsErrorTest(void) 
{
}


// METHODS

// @todo

void FinalSolutionsErrorTest::test_constructor(void)
{
   message += "test_constructor\n";

   // Default

   NeuralNetwork nn;
   MathematicalModel mm;

   FinalSolutionsError fse(&nn, &mm);

   assert_true(fse.get_neural_network_pointer() != NULL, LOG);
   assert_true(fse.get_mathemtaical_model_pointer() != NULL, LOG);

}


void FinalSolutionsErrorTest::test_destructor(void)
{
   message += "test_destructor\n";
}

// @todo

void FinalSolutionsErrorTest::test_calculate_performance(void)
{
   message += "test_calculate_performance\n";

//   NeuralNetwork nn;
//   Vector<double> network_parameters; 

//   FinalSolutionsError fse;

//   double performance;

   // Test
}


void FinalSolutionsErrorTest::test_to_XML(void)   
{
	message += "test_to_XML\n"; 

    FinalSolutionsError fse;

    tinyxml2::XMLDocument* document = fse.to_XML();

    assert_true(document != NULL, LOG);

    delete document;

}


void FinalSolutionsErrorTest::test_from_XML(void)
{
	message += "test_from_XML\n"; 

    std::string file_name = "../data/final_solutions_error.xml";

//   FinalSolutionsError fse;

//   fse.set_display(false);

//   fse.save(file_name);

//   fse.load(file_name);

//   assert_true(fse.get_display() == false, LOG);
}




void FinalSolutionsErrorTest::run_test_case(void)
{
   message += "Running final solutions error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Performance methods

   test_calculate_performance();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   message += "End of final solutions error test case...\n";
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
