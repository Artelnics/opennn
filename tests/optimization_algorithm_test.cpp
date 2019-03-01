/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   A L G O R I T H M   T E S T   C L A S S                                                  */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "mock_optimization_algorithm.h"

#include "optimization_algorithm_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR 

OptimizationAlgorithmTest::OptimizationAlgorithmTest() : UnitTesting() 
{
}


// DESTRUCTOR

OptimizationAlgorithmTest::~OptimizationAlgorithmTest()
{
}


// METHODS


void OptimizationAlgorithmTest::test_constructor()
{
   message += "test_constructor\n"; 

   SumSquaredError sse;

   // Test

   MockOptimizationAlgorithm mta1;

   assert_true(mta1.has_loss_index() == false, LOG);

   // Test

   MockOptimizationAlgorithm mta2(&sse);

   assert_true(mta2.has_loss_index() == true, LOG);
}


void OptimizationAlgorithmTest::test_destructor()
{
   message += "test_destructor\n"; 

   MockOptimizationAlgorithm* mta = new MockOptimizationAlgorithm;

   delete mta;
}


void OptimizationAlgorithmTest::test_get_loss_index_pointer()
{
   message += "test_get_loss_index_pointer\n"; 

   SumSquaredError sse;
   
   MockOptimizationAlgorithm mta(&sse);

   assert_true(mta.get_loss_index_pointer() != nullptr, LOG);
}


void OptimizationAlgorithmTest::test_get_display()
{
   message += "test_get_warning_gradient_norm\n"; 

   MockOptimizationAlgorithm mta;

   mta.set_display(false);

   assert_true(mta.get_display() == false, LOG);
}


void OptimizationAlgorithmTest::test_set()
{
   message += "test_set\n"; 
}


void OptimizationAlgorithmTest::test_set_default()
{
   message += "test_set_default\n"; 
}


void OptimizationAlgorithmTest::test_set_loss_index_pointer()
{
   message += "test_set_loss_index_pointer\n"; 
}


void OptimizationAlgorithmTest::test_set_display()
{
   message += "test_set_display\n"; 
}


void OptimizationAlgorithmTest::test_perform_training()
{
   message += "test_perform_training\n";
}


void OptimizationAlgorithmTest::test_to_XML()
{
   message += "test_to_XML\n";
}
   

void OptimizationAlgorithmTest::test_from_XML()
{
   message += "test_from_XML\n";
}


void OptimizationAlgorithmTest::test_print()
{
   message += "test_print\n";
}


void OptimizationAlgorithmTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/optimization_algorithm.xml";

   MockOptimizationAlgorithm mta;
   mta.save(file_name);
}


void OptimizationAlgorithmTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/optimization_algorithm.xml";

   MockOptimizationAlgorithm mta;
   mta.save(file_name);
   mta.load(file_name);
}


void OptimizationAlgorithmTest::run_test_case()
{
   message += "Running training algorithm test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_loss_index_pointer();
   test_get_display();

   // Set methods

   test_set_loss_index_pointer();
   test_set_display();

   test_set();
   test_set_default();   

   // Training methods

   test_perform_training();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   test_print();
   test_save();
   test_load();

   message += "End of training algorithm test case.\n";
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

