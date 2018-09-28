/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   A L G O R I T H M   T E S T   C L A S S                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "mock_training_algorithm.h"

#include "training_algorithm_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR 

TrainingAlgorithmTest::TrainingAlgorithmTest() : UnitTesting() 
{
}


// DESTRUCTOR

TrainingAlgorithmTest::~TrainingAlgorithmTest()
{
}


// METHODS


void TrainingAlgorithmTest::test_constructor()
{
   message += "test_constructor\n"; 

   LossIndex pf;

   // Test

   MockTrainingAlgorithm mta1;

   assert_true(mta1.has_loss_index() == false, LOG);

   // Test

   MockTrainingAlgorithm mta2(&pf);

   assert_true(mta2.has_loss_index() == true, LOG);
}


void TrainingAlgorithmTest::test_destructor()
{
   message += "test_destructor\n"; 

   MockTrainingAlgorithm* mta = new MockTrainingAlgorithm;

   delete mta;
}


void TrainingAlgorithmTest::test_get_loss_index_pointer()
{
   message += "test_get_loss_index_pointer\n"; 

   LossIndex pf;
   
   MockTrainingAlgorithm mta(&pf);

   assert_true(mta.get_loss_index_pointer() != NULL, LOG);
}


void TrainingAlgorithmTest::test_get_display()
{
   message += "test_get_warning_gradient_norm\n"; 

   MockTrainingAlgorithm mta;

   mta.set_display(false);

   assert_true(mta.get_display() == false, LOG);
}


void TrainingAlgorithmTest::test_set()
{
   message += "test_set\n"; 
}


void TrainingAlgorithmTest::test_set_default()
{
   message += "test_set_default\n"; 
}


void TrainingAlgorithmTest::test_set_loss_index_pointer()
{
   message += "test_set_loss_index_pointer\n"; 
}


void TrainingAlgorithmTest::test_set_display()
{
   message += "test_set_display\n"; 
}


void TrainingAlgorithmTest::test_perform_training()
{
   message += "test_perform_training\n";
}


void TrainingAlgorithmTest::test_to_XML()
{
   message += "test_to_XML\n";
}
   

void TrainingAlgorithmTest::test_from_XML()
{
   message += "test_from_XML\n";
}


void TrainingAlgorithmTest::test_print()
{
   message += "test_print\n";
}


void TrainingAlgorithmTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/training_algorithm.xml";

   MockTrainingAlgorithm mta;
   mta.save(file_name);
}


void TrainingAlgorithmTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/training_algorithm.xml";

   MockTrainingAlgorithm mta;
   mta.save(file_name);
   mta.load(file_name);
}


void TrainingAlgorithmTest::run_test_case()
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

