/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R A N D O M   S E A R C H   T E S T   C L A S S                                                            */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "random_search_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR 

RandomSearchTest::RandomSearchTest() : UnitTesting()
{
}


// DESTRUCTOR

/// Destructor.

RandomSearchTest::~RandomSearchTest()
{
}

// METHODS

void RandomSearchTest::test_constructor()
{
   message += "test_constructor\n"; 

   LossIndex pf;

   // Default constructor

   RandomSearch rs1; 
   assert_true(rs1.has_loss_index() == false, LOG);

   // Loss index constructor

   RandomSearch rs2(&pf); 
   assert_true(rs2.has_loss_index() == true, LOG);
}


void RandomSearchTest::test_destructor()
{
   message += "test_destructor\n";
}


void RandomSearchTest::test_get_training_rate_reduction_factor()
{
   message += "test_get_training_rate_reduction_factor\n";
}


void RandomSearchTest::test_get_reserve_parameters_history()
{
   message += "test_get_reserve_parameters_history\n";
}


void RandomSearchTest::test_get_reserve_parameters_norm_history()
{
   message += "test_get_reserve_parameters_norm_history\n";
}


void RandomSearchTest::test_get_reserve_loss_history()
{
   message += "test_get_reserve_loss_history\n";
}


void RandomSearchTest::test_set_training_rate_reduction_factor()
{
   message += "test_set_training_rate_reduction_factor\n";
}


void RandomSearchTest::test_set_reserve_parameters_history()
{
   message += "test_set_reserve_parameters_history\n";
}


void RandomSearchTest::test_set_reserve_parameters_norm_history()
{
   message += "test_set_reserve_parameters_norm_history\n";
}


void RandomSearchTest::test_set_reserve_loss_history()
{
   message += "test_set_reserve_loss_history\n";
}


void RandomSearchTest::test_calculate_training_direction()
{
   message += "test_calculate_training_direction\n";
}


void RandomSearchTest::test_perform_training()
{
   message += "test_perform_training\n";

   DataSet ds;

   NeuralNetwork nn;

   LossIndex pf(&nn, &ds);

   RandomSearch rs(&pf);

   RandomSearch::RandomSearchResults* rstr;

   // Test 

   nn.set(1, 1);
   
   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   rs.set_display(false);
   rs.set_maximum_iterations_number(1),

   rs.set_reserve_all_training_history(true);

   rs.set_display_period(1);
   
   rstr = rs.perform_training();

   assert_true(rstr != NULL, LOG);   

   delete rstr;
}


void RandomSearchTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   RandomSearch rs;
   rs.set_reserve_all_training_history(true);
}


void RandomSearchTest::test_to_XML()
{
   message += "test_to_XML\n";

   RandomSearch rs;

   tinyxml2::XMLDocument* document = rs.to_XML();

   // Test

   document = rs.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void RandomSearchTest::test_from_XML()
{
   message += "test_from_XML\n";

   RandomSearch rs1;
   RandomSearch rs2;

   tinyxml2::XMLDocument* document;

   // Test

   rs1.set_display(true);

   document = rs1.to_XML();

   rs2.from_XML(*document);

   delete document;

   assert_true(rs2 == rs1, LOG);
}


void RandomSearchTest::run_test_case()
{
   message += "Running random search test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor(); 

   // Get methods

   test_get_training_rate_reduction_factor();

   test_get_reserve_parameters_history();
   test_get_reserve_parameters_norm_history();

   test_get_reserve_loss_history();

   // Set methods

   test_set_training_rate_reduction_factor();

   test_set_reserve_parameters_history();
   test_set_reserve_parameters_norm_history();

   test_set_reserve_loss_history();

   // Training methods

   test_calculate_training_direction();

   test_perform_training();

   // Training history methods

   test_set_reserve_all_training_history();

   // Utiltity methods

   test_to_XML();
   test_from_XML();

   message += "End of random search test case.\n";
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
