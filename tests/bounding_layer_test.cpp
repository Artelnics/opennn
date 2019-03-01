/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   B O U N D I N G   L A Y E R   T E S T   C L A S S                                                          */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   E-mail: artelnics@artelnics.com                                                                            */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "bounding_layer_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

BoundingLayerTest::BoundingLayerTest() : UnitTesting()
{
}


// DESTRUCTOR

BoundingLayerTest::~BoundingLayerTest()
{
}


// METHODS

void BoundingLayerTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   BoundingLayer bl1;

   assert_true(bl1.get_bounding_neurons_number() == 0, LOG);

   // Copy constructor

   bl1.set(2);

   BoundingLayer bl2(bl1);

   assert_true(bl2.get_bounding_neurons_number() == 2, LOG);
}


void BoundingLayerTest::test_destructor()
{
   message += "test_destructor\n";
}


void BoundingLayerTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   BoundingLayer bl_1;
   BoundingLayer bl_2 = bl_1;

   assert_true(bl_2.get_bounding_neurons_number() == 0, LOG);
}


void BoundingLayerTest::test_get_bounding_neurons_number()
{
   message += "test_get_bounding_neurons_number\n";

   BoundingLayer bl;

   // Test

   bl.set();
   assert_true(bl.get_bounding_neurons_number() == 0, LOG);

   // Test

   bl.set(1);
   assert_true(bl.get_bounding_neurons_number() == 1, LOG);
}


void BoundingLayerTest::test_set()
{
   message += "test_set\n";
}


void BoundingLayerTest::test_set_default()
{
   message += "test_set_default\n";
}


void BoundingLayerTest::test_get_lower_bounds()
{
   message += "test_get_lower_bounds\n";
}


void BoundingLayerTest::test_get_upper_bounds()
{
   message += "test_get_upper_bounds\n";
}


void BoundingLayerTest::test_get_lower_bound()
{
   message += "test_get_lower_bound\n";
}


void BoundingLayerTest::test_get_upper_bound()
{
   message += "test_get_upper_bound\n";
}


void BoundingLayerTest::test_get_bounds()
{
   message += "test_get_bounds\n";
}


void BoundingLayerTest::test_get_display()
{
   message += "test_get_display\n";
}


void BoundingLayerTest::test_set_lower_bounds()
{
   message += "test_set_lower_bounds\n";
}


void BoundingLayerTest::test_set_upper_bounds()
{
   message += "test_set_upper_bounds\n";
}


void BoundingLayerTest::test_set_lower_bound()
{
   message += "test_set_lower_bound\n";
}


void BoundingLayerTest::test_set_upper_bound()
{
   message += "test_set_upper_bound\n";
}


void BoundingLayerTest::test_set_bounds()
{
   message += "test_set_bounds\n";
}


void BoundingLayerTest::test_set_display()
{
   message += "test_set_display\n";
}


void BoundingLayerTest::test_initialize_random()
{
   message += "test_initialize_random\n";

   BoundingLayer bl;

   // Test

   bl.initialize_random();
}


void BoundingLayerTest::test_calculate_outputs()
{
   message += "test_calculate_outputs\n";
/*
   BoundingLayer bl(1);
   bl.set_lower_bound(0, -1.0);
   bl.set_upper_bound(0,  1.0);
   bl.set_bounding_method("Bounding");
   Vector<double> inputs(1);

   // Test

   Vector<double> outputs(1);
   inputs[0] = -2.0; 
   outputs = bl.calculate_outputs(inputs);
   assert_true(outputs.size() == 1, LOG);
   assert_true(outputs == -1.0, LOG);

   // Test

   inputs[0] = 2.0;
   outputs = bl.calculate_outputs(inputs);
   assert_true(outputs.size() == 1, LOG);
   assert_true(outputs == 1.0, LOG);
*/
}


void BoundingLayerTest::test_to_XML()
{
   message += "test_to_XML\n";

   BoundingLayer bl;

   tinyxml2::XMLDocument* document;

   // Test

   document = bl.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;

}


void BoundingLayerTest::test_from_XML()
{
   message += "test_from_XML\n";

   BoundingLayer bl;

   tinyxml2::XMLDocument* blep;

   // Test

   blep = bl.to_XML();

   bl.from_XML(*blep);

}


void BoundingLayerTest::test_write_expression()
{
   message += "test_write_expression\n";


}


void BoundingLayerTest::run_test_case()
{
   message += "Running bounding layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Bounding layer architecture

   test_get_bounding_neurons_number();

   // Variables bounds

   test_get_lower_bounds();
   test_get_lower_bound();

   test_get_upper_bounds();
   test_get_upper_bound();

   test_get_bounds();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Variables bounds

   test_set_lower_bounds();
   test_set_lower_bound();

   test_set_upper_bounds();
   test_set_upper_bound();

   test_set_bounds();

   // Display messages

   test_set_display();

   // Initialization methods

   test_initialize_random();

   // Lower and upper bounds

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of bounding layer test case.\n";
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
