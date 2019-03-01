/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O U T P U T S   T E S T   C L A S S                                                                        */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "outputs_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

OutputsTest::OutputsTest() : UnitTesting()
{
}


// DESTRUCTOR

OutputsTest::~OutputsTest()
{
}


// METHODS

void OutputsTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   Outputs o1;

   assert_true(o1.get_outputs_number() == 0, LOG);

   // Outputs number constructor

   Outputs o2(2);

   assert_true(o2.get_outputs_number() == 2, LOG);

   // Copy constructor

   Outputs o3(o2);

   assert_true(o3.get_outputs_number() == 2, LOG);

}


void OutputsTest::test_destructor()
{
   message += "test_destructor\n";
}


void OutputsTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   Outputs o_1;
   Outputs o_2 = o_1;

   assert_true(o_2.get_outputs_number() == 0, LOG);

}


void OutputsTest::test_get_outputs_number()
{
   message += "test_get_outputs_number\n";

   Outputs o;

   // Test

   o.set();
   assert_true(o.get_outputs_number() == 0, LOG);

   // Test

   o.set(2);
   assert_true(o.get_outputs_number() == 2, LOG);
}


void OutputsTest::test_set()
{
   message += "test_set\n";
}


void OutputsTest::test_set_default()
{
   message += "test_set_default\n";
}


void OutputsTest::test_get_names()
{
   message += "test_get_names\n";

   Outputs o;

   Vector<string> names = o.get_names();

   assert_true(names.size() == 0, LOG);

}


void OutputsTest::test_get_name()
{
   message += "test_get_name\n";

   Outputs o(1);

   o.set_name(0, "y");

   assert_true(o.get_name(0) == "y", LOG);

}


void OutputsTest::test_get_units()
{
   message += "test_get_units\n";

   Outputs o;

   Vector<string> units = o.get_units();

   assert_true(units.size() == 0, LOG);
}


void OutputsTest::test_get_unit()
{
   message += "test_get_unit\n";

   Outputs o(1);

   o.set_unit(0, "m");

   assert_true(o.get_unit(0) == "m", LOG);
}


void OutputsTest::test_get_descriptions()
{
   message += "test_get_descriptions\n";

   Outputs o;

   Vector<string> descriptions = o.get_descriptions();

   assert_true(descriptions.size() == 0, LOG);
}


void OutputsTest::test_get_description()
{
   message += "test_get_description\n";

   Outputs o(1);

   o.set_description(0, "info");

   assert_true(o.get_description(0) == "info", LOG);
}


void OutputsTest::test_get_display()
{
   message += "test_get_display\n";
}


void OutputsTest::test_set_names()
{
   message += "test_set_names\n";
}


void OutputsTest::test_set_name()
{
   message += "test_set_name\n";
}


void OutputsTest::test_set_units()
{
   message += "test_set_units\n";
}


void OutputsTest::test_set_unit()
{
   message += "test_set_unit\n";
}


void OutputsTest::test_set_descriptions()
{
   message += "test_set_descriptions\n";
}


void OutputsTest::test_set_description()
{
   message += "test_set_description\n";
}


void OutputsTest::test_set_display()
{
   message += "test_set_display\n";
}


void OutputsTest::test_initialize_random()
{
   message += "test_initialize_random\n";

   Outputs o;

   // Test

//   o.initialize_random();

}


void OutputsTest::test_to_XML()
{
   message += "test_to_XML\n";

   Outputs  o;

   tinyxml2::XMLDocument* document;

   // Test

   document = o.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;
}


void OutputsTest::test_from_XML()
{
   message += "test_from_XML\n";

   Outputs  o;

   tinyxml2::XMLDocument* document;

   // Test

   document = o.to_XML();

   o.from_XML(*document);

   delete document;
}


void OutputsTest::run_test_case()
{
   message += "Running outputs test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Outputs number

   test_get_outputs_number();

   // Output variables information

   test_get_names();
   test_get_name();

   test_get_units();
   test_get_unit();

   test_get_descriptions();
   test_get_description();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Output variables information

   test_set_names();
   test_set_name();

   test_set_units();
   test_set_unit();

   test_set_descriptions();
   test_set_description();

   // Display messages

   test_set_display();

   // Initialization methods

   test_initialize_random();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of outputs test case.\n";
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
