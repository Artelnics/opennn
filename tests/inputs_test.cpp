/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   T E S T   C L A S S                                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "inputs_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

InputsTest::InputsTest(void) : UnitTesting()
{
}


// DESTRUCTOR

InputsTest::~InputsTest(void)
{
}


// METHODS

void InputsTest::test_constructor(void)
{
   message += "test_constructor\n";

   // Default constructor

   Inputs i1;

   assert_true(i1.get_inputs_number() == 0, LOG);

   // Inputs number constructor

   Inputs i2(1);

   assert_true(i2.get_inputs_number() == 1, LOG);

   // Copy constructor

   Inputs i3(i2);

   assert_true(i3.get_inputs_number() == 1, LOG);
}


void InputsTest::test_destructor(void)
{
   message += "test_destructor\n";
}


void InputsTest::test_assignment_operator(void)
{
   message += "test_assignment_operator\n";

   Inputs i_1;
   Inputs i_2 = i_1;

   assert_true(i_2.get_inputs_number() == 0, LOG);
}


void InputsTest::test_get_inputs_number(void)
{
   message += "test_get_inputs_number\n";

   Inputs i;

   // Test

   i.set();
   assert_true(i.get_inputs_number() == 0, LOG);

   // Test

   i.set(1);
   assert_true(i.get_inputs_number() == 1, LOG);
}


void InputsTest::test_set(void)
{
   message += "test_set\n";
}


void InputsTest::test_set_default(void)
{
   message += "test_set_default\n";
}


void InputsTest::test_arrange_names(void)
{
   message += "test_arrange_names\n";

   Inputs i;

   Vector<std::string> names = i.arrange_names();

   assert_true(names.size() == 0, LOG);

}


void InputsTest::test_get_name(void)
{
   message += "test_get_name\n";

   Inputs i(1);

   i.set_name(0, "x");

   assert_true(i.get_name(0) == "x", LOG);
}


void InputsTest::test_arrange_units(void)
{
   message += "test_arrange_units\n";

   Inputs i;

   Vector<std::string> units = i.arrange_units();

   assert_true(units.size() == 0, LOG);

}


void InputsTest::test_get_unit(void)
{
   message += "test_get_unit\n";

   Inputs i(1);

   i.set_unit(0, "m");

   assert_true(i.get_unit(0) == "m", LOG);
}


void InputsTest::test_arrange_descriptions(void)
{
   message += "test_arrange_descriptions\n";

   Inputs i;

   Vector<std::string> descriptions = i.arrange_descriptions();

   assert_true(descriptions.size() == 0, LOG);
}


void InputsTest::test_get_description(void)
{
   message += "test_get_description\n";

   Inputs i(1);

   i.set_description(0, "info");

   assert_true(i.get_description(0) == "info", LOG);
}


void InputsTest::test_get_display(void)
{
   message += "test_get_display\n";
}


void InputsTest::test_set_names(void)
{
   message += "test_set_names\n";
}


void InputsTest::test_set_name(void)
{
   message += "test_set_name\n";
}


void InputsTest::test_set_units(void)
{
   message += "test_set_units\n";
}


void InputsTest::test_set_unit(void)
{
   message += "test_set_unit\n";
}


void InputsTest::test_set_descriptions(void)
{
   message += "test_set_descriptions\n";
}


void InputsTest::test_set_description(void)
{
   message += "test_set_description\n";
}


void InputsTest::test_set_display(void)
{
   message += "test_set_display\n";
}


void InputsTest::test_to_XML(void)
{
   message += "test_to_XML\n";

   Inputs  i;

   tinyxml2::XMLDocument* document;

   document = i.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void InputsTest::test_from_XML(void)
{
   message += "test_from_XML\n";

   Inputs  i;

   tinyxml2::XMLDocument* document;

   // Test

   document = i.to_XML();

   i.from_XML(*document);

   delete document;
}


void InputsTest::run_test_case(void)
{
   message += "Running inputs test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Inputs number

   test_get_inputs_number();

   // Inputs information

   test_arrange_names();
   test_get_name();

   test_arrange_units();
   test_get_unit();

   test_arrange_descriptions();
   test_get_description();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();


   // Input variables information

   test_set_names();
   test_set_name();

   test_set_units();
   test_set_unit();

   test_set_descriptions();
   test_set_description();

   // Display messages

   test_set_display();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of inputs test case.\n";
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
