/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V A R I A B L E S   T E S T   C L A S S                                                                    */
/*                                                                                                              */ 
 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "variables_test.h"


using namespace OpenNN;

// GENERAL CONSTRUCTOR

VariablesTest::VariablesTest() : UnitTesting()
{
}


// DESTRUCTOR

VariablesTest::~VariablesTest()
{
}


// METHODS


void VariablesTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   Variables v1;

   assert_true(v1.get_variables_number() == 0, LOG);
   assert_true(v1.get_inputs_number() == 0, LOG);
   assert_true(v1.get_targets_number() == 0, LOG);

   // Variables number constructor

   Variables v2(1);

   assert_true(v2.get_variables_number() == 1, LOG);
   assert_true(v2.get_inputs_number() == 0, LOG);
   assert_true(v2.get_targets_number() == 0, LOG);

   // XML constructor

   tinyxml2::XMLDocument* v1_document = v1.to_XML();

   Variables v3(*v1_document);

   assert_true(v3.get_variables_number() == 0, LOG);
   assert_true(v3.get_inputs_number() == 0, LOG);
   assert_true(v3.get_targets_number() == 0, LOG);

   // Copy constructor 

   Variables v5(1, 1);

   Variables v6(v5);

   assert_true(v6.get_variables_number() == 2, LOG);
   assert_true(v6.get_inputs_number() == 1, LOG);
   assert_true(v6.get_targets_number() == 1, LOG);
}


void VariablesTest::test_destructor()
{
   message += "test_destructor\n";

   Variables* vip = new Variables;

   delete vip;
}


void VariablesTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   Variables v1(2);
   Variables v2 = v1;

   assert_true(v2.get_inputs_number() == 1, LOG);
   assert_true(v2.get_targets_number() == 1, LOG);
}


void VariablesTest::test_get_variables_number()
{
   message += "test_get_variables_number\n";

   Variables v;

   assert_true(v.get_variables_number() == 0, LOG);
}


void VariablesTest::test_get_inputs_number()
{
   message += "test_get_inputs_number\n";

   Variables v;

   assert_true(v.get_inputs_number() == 0, LOG);
}


void VariablesTest::test_get_targets_number()
{
   message += "test_get_targets_number\n";

   Variables v;

   assert_true(v.get_targets_number() == 0, LOG);
}


void VariablesTest::test_get_inputs_indices()
{
   message += "test_get_inputs_indices\n";

   Variables v;

   Vector<size_t> inputs_indices = v.get_inputs_indices();

   assert_true(inputs_indices.size() == 0, LOG);
}


void VariablesTest::test_get_targets_indices()
{
   message += "test_get_targets_indices\n";

   Variables v;

   Vector<size_t> targets_indices = v.get_targets_indices();

   assert_true(targets_indices.size() == 0, LOG);
}


void VariablesTest::test_get_used_indices()
{
    message += "test_get_used_indices\n";

    Variables v;

    Vector<size_t> used_indices;

    used_indices = v.get_used_indices();

    assert_true(used_indices.size() == 0, LOG);
}



void VariablesTest::test_get_names()
{
   message += "test_get_names\n";

   Variables v;

   Vector<string> names = v.get_names();

   assert_true(names.size() == 0, LOG);
}


void VariablesTest::test_get_name()
{
   message += "test_get_name\n";
}


void VariablesTest::test_get_inputs_name()
{
   message += "test_get_inputs_name\n";
}


void VariablesTest::test_get_targets_name()
{
   message += "test_get_targets_name\n";
}


void VariablesTest::test_get_units()
{
   message += "test_get_units\n";

   Variables v;

   Vector<string> units = v.get_units();

   assert_true(units.size() == 0, LOG);

}


void VariablesTest::test_get_unit()
{
   message += "test_get_unit\n";
}


void VariablesTest::test_get_inputs_units()
{
   message += "test_get_inputs_units\n";
}


void VariablesTest::test_get_targets_units()
{
   message += "test_get_targets_units\n";
}


void VariablesTest::test_get_descriptions()
{
   message += "test_get_descriptions\n";

   Variables v;

   Vector<string> descriptions = v.get_descriptions();

   assert_true(descriptions.size() == 0, LOG);

}


void VariablesTest::test_get_description()
{
   message += "test_get_description\n";
}


void VariablesTest::test_get_inputs_description()
{
   message += "test_get_inputs_description\n";
}


void VariablesTest::test_get_target_descriptions()
{
   message += "test_get_target_descriptions\n";
}


// @todo Columns number

void VariablesTest::test_get_information()
{
   message += "test_get_information\n";

   Variables v(1);

   Matrix<string> information = v.get_information();

   size_t rows_number = information.get_rows_number();
//   size_t columns_number = information.get_columns_number();

   assert_true(rows_number == 1, LOG);
//   assert_true(columns_number == 4, LOG);
}


void VariablesTest::test_get_display()
{
   message += "test_get_display\n";

   Variables v;

   v.set_display(true);

   assert_true(v.get_display() == true, LOG);

   v.set_display(false);

   assert_true(v.get_display() == false, LOG);
}


void VariablesTest::test_set()
{
   message += "test_set\n";

   Variables v;

   // Instances and inputs and target variables

   v.set(1);

   assert_true(v.get_inputs_number() == 0, LOG);
   assert_true(v.get_targets_number() == 0, LOG);
}


void VariablesTest::test_set_variables_number()
{
   message += "test_set_variables_number\n";

   Variables v(1);

   v.set_variables_number(2);

   assert_true(v.get_variables_number() == 2, LOG);
}


void VariablesTest::test_set_input()
{
   message += "test_set_input\n";
}


void VariablesTest::test_set_target()
{
   message += "test_set_target\n";
}


void VariablesTest::test_set_names()
{
   message += "test_set_names\n";
}


void VariablesTest::test_set_name()
{
   message += "test_set_name\n";
}


void VariablesTest::test_set_units()
{
   message += "test_set_units\n";
}


void VariablesTest::test_set_unit()
{
   message += "test_set_unit\n";
}


void VariablesTest::test_set_descriptions()
{
   message += "test_set_descriptions\n";
}


void VariablesTest::test_set_description()
{
   message += "test_set_description\n";
}


void VariablesTest::test_set_display()
{
   message += "test_set_display\n";
}


// @todo

void VariablesTest::test_convert_time_series()
{
    message += "test_convert_time_series\n";

    Variables v;

    // Test

    v.set(1);

//    v.convert_time_series(1);

//    assert_true(v.get_variables_number() == 2, LOG);
//    assert_true(v.get_inputs_number() == 1, LOG);
//    assert_true(v.get_targets_number() == 1, LOG);
}


void VariablesTest::test_to_XML()
{
   message += "test_to_XML\n";

   Variables v;
   
   tinyxml2::XMLDocument* document;

   // Test

   v.set(2);

   document = v.to_XML();

   assert_true(document != nullptr, LOG);

   // Test

   v.set(2);

   v.set_use(0, Variables::Target);
   v.set_use(1, Variables::Input);

   document = v.to_XML();

   v.set();

   v.from_XML(*document);

   assert_true(v.get_variables_number() == 2, LOG);
   assert_true(v.get_use(0) == Variables::Target, LOG);
   assert_true(v.get_use(1) == Variables::Input, LOG);

}


void VariablesTest::test_from_XML()
{
   message += "test_from_XML\n";

   Variables v;

   // Test

   v.set(3);

   tinyxml2::XMLDocument* document = v.to_XML();

   v.from_XML(*document);
}


void VariablesTest::run_test_case()
{
   message += "Running variables test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   test_get_variables_number();

   // Variables methods

   test_get_inputs_number();
   test_get_inputs_indices();

   test_get_targets_number();
   test_get_targets_indices();
   test_get_used_indices();

   // Information methods

   test_get_names();
   test_get_name();

   test_get_inputs_name();
   test_get_targets_name();

   test_get_units();
   test_get_unit();

   test_get_inputs_units();
   test_get_targets_units();

   test_get_descriptions();
   test_get_description();

   test_get_inputs_description();
   test_get_target_descriptions();
 
   test_get_information();

   test_get_display();

   // Set methods

   test_set();

   // Variables methods

   test_set_input();
   test_set_target();

   // Information methods

   test_set_names();
   test_set_name();

   test_set_units();
   test_set_unit();

   test_set_descriptions();
   test_set_description();

   test_set_display();

   // Data methods

   test_set_variables_number();

   test_convert_time_series();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of variables test case.\n";
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

