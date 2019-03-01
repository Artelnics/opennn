/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M I S S I N G   V A L U E S   T E S T   C L A S S                                                          */
/*                                                                                                              */ 
 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "missing_values_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

MissingValuesTest::MissingValuesTest() : UnitTesting()
{
}


// DESTRUCTOR

MissingValuesTest::~MissingValuesTest()
{
}


// METHODS

void MissingValuesTest::test_constructor()
{
   message += "test_constructor\n";

   // Test

   MissingValues mv0;

   assert_true(mv0.get_missing_values_number() == 0, LOG);

   // Test

   MissingValues mv2(1, 1);
   mv2.set_display(false);

   assert_true(mv2.get_display() == false, LOG);

}


void MissingValuesTest::test_destructor()
{
   message += "test_destructor\n";

   MissingValues* iip = new MissingValues(1, 1);

   delete iip;

}


void MissingValuesTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   MissingValues mv1(1,1);

   mv1.append(0, 0);

   MissingValues mv2 = mv1;

   assert_true(mv2.get_missing_values_number() == 1, LOG);
}


void MissingValuesTest::test_get_missing_values_number()
{
   message += "test_get_missing_values_number\n";

   MissingValues mv;

   assert_true(mv.get_missing_values_number() == 0, LOG);
}


void MissingValuesTest::test_get_display()
{
   message += "test_get_display\n";

   MissingValues mv;

   mv.set_display(true);

   assert_true(mv.get_display() == true, LOG);

   mv.set_display(false);

   assert_true(mv.get_display() == false, LOG);
}


void MissingValuesTest::test_set()
{
   message += "test_set\n";

   MissingValues mv;

   // Test

   mv.set(1, 1);

   mv.append(0, 0);

   assert_true(mv.get_missing_values_number() == 1, LOG);
}


void MissingValuesTest::test_set_missing_values_number()
{
   message += "test_set_missing_values_number\n";

   MissingValues mv(1,1);

   mv.set_missing_values_number(2);

   assert_true(mv.get_missing_values_number() == 2, LOG);
}


void MissingValuesTest::test_set_display()
{
   message += "test_set_display\n";
}


void MissingValuesTest::test_convert_time_series()
{
    message += "test_convert_time_series\n";

    MissingValues mv;

    // Test

    mv.set();

    mv.convert_time_series(0);

    assert_true(mv.get_missing_values_number() == 0, LOG);

    // Test

    mv.set(1,1);

    mv.append(0, 0);

    mv.convert_time_series(0);

    assert_true(mv.get_missing_values_number() == 1, LOG);

    // Test

    mv.set(1, 2);

    mv.append(0, 0);
    mv.append(0, 1);

    mv.convert_time_series(1);

//    assert_true(mv.get_missing_values_number() == 2, LOG);
}


// @todo Complete method and tests.

void MissingValuesTest::test_to_XML()
{
   message += "test_to_XML\n";

   MissingValues mv;

   tinyxml2::XMLDocument* document = mv.to_XML();

   assert_true(document != nullptr, LOG);

   // Test

   mv.set(2, 2);

   mv.set_scrubbing_method(MissingValues::Mean);

   document = mv.to_XML();

   mv.set();

   mv.from_XML(*document);

   assert_true(mv.get_instances_number() == 2, LOG);
   assert_true(mv.get_variables_number() == 2, LOG);
   assert_true(mv.get_scrubbing_method() == MissingValues::Mean, LOG);
}


// @todo Complete method and tests.

void MissingValuesTest::test_from_XML()
{
   message += "test_from_XML\n";

//   MissingValues mv;

//   tinyxml2::XMLDocument* document = i.to_XML();
   
//   i.from_XML(*document);

}


void MissingValuesTest::run_test_case()
{
   message += "Running missing values test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   test_get_missing_values_number();

   test_get_display();

   // Set methods

   test_set();

   test_set_display();

   test_set_missing_values_number();

   test_convert_time_series();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of missing values test case.\n";
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
