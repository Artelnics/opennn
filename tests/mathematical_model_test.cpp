/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T H E M A T I C A L   M O D E L   T E S T   C L A S S                                                  */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "mathematical_model_test.h"

using namespace OpenNN;


MathematicalModelTest::MathematicalModelTest() : UnitTesting() 
{   
}


MathematicalModelTest::~MathematicalModelTest()
{
}


void MathematicalModelTest::test_get_display()
{
   message += "test_get_display\n";
}


void MathematicalModelTest::test_set_display()
{
   message += "test_set_display\n";
}


void MathematicalModelTest::test_to_XML()   
{
   message += "test_to_XML\n";

   MathematicalModel mm;

   tinyxml2::XMLDocument* document;
   
   // Test

   document = mm.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void MathematicalModelTest::test_from_XML()   
{
   message += "test_from_XML\n";

   MathematicalModel mm1;
   MathematicalModel mm2;

   tinyxml2::XMLDocument* document;
   
   // Test

   document = mm1.to_XML();

   assert_true(document != NULL, LOG);

   mm2.from_XML(*document);

   assert_true(mm1 == mm2, LOG);

}


void MathematicalModelTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/mathematical_model.xml";

   MathematicalModel mm;

   // Test

   mm.save(file_name);
}


void MathematicalModelTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/mathematical_model.xml";

   MathematicalModel mm;

   // Test

   mm.save(file_name);
   mm.load(file_name);
}


void MathematicalModelTest::run_test_case()
{
   message += "Running mathematical model test case...\n";  

   // Constructor and destructor methods

   // Get methods

   test_get_display();

   // Set methods

   test_set_display();

   // Serialization methods

   test_to_XML();   
   test_from_XML();   

   test_save();
   test_load();

   message += "End of mathematical model test case.\n";
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
