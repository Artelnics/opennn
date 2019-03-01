/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N S T A N C E S   T E S T   C L A S S                                                                    */
/*                                                                                                              */ 
 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "instances_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

InstancesTest::InstancesTest() : UnitTesting() 
{
}


// DESTRUCTOR

InstancesTest::~InstancesTest()
{
}


// METHODS

void InstancesTest::test_constructor()
{
   message += "test_constructor\n";

   // Test

   Instances ii0;

   assert_true(ii0.get_instances_number() == 0, LOG);

   // Test

   Instances ii1(1);
  
   assert_true(ii1.get_training_instances_number() == 1, LOG);

   // Test

   Instances ii2(1);
   ii2.set_display(false);

   // Test

   Instances ii4;

   assert_true(ii4.get_training_instances_number() == 0, LOG);
   assert_true(ii4.get_selection_instances_number() == 0, LOG);
   assert_true(ii4.get_testing_instances_number() == 0, LOG);

   Instances ii5(1);

   Instances ii6(ii5);

   assert_true(ii6.get_training_instances_number() == 1, LOG);

}


void InstancesTest::test_destructor()
{
   message += "test_destructor\n";

   Instances* iip = new Instances(1);

   delete iip;

}


void InstancesTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   Instances ii1(1);
   Instances ii2 = ii1;

   assert_true(ii2.get_instances_number() == 1, LOG);   
}


void InstancesTest::test_get_instances_number() 
{
   message += "test_get_instances_number\n";

   Instances i;

   assert_true(i.get_instances_number() == 0, LOG);
}


void InstancesTest::test_get_training_instances_number() 
{
   message += "test_get_training_instances_number\n";

   Instances i;

   // Test

   i.set();

   assert_true(i.get_training_instances_number() == 0, LOG);

   // Test

   i.set(1);

   assert_true(i.get_training_instances_number() == 1, LOG);

}


void InstancesTest::test_get_training_indices() 
{
   message += "test_get_training_indices\n";

   Instances i;

   Vector<size_t> training_indices;

   training_indices = i.get_training_indices();

   assert_true(training_indices.size() == 0, LOG);
   
   i.set(1);

   training_indices = i.get_training_indices();

   assert_true(training_indices.size() == 1, LOG);

}


void InstancesTest::test_get_selection_instances_number() 
{
   message += "test_get_selection_instances_number\n";

   Instances i;

   assert_true(i.get_selection_instances_number() == 0, LOG);
   
   i.set(1);

   assert_true(i.get_selection_instances_number() == 0, LOG);
}


void InstancesTest::test_get_selection_indices() 
{
   message += "test_get_selection_indices\n";

   Instances i;

   Vector<size_t> selection_indices;

   selection_indices = i.get_selection_indices();

   assert_true(selection_indices.size() == 0, LOG);
   
   i.set(1);

   selection_indices = i.get_selection_indices();

   assert_true(selection_indices.size() == 0, LOG);
}


void InstancesTest::test_get_testing_instances_number() 
{
   message += "test_get_testing_instances_number\n";

   Instances i;

   assert_true(i.get_testing_instances_number() == 0, LOG);
   
   i.set(1);

   assert_true(i.get_testing_instances_number() == 0, LOG);

}


void InstancesTest::test_get_testing_indices() 
{
   message += "test_get_testing_indices\n";
 
   Instances i;

   Vector<size_t> testing_indices;

   testing_indices = i.get_testing_indices();

   assert_true(testing_indices.size() == 0, LOG);
   
   i.set(1);

   testing_indices = i.get_testing_indices();

   assert_true(testing_indices.size() == 0, LOG);
}


void InstancesTest::test_get_used_indices()
{
    message += "test_get_used_indices\n";

    Instances i;

    Vector<size_t> used_indices;

    used_indices = i.get_used_indices();

    assert_true(used_indices.size() == 0, LOG);

    i.set(1);

    used_indices = i.get_used_indices();

    assert_true(used_indices.size() == 1, LOG);

    i.set(4);

    i.set_use(0, Instances::Training);
    i.set_use(1, Instances::Unused);
    i.set_use(2, Instances::Testing);
    i.set_use(3, Instances::Selection);

    used_indices = i.get_used_indices();

    assert_true(used_indices.size() == 3, LOG);
    assert_true(used_indices[0] == 0, LOG);
    assert_true(used_indices[1] == 2, LOG);
    assert_true(used_indices[2] == 3, LOG);

}


void InstancesTest::test_get_display() 
{
   message += "test_get_display\n";

   Instances i;

   i.set_display(true);

   assert_true(i.get_display() == true, LOG);

   i.set_display(false);

   assert_true(i.get_display() == false, LOG);
}


void InstancesTest::test_set() 
{
   message += "test_set\n";

   Instances i;

   // Instances and inputs and target variables

   i.set(1);

   assert_true(i.get_instances_number() == 1, LOG);
}


void InstancesTest::test_set_instances_number() 
{
   message += "test_set_instances_number\n";

   Instances i(1);

   i.set_instances_number(2);

   assert_true(i.get_instances_number() == 2, LOG);
}


void InstancesTest::test_set_training()
{
   message += "test_set_training\n";
   
   Instances i(1);
   i.set_training();

   assert_true(i.get_training_instances_number() == 1, LOG);
   assert_true(i.get_selection_instances_number() == 0, LOG);
   assert_true(i.get_testing_instances_number() == 0, LOG);

   assert_true(i.get_training_indices() == 0, LOG);

}


void InstancesTest::test_set_selection()
{
   message += "test_set_selection\n";

   Instances i(1);
   i.set_selection();

   assert_true(i.get_training_instances_number() == 0, LOG);
   assert_true(i.get_selection_instances_number() == 1, LOG);
   assert_true(i.get_testing_instances_number() == 0, LOG);

   assert_true(i.get_selection_indices() == 0, LOG);

}


void InstancesTest::test_set_testing()
{
   message += "test_set_testing\n";

   Instances i(1);
   i.set_testing();

   assert_true(i.get_training_instances_number() == 0, LOG);
   assert_true(i.get_selection_instances_number() == 0, LOG);
   assert_true(i.get_testing_instances_number() == 1, LOG);

   assert_true(i.get_testing_indices() == 0, LOG);

}


void InstancesTest::test_set_unused()
{
    message += "test_set_unused\n";

    Instances i(5);

    // Test

    i.set_training();

    Vector<size_t> unused_indices(2);

    unused_indices[0] = 0;
    unused_indices[1] = 4;

    i.set_unused(unused_indices);

    assert_true(i.get_training_instances_number() == 3, LOG);
    assert_true(i.get_unused_instances_number() == 2, LOG);
    assert_true(i.get_use(0) == Instances::Unused, LOG);
    assert_true(i.get_use(1) == Instances::Training, LOG);
    assert_true(i.get_use(4) == Instances::Unused, LOG);
}


void InstancesTest::test_set_display() 
{
   message += "test_set_display\n";
}


void InstancesTest::test_split_random_indices() 
{
   message += "test_split_random_indices\n";

   size_t training_instances_number = 0;
   size_t selection_instances_number = 0;
   size_t testing_instances_number = 0;

   Instances i(12);

   // All data for training

   i.split_random_indices(1.0, 0.0, 0.0);
  
   training_instances_number = i.get_training_instances_number();
   selection_instances_number = i.get_selection_instances_number();
   testing_instances_number = i.get_testing_instances_number();

   assert_true(training_instances_number == 12, LOG);
   assert_true(selection_instances_number == 0, LOG);
   assert_true(testing_instances_number == 0, LOG);

   // All data for selection 

   i.split_random_indices(0.0, 10.0, 0.0);
  
   training_instances_number = i.get_training_instances_number();
   selection_instances_number = i.get_selection_instances_number();
   testing_instances_number = i.get_testing_instances_number();

   assert_true(training_instances_number == 0, LOG);
   assert_true(selection_instances_number == 12, LOG);
   assert_true(testing_instances_number == 0, LOG);

   // All data for testing 

   i.split_random_indices(0.0, 0.0, 100.0);
  
   training_instances_number = i.get_training_instances_number();
   selection_instances_number = i.get_selection_instances_number();
   testing_instances_number = i.get_testing_instances_number();

   assert_true(training_instances_number == 0, LOG);
   assert_true(selection_instances_number == 0, LOG);
   assert_true(testing_instances_number == 12, LOG);

   // Split data for training, selection and testing

   i.split_random_indices(0.5, 0.25, 0.25);
  
   training_instances_number = i.get_training_instances_number();
   selection_instances_number = i.get_selection_instances_number();
   testing_instances_number = i.get_testing_instances_number();

   assert_true(training_instances_number == 6, LOG);
   assert_true(selection_instances_number == 3, LOG);
   assert_true(testing_instances_number == 3, LOG);

   // Split data for training, selection and testing

   i.split_random_indices(2.0, 1.0, 1.0);
  
   training_instances_number = i.get_training_instances_number();
   selection_instances_number = i.get_selection_instances_number();
   testing_instances_number = i.get_testing_instances_number();

   assert_true(training_instances_number == 6, LOG);
   assert_true(selection_instances_number == 3, LOG);
   assert_true(testing_instances_number == 3, LOG);

   // Test

   i.set_instances_number(10);
   i.split_random_indices();

   training_instances_number = i.get_training_instances_number();
   selection_instances_number = i.get_selection_instances_number();
   testing_instances_number = i.get_testing_instances_number();

   assert_true(training_instances_number == 6, LOG);
   assert_true(selection_instances_number == 2, LOG);
   assert_true(testing_instances_number == 2, LOG);

}


void InstancesTest::test_split_sequential_indices()
{
   message += "test_split_sequential_indices\n";


  size_t training_instances_number = 0;
  size_t selection_instances_number = 0;
  size_t testing_instances_number = 0;

  Instances i(12);

  // All data for training

  i.split_sequential_indices(1.0, 0.0, 0.0);

  training_instances_number = i.get_training_instances_number();
  selection_instances_number = i.get_selection_instances_number();
  testing_instances_number = i.get_testing_instances_number();

  assert_true(training_instances_number == 12, LOG);
  assert_true(selection_instances_number == 0, LOG);
  assert_true(testing_instances_number == 0, LOG);

  // All data for selection

  i.split_sequential_indices(0.0, 10.0, 0.0);

  training_instances_number = i.get_training_instances_number();
  selection_instances_number = i.get_selection_instances_number();
  testing_instances_number = i.get_testing_instances_number();

  assert_true(training_instances_number == 0, LOG);
  assert_true(selection_instances_number == 12, LOG);
  assert_true(testing_instances_number == 0, LOG);

  // All data for testing

  i.split_sequential_indices(0.0, 0.0, 100.0);

  training_instances_number = i.get_training_instances_number();
  selection_instances_number = i.get_selection_instances_number();
  testing_instances_number = i.get_testing_instances_number();

  assert_true(training_instances_number == 0, LOG);
  assert_true(selection_instances_number == 0, LOG);
  assert_true(testing_instances_number == 12, LOG);

  // Split data for training, selection and testing

  i.split_sequential_indices(0.5, 0.25, 0.25);

  training_instances_number = i.get_training_instances_number();
  selection_instances_number = i.get_selection_instances_number();
  testing_instances_number = i.get_testing_instances_number();

  assert_true(training_instances_number == 6, LOG);
  assert_true(selection_instances_number == 3, LOG);
  assert_true(testing_instances_number == 3, LOG);

  // Split data for training, selection and testing

  i.split_sequential_indices(2.0, 1.0, 1.0);

  training_instances_number = i.get_training_instances_number();
  selection_instances_number = i.get_selection_instances_number();
  testing_instances_number = i.get_testing_instances_number();

  assert_true(training_instances_number == 6, LOG);
  assert_true(selection_instances_number == 3, LOG);
  assert_true(testing_instances_number == 3, LOG);

  // Test

  i.set_instances_number(10);
  i.split_sequential_indices();

  training_instances_number = i.get_training_instances_number();
  selection_instances_number = i.get_selection_instances_number();
  testing_instances_number = i.get_testing_instances_number();

  assert_true(training_instances_number == 6, LOG);
  assert_true(selection_instances_number == 2, LOG);
  assert_true(testing_instances_number == 2, LOG);
}


void InstancesTest::test_to_XML() 
{
   message += "test_to_XML\n";

//   Instances i;

//   tinyxml2::XMLDocument* document = i.to_XML();

//   assert_true(document != nullptr, LOG);

//   // Test

//   i.set(2);

//   i.set_use(0, Instances::Testing);
//   i.set_use(1, Instances::Unused);

//   document = i.to_XML();

//   i.set();

//   i.from_XML(*document);

//   assert_true(i.get_instances_number() == 2, LOG);
//   assert_true(i.get_use(0) == Instances::Testing, LOG);
//   assert_true(i.get_use(1) == Instances::Unused, LOG);
}


void InstancesTest::test_from_XML() 
{
   message += "test_from_XML\n";

//   Instances i;

//   tinyxml2::XMLDocument* document = i.to_XML();
   
//   i.from_XML(*document);

}


void InstancesTest::run_test_case()
{
   message += "Running instances test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   test_get_instances_number();

   // Instances methods 

   test_get_training_instances_number();
   test_get_training_indices();

   test_get_selection_instances_number();
   test_get_selection_indices();

   test_get_testing_instances_number();
   test_get_testing_indices();

   test_get_used_indices();

   test_get_display();

   // Set methods

   test_set();

   // Instances methods

   test_set_training();
   test_set_selection();
   test_set_testing();

   test_set_unused();

   test_set_display();

   // Data methods

   test_set_instances_number();

   // Splitting methods

   test_split_random_indices();
   test_split_sequential_indices();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of instances test case.\n";
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
