/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N S T A N C E S   T E S T    C L A S S   H E A D E R                                                     */
/*                                                                                                              */ 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INSTANCESTEST_H__
#define __INSTANCESTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class InstancesTest : public UnitTesting 
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:  

   // GENERAL CONSTRUCTOR

   explicit InstancesTest();


   // DESTRUCTOR

   virtual ~InstancesTest();


    // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   void test_get_instances_number();

   // Instances methods 

   void test_get_training_instances_number();
   void test_get_selection_instances_number();
   void test_get_testing_instances_number();

   void test_get_selection_indices();
   void test_get_training_indices();
   void test_get_testing_indices();
   void test_get_used_indices();

   void test_get_display();

   // Set methods

   void test_set();

   void test_set_instances_number();

   // Instances methods

   void test_set_training();
   void test_set_selection();
   void test_set_testing();

   void test_set_unused();

   void test_set_display();

   // Splitting methods

   void test_split_random_indices();
   void test_split_sequential_indices();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Unit testing methods

   void run_test_case();
};

#endif


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
