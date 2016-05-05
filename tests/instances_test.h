/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N S T A N C E S   T E S T    C L A S S   H E A D E R                                                     */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
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

   explicit InstancesTest(void);


   // DESTRUCTOR

   virtual ~InstancesTest(void);


    // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   void test_get_instances_number(void);

   // Instances methods 

   void test_count_training_instances_number(void);
   void test_count_selection_instances_number(void);
   void test_count_testing_instances_number(void);

   void test_arrange_selection_indices(void);
   void test_arrange_training_indices(void);
   void test_arrange_testing_indices(void);
   void test_arrange_used_indices(void);

   void test_get_display(void);

   // Set methods

   void test_set(void);

   void test_set_instances_number(void);

   // Instances methods

   void test_set_training(void);
   void test_set_selection(void);
   void test_set_testing(void);

   void test_set_unused(void);

   void test_set_display(void);

   // Splitting methods

   void test_split_random_indices(void);
   void test_split_sequential_indices(void);

   // Serialization methods

   void test_to_XML(void);
   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);
};

#endif


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
