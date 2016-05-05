/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V A R I A B L E S   T E S T   C L A S S   H E A D E R                                                      */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __VARIABLESTEST_H__
#define __VARIABLESTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class VariablesTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:  

   // GENERAL CONSTRUCTOR

   explicit VariablesTest(void);

   // DESTRUCTOR

   virtual ~VariablesTest(void);

    // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   void test_get_variables_number(void);

   // Variables methods

   void test_count_inputs_number(void);
   void test_count_targets_number(void);

   void test_arrange_inputs_indices(void);
   void test_arrange_targets_indices(void);
   void test_arrange_used_indices(void);

   // Information methods 

   void test_arrange_names(void);
   void test_get_name(void);

   void test_arrange_inputs_name(void);
   void test_arrange_targets_name(void);

   void test_arrange_units(void);
   void test_get_unit(void);

   void test_arrange_inputs_units(void);
   void test_arrange_targets_units(void);

   void test_arrange_descriptions(void);
   void test_get_description(void);

   void test_arrange_inputs_description(void);
   void test_arrange_target_descriptions(void);

   void test_arrange_information(void);

   void test_get_display(void);

   // Set methods

   void test_set(void);

   void test_set_variables_number(void);

   // Variables methods

   void test_set_input(void);
   void test_set_target(void);

   // Information methods

   void test_set_names(void);
   void test_set_name(void);

   void test_set_units(void);
   void test_set_unit(void);

   void test_set_descriptions(void);
   void test_set_description(void);

   void test_set_display(void);

   void test_convert_time_series(void);

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
