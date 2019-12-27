//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BOUNDINGLAYERTEST_H
#define BOUNDINGLAYERTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class BoundingLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public: 

   explicit BoundingLayerTest();

   virtual ~BoundingLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   // Architecture

   void test_get_neurons_number();
   
   // Variables bounds

   void test_get_lower_bounds();
   void test_get_lower_bound();

   void test_get_upper_bounds();
   void test_get_upper_bound();

   void test_get_bounds();
   void test_get_type();

   // Display messages

   void test_get_display();

   // Set methods

   void test_set();
   void test_set_default();

   // Variables bounds

   void test_set_lower_bounds();
   void test_set_lower_bound();

   void test_set_upper_bounds();
   void test_set_upper_bound();

   void test_set_bounds();

   // Display messages

   void test_set_display();

   // Output methods

   void test_calculate_outputs();
   void test_calculate_derivatives();

   // Expression methods

   void test_write_expression();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Unit testing methods

   void run_test_case();
};

#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
