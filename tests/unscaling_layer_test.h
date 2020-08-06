//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   T E S T   C L A S S   H E A D E R     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNSCALINGLAYERTEST_H
#define UNSCALINGLAYERTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class UnscalingLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit UnscalingLayerTest();

   virtual ~UnscalingLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   void test_get_dimensions();

   // Multilayer perceptron architecture 

   void test_get_inputs_number();
   void test_get_neurons_number();

   // Statistics

   void test_get_descriptives();
   void test_get_descriptives_matrix();

   void test_get_minimums();
   void test_get_maximums();

   // Variables scaling and unscaling

   void test_get_unscaling_method();
   void test_write_scaling_methods();

   // Display messages

   void test_get_display();

   // Set methods

   void test_set();
   void test_set_inputs_number();
   void test_set_neurons_number();
   void test_set_default();

   // Output variables descriptives

   void test_set_descriptives();
   void test_set_descriptives_eigen();
   void test_set_item_descriptives();

   void test_set_minimum();
   void test_set_maximum();
   void test_set_mean();
   void test_set_standard_deviation();

   // Variables scaling and unscaling

   void test_set_unscaling_method();

   // Display messages

   void test_set_display();

   // Check methods

   void test_is_empty();

   // Outputs unscaling

   void test_calculate_outputs();

   void test_calculate_minimum_maximum_outputs();
   void test_calculate_mean_standard_deviation_outputs();
   void test_calculate_logarithmic_outputs();

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
// Copyright (C) 2005-2020 Artificial Intelligence Techniques, SL.
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
