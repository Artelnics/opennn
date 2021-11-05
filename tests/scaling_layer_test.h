//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   T E S T   C L A S S   H E A D E R         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYERTEST_H
#define SCALINGLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class ScalingLayerTest : public UnitTesting
{

public:  

   explicit ScalingLayerTest();

   virtual ~ScalingLayerTest();

   // Constructor and destructor methods

   void test_constructor();

   // Set methods

   void test_set();

   void test_set_inputs_number();
   void test_set_neurons_number();

   void test_set_default();

   // Descriptives

   void test_set_descriptives();
   void test_set_item_descriptives();

   // Scaling method

   void test_set_scaling_method();

   // Input range

   void test_is_empty();

   void test_check_range();

   // Scaling 

   void test_calculate_outputs();
   void test_calculate_minimum_maximum_output();
   void test_calculate_mean_standard_deviation_output();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Unit testing methods

   void run_test_case();

private:

   ScalingLayer scaling_layer;

   Tensor<Descriptives, 1> descriptives;
};


#endif



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
