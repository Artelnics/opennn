//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   T E S T   C L A S S                     
//
//   Artificial Intelligence Techniques SL
//   E-mail: artelnics@artelnics.com                                       

#include "bounding_layer_test.h"


BoundingLayerTest::BoundingLayerTest() : UnitTesting()
{
}


BoundingLayerTest::~BoundingLayerTest()
{
}


void BoundingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default constructor

   BoundingLayer bounding_layer_1;

   assert_true(bounding_layer_1.get_neurons_number() == 0, LOG);

}


void BoundingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   // Test

   bounding_layer.set();
   assert_true(bounding_layer.get_neurons_number() == 0, LOG);

   // Test

   bounding_layer.set(1);
   assert_true(bounding_layer.get_neurons_number() == 1, LOG);
}


void BoundingLayerTest::test_get_type()
{
   cout << "test_get_type\n";

   assert_true(bounding_layer.get_type() == Layer::Bounding, LOG);
}


void BoundingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   BoundingLayer bounding_layer;
   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   // Test

   bounding_layer.set(1);
   bounding_layer.set_lower_bound(0, -1.0);
   bounding_layer.set_upper_bound(0,  1.0);
   bounding_layer.set_bounding_method("Bounding");

   inputs.resize(1, 1);
   inputs(0) = -2.0;
   outputs = bounding_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs(0) == -1.0, LOG);

   // Test

   inputs(0) = 2.0;
   outputs = bounding_layer.calculate_outputs(inputs);
   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs(0) == 1.0, LOG);
}


void BoundingLayerTest::run_test_case()
{
   cout << "Running bounding layer test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Get methods

   // Bounding layer architecture

   test_get_neurons_number();

   // Lower and upper bounds

   test_calculate_outputs();

   cout << "End of bounding layer test case.\n\n";
}


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
