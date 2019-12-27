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

   // Copy constructor

   bounding_layer_1.set(2);

   BoundingLayer bounding_layer2(bounding_layer_1);

   assert_true(bounding_layer2.get_neurons_number() == 2, LOG);
}


void BoundingLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void BoundingLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   BoundingLayer bounding_layer_1;
   BoundingLayer bounding_layer_2 = bounding_layer_1;

   assert_true(bounding_layer_2.get_neurons_number() == 0, LOG);
}


void BoundingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   BoundingLayer bounding_layer;

   // Test

   bounding_layer.set();
   assert_true(bounding_layer.get_neurons_number() == 0, LOG);

   // Test

   bounding_layer.set(1);
   assert_true(bounding_layer.get_neurons_number() == 1, LOG);
}


void BoundingLayerTest::test_set()
{
   cout << "test_set\n";
}


void BoundingLayerTest::test_set_default()
{
   cout << "test_set_default\n";
}


void BoundingLayerTest::test_get_lower_bounds()
{
   cout << "test_get_lower_bounds\n";
}


void BoundingLayerTest::test_get_upper_bounds()
{
   cout << "test_get_upper_bounds\n";
}


void BoundingLayerTest::test_get_lower_bound()
{
   cout << "test_get_lower_bound\n";
}


void BoundingLayerTest::test_get_upper_bound()
{
   cout << "test_get_upper_bound\n";
}


void BoundingLayerTest::test_get_bounds()
{
   cout << "test_get_bounds\n";
}


void BoundingLayerTest::test_get_type()
{
   cout << "test_get_type\n";

   BoundingLayer bounding_layer;

   assert_true(bounding_layer.get_type() == Layer::Bounding, LOG);

}


void BoundingLayerTest::test_get_display()
{
   cout << "test_get_display\n";
}


void BoundingLayerTest::test_set_lower_bounds()
{
   cout << "test_set_lower_bounds\n";
}


void BoundingLayerTest::test_set_upper_bounds()
{
   cout << "test_set_upper_bounds\n";
}


void BoundingLayerTest::test_set_lower_bound()
{
   cout << "test_set_lower_bound\n";
}


void BoundingLayerTest::test_set_upper_bound()
{
   cout << "test_set_upper_bound\n";
}


void BoundingLayerTest::test_set_bounds()
{
   cout << "test_set_bounds\n";
}


void BoundingLayerTest::test_set_display()
{
   cout << "test_set_display\n";
}


void BoundingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   BoundingLayer bounding_layer(1);
   bounding_layer.set_lower_bound(0, -1.0);
   bounding_layer.set_upper_bound(0,  1.0);
   bounding_layer.set_bounding_method("Bounding");

   Tensor<double> inputs(1, 1);

   // Test

   Tensor<double> outputs(1, 1);
   inputs[0] = -2.0; 
   outputs = bounding_layer.calculate_outputs(inputs);
   assert_true(outputs.get_dimensions_number() == 2, LOG);
   assert_true(outputs == -1.0, LOG);

   // Test

   inputs[0] = 2.0;
   outputs = bounding_layer.calculate_outputs(inputs);
   assert_true(outputs.get_dimensions_number() == 2, LOG);
   assert_true(outputs == 1.0, LOG);

}


void BoundingLayerTest::test_to_XML()
{
   cout << "test_to_XML\n";

   BoundingLayer bounding_layer;

   tinyxml2::XMLDocument* document;

   // Test

   document = bounding_layer.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;

}


void BoundingLayerTest::test_from_XML()
{
   cout << "test_from_XML\n";

   BoundingLayer bounding_layer;

   tinyxml2::XMLDocument* blep;

   // Test

   blep = bounding_layer.to_XML();

   bounding_layer.from_XML(*blep);

}


void BoundingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";


}


void BoundingLayerTest::run_test_case()
{
   cout << "Running bounding layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Bounding layer architecture

   test_get_neurons_number();

   // Variables bounds

   test_get_lower_bounds();
   test_get_lower_bound();

   test_get_upper_bounds();
   test_get_upper_bound();

   test_get_bounds();

   test_get_type();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Variables bounds

   test_set_lower_bounds();
   test_set_lower_bound();

   test_set_upper_bounds();
   test_set_upper_bound();

   test_set_bounds();

   // Display messages

   test_set_display();

   // Lower and upper bounds

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of bounding layer test case.\n";
}


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
