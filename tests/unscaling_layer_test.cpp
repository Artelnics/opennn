//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   T E S T   C L A S S                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "unscaling_layer_test.h"


UnscalingLayerTest::UnscalingLayerTest() : UnitTesting()
{
}


UnscalingLayerTest::~UnscalingLayerTest()
{
}


void UnscalingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

}


void UnscalingLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void UnscalingLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

//   UnscalingLayer mlp_1;
//   UnscalingLayer mlp_2 = mlp_1;

//   assert_true(mlp_2.get_neurons_number() == 0, LOG);

}


void UnscalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   UnscalingLayer ul;

   assert_true(ul.get_neurons_number() == 0, LOG);
}


void UnscalingLayerTest::test_set()
{
   cout << "test_set\n";
}


void UnscalingLayerTest::test_set_default()
{
   cout << "test_set_default\n";
}


void UnscalingLayerTest::test_get_means()
{
   cout << "test_get_means\n";

   UnscalingLayer ul;

//   assert_true(ul.get_means() == 0, LOG);
}


void UnscalingLayerTest::test_get_standard_deviations()
{
   cout << "test_get_standard_deviations\n";

   UnscalingLayer ul;

//   assert_true(ul.get_standard_deviations() == 0, LOG);
}


void UnscalingLayerTest::test_get_mean()
{
   cout << "test_get_mean\n";
}


void UnscalingLayerTest::test_get_standard_deviation()
{
   cout << "test_get_standard_deviation\n";
}


void UnscalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";
}


void UnscalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";
}


void UnscalingLayerTest::test_get_minimum()
{
   cout << "test_get_minimum\n";
}


void UnscalingLayerTest::test_get_maximum()
{
   cout << "test_get_maximum\n";
}


void UnscalingLayerTest::test_get_descriptives()
{
   cout << "test_get_descriptives\n";

   UnscalingLayer ul;

   Vector<Descriptives> descriptives;

   // Test

   descriptives = ul.get_descriptives();

   assert_true(descriptives.size() == 0, LOG);
}


void UnscalingLayerTest::test_get_display()
{
   cout << "test_get_display\n";
}


void UnscalingLayerTest::test_set_unscaling_method()
{
   cout << "test_set_unscaling_method\n";
}


void UnscalingLayerTest::test_set_means()
{
   cout << "test_set_means\n";
}


void UnscalingLayerTest::test_set_standard_deviations()
{
   cout << "test_set_standard_deviations\n";
}


void UnscalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";
}


void UnscalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";
}


void UnscalingLayerTest::test_set_minimums()
{
   cout << "test_set_minimums\n";
}


void UnscalingLayerTest::test_set_maximums()
{
   cout << "test_set_outputs_maximum\n";
}


void UnscalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";
}


void UnscalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";
}


void UnscalingLayerTest::test_set_statistics()
{
   cout << "test_set_statistics\n";
}


void UnscalingLayerTest::test_set_display()
{
   cout << "test_set_display\n";
}


void UnscalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   UnscalingLayer ul(1);

   Vector<double> inputs(1);

   ul.set_display(false);

   // Test

   ul.set_unscaling_method(UnscalingLayer::MinimumMaximum);

   inputs[0] = 0.0;

   //assert_true(ul.calculate_outputs(inputs.to_row_matrix()) == inputs, LOG);

   // Test

   ul.set_unscaling_method(UnscalingLayer::MeanStandardDeviation);

   inputs[0] = 0.0;

   //assert_true(ul.calculate_outputs(inputs.to_row_matrix()) == inputs, LOG);
}


void UnscalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   UnscalingLayer ul;

   Vector<string> inputs_names;
   Vector<string> outputs_names;

   string expression;

   // Test

   ul.set(1);
   inputs_names.set(1, "x");
   outputs_names.set(1, "y");

   expression = ul.write_expression(inputs_names, outputs_names);

   assert_true(expression.empty() == false, LOG);
}


void UnscalingLayerTest::test_get_unscaling_method()
{
   cout << "test_get_unscaling_method\n";

   UnscalingLayer ul;

   // Test

   ul.set_unscaling_method(UnscalingLayer::MeanStandardDeviation);

   assert_true(ul.get_unscaling_method() == UnscalingLayer::MeanStandardDeviation, LOG);

   // Test

   ul.set_unscaling_method(UnscalingLayer::MinimumMaximum);

   assert_true(ul.get_unscaling_method() == UnscalingLayer::MinimumMaximum, LOG);
}


void UnscalingLayerTest::test_get_unscaling_method_name()
{
   cout << "test_get_outputs_method_name\n";
}


void UnscalingLayerTest::test_to_XML()
{
   cout << "test_to_XML\n";

   UnscalingLayer  ul;

   tinyxml2::XMLDocument* uld;
   
   // Test

   uld = ul.to_XML();

   assert_true(uld != nullptr, LOG);

   delete uld;
}


void UnscalingLayerTest::test_from_XML()
{
   cout << "test_from_XML\n";

   UnscalingLayer  ul;

   tinyxml2::XMLDocument* uld;
   
   // Test

   uld = ul.to_XML();

   ul.from_XML(*uld);
 
   delete uld;
}


void UnscalingLayerTest::run_test_case()
{
   cout << "Running unscaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Unscaling layer architecture

   test_get_neurons_number();

   // Output variables descriptives

   test_get_minimums();
   test_get_minimum();

   test_get_maximums();
   test_get_maximum();

   test_get_means();
   test_get_mean();

   test_get_standard_deviations();
   test_get_standard_deviation();

   // Variables descriptives

   test_get_descriptives();

   // Variables scaling and unscaling

   test_get_unscaling_method();
   test_get_unscaling_method_name();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Output variables descriptives

   test_set_minimums();
   test_set_minimum();

   test_set_maximums();
   test_set_maximum();

   test_set_means();
   test_set_mean();

   test_set_standard_deviations();
   test_set_standard_deviation();

   // Variables descriptives

   test_set_statistics();

   // Display messages

   test_set_display();

   // Output methods

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of unscaling layer test case.\n";
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
