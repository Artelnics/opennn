//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   T E S T   C L A S S                       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_test.h"


ScalingLayerTest::ScalingLayerTest() : UnitTesting()
{
}


ScalingLayerTest::~ScalingLayerTest()
{
}


void ScalingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   ScalingLayer sl1;

   assert_true(sl1.get_type() == Layer::Scaling, LOG);
   assert_true(sl1.get_descriptives().size() == 0, LOG);

   ScalingLayer sl2(3);

   assert_true(sl2.get_descriptives().size() == 3, LOG);
   assert_true(sl2.get_scaling_methods().size() == 3, LOG);

   Vector<Descriptives> descriptives;

   descriptives.set(2);

   ScalingLayer sl3(descriptives);

   assert_true(sl3.get_descriptives().size() == 2, LOG);

   ScalingLayer sl4(sl1);

   assert_true(sl4.get_type() == Layer::Scaling, LOG);
}


void ScalingLayerTest::test_destructor()
{
   cout << "test_destructor\n";

}


void ScalingLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

}


void ScalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   ScalingLayer sl1;

   assert_true(sl1.get_neurons_number() == 0, LOG);

   sl1.set(3);

   Vector<Descriptives> descriptives(3);

   sl1.set_descriptives(descriptives);

   assert_true(sl1.get_neurons_number() == 3, LOG);
}


void ScalingLayerTest::test_set()
{
   cout << "test_set\n";
}


void ScalingLayerTest::test_set_default()
{
   cout << "test_set_default\n";
}


void ScalingLayerTest::test_get_means()
{
   cout << "test_get_means\n";

   ScalingLayer sl;

   Vector<double> means;

   // Test

   sl.set();

   assert_true(sl.get_means().empty(), LOG);

   // Test

   sl.set(1);
   sl.set_mean(0, 2.0);

   means = sl.get_means();
 
   assert_true(means.size() == 1, LOG);
   assert_true(means == 2.0, LOG);
}


void ScalingLayerTest::test_get_standard_deviations()
{
    cout << "test_get_standard_deviations\n";

    ScalingLayer sl;

    assert_true(sl.get_standard_deviations().empty(), LOG);

    sl.set(1);

    sl.set_standard_deviation(0, 3.0);

    assert_true(sl.get_standard_deviations() == 3.0, LOG);
}


void ScalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";

   ScalingLayer sl1;

   assert_true(sl1.get_minimums().empty(), LOG);

   sl1.set(1);

   sl1.set_minimum(0, 2.0);

   assert_true(sl1.get_minimums() == 2.0, LOG);
}


void ScalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";

   ScalingLayer sl1;

   assert_true(sl1.get_maximums().empty(), LOG);

   sl1.set(1);

   sl1.set_maximum(0, 2.0);

   assert_true(sl1.get_maximums() == 2.0, LOG);
}


void ScalingLayerTest::test_get_display_inputs_warning()
{
   cout << "test_get_display_inputs_warning\n";
}


void ScalingLayerTest::test_get_display()
{
   cout << "test_get_display\n";

   ScalingLayer sl;

   assert_true(sl.get_display(), LOG);
}


void ScalingLayerTest::test_set_scaling_method()
{
   cout << "test_set_scaling_method\n";
}


void ScalingLayerTest::test_set_means()
{
   cout << "test_set_means\n";
}


void ScalingLayerTest::test_set_standard_deviations()
{
   cout << "test_set_standard_deviations\n";
}


void ScalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";
}


void ScalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";
}


void ScalingLayerTest::test_set_minimums()
{
   cout << "test_set_minimums\n";
}


void ScalingLayerTest::test_set_maximums()
{
   cout << "test_set_maximums\n";
}


void ScalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";
}


void ScalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";
}


void ScalingLayerTest::test_set_statistics()
{
   cout << "test_set_statistics\n";
}


void ScalingLayerTest::test_set_display_inputs_warning()
{
   cout << "test_set_display_inputs_warning\n";
}


void ScalingLayerTest::test_set_display()
{
   cout << "test_set_display\n";
}


void ScalingLayerTest::test_check_range()
{
   cout << "test_check_range\n";

   ScalingLayer sl;
   Vector<double> inputs;

   // Test

   sl.set(1);

   inputs.set(1, 0.0);
   sl.check_range(inputs);

}


void ScalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   ScalingLayer scaling_layer;
   
   Tensor<double> inputs;

   scaling_layer.set_display(false);

   // Test

   scaling_layer.set_scaling_methods(ScalingLayer::MinimumMaximum);

   scaling_layer.set(1);

   inputs.set({1,1}, 0.0);
 
   assert_true(scaling_layer.calculate_outputs(inputs) == inputs, LOG);

   // Test

   scaling_layer.set_scaling_methods(ScalingLayer::MeanStandardDeviation);
 
   scaling_layer.set(1);

   inputs.set({1,1}, 0.0);

   assert_true(scaling_layer.calculate_outputs(inputs) == inputs, LOG);
}


void ScalingLayerTest::test_calculate_minimum_maximum_output()
{
   cout << "test_calculate_minimum_maximum_output\n";
}


void ScalingLayerTest::test_calculate_mean_standard_deviation_output()
{
   cout << "test_calculate_mean_standard_deviation_output\n";
}


void ScalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   ScalingLayer sl;

   Vector<string> inputs_names;
   Vector<string> outputs_names;

   string expression;

   // Test

   sl.set(1);
   inputs_names.set(1, "x");
   outputs_names.set(1, "y");

   expression = sl.write_expression(inputs_names, outputs_names);

   assert_true(expression.empty() == false, LOG);

}


void ScalingLayerTest::test_get_scaling_method()
{
   cout << "test_get_scaling_method\n";

   ScalingLayer sl(1);

   // Test

   sl.set_scaling_methods(ScalingLayer::MeanStandardDeviation);

   assert_true(sl.get_scaling_methods()[0] == ScalingLayer::MeanStandardDeviation, LOG);

   // Test

   sl.set_scaling_methods(ScalingLayer::MinimumMaximum);

   assert_true(sl.get_scaling_methods()[0] == ScalingLayer::MinimumMaximum, LOG);
}


void ScalingLayerTest::test_get_scaling_method_name()
{
   cout << "test_get_scaling_method_name\n";
}


void ScalingLayerTest::test_to_XML()
{
   cout << "test_to_XML\n";

   ScalingLayer  sl;

   tinyxml2::XMLDocument* sld;
   
   // Test

   sld = sl.to_XML();

   assert_true(sld != nullptr, LOG);

   delete sld;
}


void ScalingLayerTest::test_from_XML()
{
   cout << "test_from_XML\n";

   ScalingLayer  sl;

   tinyxml2::XMLDocument* sld;
   
   // Test

   sld = sl.to_XML();

   sl.from_XML(*sld);

   delete sld;
}


void ScalingLayerTest::run_test_case()
{
   cout << "Running scaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Scaling layer architecture

   test_get_neurons_number();

   // Input variables descriptives

   test_get_minimums();

   test_get_maximums();

   test_get_means();

   test_get_standard_deviations();

   // Variables scaling and unscaling

   test_get_scaling_method();
   test_get_scaling_method_name();

   // Display warning 

   test_get_display_inputs_warning();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Input variables descriptives

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

   // Variables scaling and unscaling

   test_set_scaling_method();

   // Display inputs warning

   test_set_display_inputs_warning();

   // Display messages

   test_set_display();

   // Input range

   test_check_range();

   // Scaling and unscaling

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of scaling layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
//
// This library sl free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library sl distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
