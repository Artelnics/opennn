/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S C A L I N G   L A Y E R   T E S T   C L A S S                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "scaling_layer_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

ScalingLayerTest::ScalingLayerTest(void) : UnitTesting()
{
}


// DESTRUCTOR

ScalingLayerTest::~ScalingLayerTest(void)
{
}


// METHODS

void ScalingLayerTest::test_constructor(void)
{
   message += "test_constructor\n";

   // Copy constructor

}


void ScalingLayerTest::test_destructor(void)
{
   message += "test_destructor\n";

}


void ScalingLayerTest::test_assignment_operator(void)
{
   message += "test_assignment_operator\n";

}


void ScalingLayerTest::test_get_scaling_neurons_number(void)
{
   message += "test_get_scaling_neurons_number\n";
}


void ScalingLayerTest::test_set(void)
{
   message += "test_set\n";
}


void ScalingLayerTest::test_set_default(void)
{
   message += "test_set_default\n";
}


void ScalingLayerTest::test_arrange_means(void)
{
   message += "test_arrange_means\n";

   ScalingLayer sl;

   Vector<double> means;

   // Test

   sl.set();

   assert_true(sl.arrange_means().empty(), LOG);

   // Test

   sl.set(1);
   sl.set_mean(0, 2.0);

   means = sl.arrange_means();
 
   assert_true(means.size() == 1, LOG);
   assert_true(means == 2.0, LOG);

}


void ScalingLayerTest::test_arrange_standard_deviations(void)
{
   message += "test_get_standard_deviations\n";

   ScalingLayer sl;

   assert_true(sl.arrange_standard_deviations() == 0, LOG);

}


void ScalingLayerTest::test_get_mean(void)
{
   message += "test_get_mean\n";
}


void ScalingLayerTest::test_get_standard_deviation(void)
{
   message += "test_get_standard_deviation\n";
}


void ScalingLayerTest::test_arrange_minimums(void)
{
   message += "test_arrange_minimums\n";
}


void ScalingLayerTest::test_arrange_maximums(void)
{
   message += "test_get_maximums\n";
}


void ScalingLayerTest::test_get_minimum(void)
{
   message += "test_get_minimum\n";
}


void ScalingLayerTest::test_get_maximum(void)
{
   message += "test_get_maximum\n";
}


void ScalingLayerTest::test_get_display_inputs_warning(void)
{
   message += "test_get_display_inputs_warning\n";
}


void ScalingLayerTest::test_get_display(void)
{
   message += "test_get_display\n";
}


void ScalingLayerTest::test_set_scaling_method(void)
{
   message += "test_set_scaling_method\n";
}


void ScalingLayerTest::test_set_means(void)
{
   message += "test_set_means\n";
}


void ScalingLayerTest::test_set_standard_deviations(void)
{
   message += "test_set_standard_deviations\n";
}


void ScalingLayerTest::test_set_means_standard_deviations(void)
{
   message += "test_set_means_standard_deviation\n";
}


void ScalingLayerTest::test_set_mean(void)
{
   message += "test_set_mean\n";
}


void ScalingLayerTest::test_set_standard_deviation(void)
{
   message += "test_set_standard_deviation\n";
}


void ScalingLayerTest::test_set_minimums(void)
{
   message += "test_set_minimums\n";
}


void ScalingLayerTest::test_set_maximums(void)
{
   message += "test_set_maximums\n";
}


void ScalingLayerTest::test_set_minimum(void)
{
   message += "test_set_minimum\n";
}


void ScalingLayerTest::test_set_maximum(void)
{
   message += "test_set_maximum\n";
}


void ScalingLayerTest::test_set_minimums_maximums(void)
{
   message += "test_set_means_standard_deviation\n";
}


void ScalingLayerTest::test_set_statistics(void)
{
   message += "test_set_statistics\n";
}


void ScalingLayerTest::test_set_display_inputs_warning(void)
{
   message += "test_set_display_inputs_warning\n";
}


void ScalingLayerTest::test_set_display(void)
{
   message += "test_set_display\n";
}


void ScalingLayerTest::test_initialize_random(void)
{
   message += "test_initialize_random\n";

   ScalingLayer sl;

   // Test

   sl.initialize_random();
}


void ScalingLayerTest::test_check_range(void)
{
   message += "test_check_range\n";

   ScalingLayer sl;
   Vector<double> inputs;

   // Test

   sl.set(1);

   inputs.set(1, 0.0);
   sl.check_range(inputs);

}


void ScalingLayerTest::test_calculate_outputs(void)
{
   message += "test_calculate_outputs\n";

   ScalingLayer sl;
   
   Vector<double> inputs;

   sl.set_display(false);

   // Test

   sl.set_scaling_method(ScalingLayer::MinimumMaximum);

   sl.set(1);

   inputs.set(1, 0.0);
 
   assert_true(sl.calculate_outputs(inputs) == inputs, LOG);

   // Test

   sl.set_scaling_method(ScalingLayer::MeanStandardDeviation);
 
   sl.set(1);

   inputs.set(1, 0.0);

   assert_true(sl.calculate_outputs(inputs) == inputs, LOG);

}


void ScalingLayerTest::test_calculate_derivative(void)
{
   message += "test_calculate_derivative\n";
}


void ScalingLayerTest::test_calculate_second_derivative(void)
{
   message += "test_calculate_second_derivative\n";
}


void ScalingLayerTest::test_calculate_minimum_maximum_output(void)
{
   message += "test_calculate_minimum_maximum_output\n";
}


void ScalingLayerTest::test_calculate_minimum_maximum_derivative(void)
{
   message += "test_calculate_minimum_maximum_derivative\n";
}


void ScalingLayerTest::test_calculate_minimum_maximum_second_derivative(void)
{
   message += "test_calculate_minimum_maximum_second_derivative\n";
}


void ScalingLayerTest::test_calculate_mean_standard_deviation_output(void)
{
   message += "test_calculate_mean_standard_deviation_output\n";
}


void ScalingLayerTest::test_calculate_mean_standard_deviation_derivative(void)
{
   message += "test_calculate_mean_standard_deviation_derivative\n";
}


void ScalingLayerTest::test_calculate_mean_standard_deviation_second_derivative(void)
{
   message += "test_calculate_mean_standard_deviation_second_derivative\n";
}


void ScalingLayerTest::test_write_expression(void)
{
   message += "test_write_expression\n";

   ScalingLayer sl;

   Vector<std::string> inputs_name;
   Vector<std::string> outputs_name;

   std::string expression;

   // Test

   sl.set(1);
   inputs_name.set(1, "x");
   outputs_name.set(1, "y");

   expression = sl.write_expression(inputs_name, outputs_name);

   assert_true(expression.empty() == false, LOG);

}


void ScalingLayerTest::test_get_scaling_method(void)
{
   message += "test_get_scaling_method\n";

   ScalingLayer sl;

   // Test

   sl.set_scaling_method(ScalingLayer::MeanStandardDeviation);

   assert_true(sl.get_scaling_method() == ScalingLayer::MeanStandardDeviation, LOG);

   // Test

   sl.set_scaling_method(ScalingLayer::MinimumMaximum);

   assert_true(sl.get_scaling_method() == ScalingLayer::MinimumMaximum, LOG);
}


void ScalingLayerTest::test_get_scaling_method_name(void)
{
   message += "test_get_scaling_method_name\n";
}


void ScalingLayerTest::test_to_XML(void)
{
   message += "test_to_XML\n";

   ScalingLayer  sl;

   tinyxml2::XMLDocument* sld;
   
   // Test

   sld = sl.to_XML();

   assert_true(sld != NULL, LOG);

   delete sld;
}


void ScalingLayerTest::test_from_XML(void)
{
   message += "test_from_XML\n";

   ScalingLayer  sl;

   tinyxml2::XMLDocument* sld;
   
   // Test

   sld = sl.to_XML();

   sl.from_XML(*sld);

   delete sld;
}


void ScalingLayerTest::run_test_case(void)
{
   message += "Running scaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Scaling layer architecture

   test_get_scaling_neurons_number();

   // Input variables statistics

   test_arrange_minimums();
   test_get_minimum();

   test_arrange_maximums();
   test_get_maximum();

   test_arrange_means();
   test_get_mean();

   test_arrange_standard_deviations();
   test_get_standard_deviation();

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

   // Input variables statistics

   test_set_minimums();
   test_set_minimum();

   test_set_maximums();
   test_set_maximum();

   test_set_means();
   test_set_mean();

   test_set_standard_deviations();
   test_set_standard_deviation();

   test_set_means_standard_deviations();
   test_set_minimums_maximums();

   // Variables statistics

   test_set_statistics();

   // Variables scaling and unscaling

   test_set_scaling_method();

   // Display inputs warning

   test_set_display_inputs_warning();

   // Display messages

   test_set_display();

   // Neural network initialization methods

   test_initialize_random();

   // Input range

   test_check_range();

   // Scaling and unscaling

   test_calculate_outputs();
   test_calculate_derivative();
   test_calculate_second_derivative();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of scaling layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
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
