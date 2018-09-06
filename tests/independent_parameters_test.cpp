/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N D E P E N D E N T   P A R A M E T E R S   T E S T   C L A S S                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "independent_parameters_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

IndependentParametersTest::IndependentParametersTest() : UnitTesting()
{
}


// DESTRUCTOR

IndependentParametersTest::~IndependentParametersTest()
{
}


// METHODS


void IndependentParametersTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   IndependentParameters mlp1;

   assert_true(mlp1.get_parameters_number() == 0, LOG);

   // Independent parameters constructor

   IndependentParameters mlp4(1);

   assert_true(mlp4.get_parameters_number() == 1, LOG);

   // Copy constructor

}


void IndependentParametersTest::test_destructor()
{
   message += "test_destructor\n";
}


void IndependentParametersTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   IndependentParameters mlp_1;
   IndependentParameters mlp_2 = mlp_1;

   assert_true(mlp_2.get_parameters_number() == 0, LOG);
}


void IndependentParametersTest::test_set()
{
   message += "test_set\n";
}


void IndependentParametersTest::test_set_default()
{
   message += "test_set_default\n";
}


void IndependentParametersTest::test_count_parameters_number()
{
   message += "test_count_parameters_number\n";
}


void IndependentParametersTest::test_get_parameters()   
{
   message += "test_get_parameters\n";
}


void IndependentParametersTest::test_get_parameter()   
{
   message += "test_get_parameter\n";
}


void IndependentParametersTest::test_get_parameters_name()
{
   message += "test_get_parameters_name\n";
}


void IndependentParametersTest::test_get_parameter_name()
{
   message += "test_get_parameter_name\n";
}


void IndependentParametersTest::test_get_parameters_units()
{
   message += "test_get_parameters_units\n";
}


void IndependentParametersTest::test_get_parameter_units()
{
   message += "test_get_parameter_units\n";
}


void IndependentParametersTest::test_get_parameters_information()
{
   message += "test_get_parameters_information\n";
}


void IndependentParametersTest::test_get_parameters_description()
{
   message += "test_get_parameters_description\n";
}


void IndependentParametersTest::test_get_parameter_description()
{
   message += "test_get_parameter_description\n";
}


void IndependentParametersTest::test_get_parameters_mean()
{
   message += "test_get_parameters_mean\n";
}


void IndependentParametersTest::test_get_parameters_standard_deviation()
{
   message += "test_get_parameters_standard_deviation\n";
}


void IndependentParametersTest::test_get_parameters_mean_standard_deviation()
{
   message += "test_get_parameters_mean_standard_deviation\n";
}


void IndependentParametersTest::test_get_parameter_mean()
{
   message += "test_get_parameter_mean\n";
}


void IndependentParametersTest::test_get_parameter_standard_deviation()
{
   message += "test_get_parameter_standard_deviation\n";
}


void IndependentParametersTest::test_get_parameters_minimum()
{
   message += "test_get_parameters_minimum\n";
}


void IndependentParametersTest::test_get_parameters_maximum()
{
   message += "test_get_parameters_maximum\n";
}


void IndependentParametersTest::test_get_parameters_minimum_maximum()
{
   message += "test_get_parameters_minimum_maximum\n";
}


void IndependentParametersTest::test_get_parameter_minimum()
{
   message += "test_get_parameter_minimum\n";
}


void IndependentParametersTest::test_get_parameter_maximum()
{
   message += "test_get_parameter_maximum\n";
}


void IndependentParametersTest::test_get_parameters_statistics()
{
   message += "test_get_parameters_statistics\n";

   IndependentParameters ip;

   // Test

//   Vector< Statistics<double> > statistics = ip.arrange_statistics();

//   assert_true(statistics.size() == 4, LOG);
}


void IndependentParametersTest::test_get_parameters_lower_bound()
{
   message += "test_get_parameters_lower_bound\n";
}


void IndependentParametersTest::test_get_parameters_upper_bound()
{
   message += "test_get_parameters_upper_bound\n";
}


void IndependentParametersTest::test_get_parameter_lower_bound()
{
   message += "test_get_parameter_lower_bound\n";
}


void IndependentParametersTest::test_get_parameter_upper_bound()
{
   message += "test_get_parameter_upper_bound\n";
}


void IndependentParametersTest::test_get_parameters_bounds()
{
   message += "test_get_parameters_bounds\n";
}


void IndependentParametersTest::test_get_display()
{
   message += "test_get_display\n";
}


void IndependentParametersTest::test_set_parameters_scaling_method()
{
   message += "test_set_parameters_scaling_method\n";
}


void IndependentParametersTest::test_set_parameters_number()
{
   message += "test_set_parameters_number\n";
}


void IndependentParametersTest::test_set_parameters()
{
   message += "test_set_parameters\n";
}


void IndependentParametersTest::test_set_parameter()
{
   message += "test_set_parameter\n";
}


void IndependentParametersTest::test_set_parameters_name()
{
   message += "test_set_parameters_name\n";
}


void IndependentParametersTest::test_set_parameter_name()
{
   message += "test_set_parameter_name\n";
}


void IndependentParametersTest::test_set_parameters_units()
{
   message += "test_set_parameters_units\n";
}


void IndependentParametersTest::test_set_parameter_units()
{
   message += "test_set_parameter_units\n";
}


void IndependentParametersTest::test_set_parameters_description()
{
   message += "test_set_parameters_description\n";
}


void IndependentParametersTest::test_set_parameter_description()
{
   message += "test_set_parameter_description\n";
}


void IndependentParametersTest::test_set_parameters_mean()
{
   message += "test_set_parameters_mean\n";
}


void IndependentParametersTest::test_set_parameters_standard_deviation()
{
   message += "test_set_parameters_standard_deviation\n";
}


void IndependentParametersTest::test_set_parameters_mean_standard_deviation()
{
   message += "test_set_parameters_mean_standard_deviation\n";
}


void IndependentParametersTest::test_set_parameter_mean()
{
   message += "test_set_parameter_mean\n";
}


void IndependentParametersTest::test_set_parameter_standard_deviation()
{
   message += "test_set_parameter_standard_deviation\n";
}


void IndependentParametersTest::test_set_parameters_minimum()
{
   message += "test_set_parameters_minimum\n";
}


void IndependentParametersTest::test_set_parameter_minimum()
{
   message += "test_set_parameter_minimum\n";
}


void IndependentParametersTest::test_set_parameters_maximum()
{
   message += "test_set_parameters_maximum\n";
}


void IndependentParametersTest::test_set_parameter_maximum()
{
   message += "test_set_parameter_maximum\n";
}


void IndependentParametersTest::test_set_parameters_minimum_maximum()
{
   message += "test_set_parameters_minimum_maximum\n";
}


void IndependentParametersTest::test_set_parameters_statistics()
{
   message += "test_set_parameters_statistics\n";
}


void IndependentParametersTest::test_set_parameters_lower_bound()
{
   message += "test_set_parameters_lower_bound\n";
}


void IndependentParametersTest::test_set_parameters_upper_bound()
{
   message += "test_set_parameters_upper_bound\n";
}


void IndependentParametersTest::test_set_parameters_bounds()
{
   message += "test_set_parameters_bounds\n";
}


void IndependentParametersTest::test_set_parameter_lower_bound()
{
   message += "test_set_parameter_lower_bounds\n";
}


void IndependentParametersTest::test_set_parameter_upper_bound()
{
   message += "test_set_parameter_upper_bound\n";
}


void IndependentParametersTest::test_set_display()
{
   message += "test_set_display\n";
}


void IndependentParametersTest::test_initialize_random()
{
   message += "test_initialize_random\n";

   IndependentParameters ip;

   size_t parameters_number;

   // Test

   ip.initialize_random();

   parameters_number = ip.get_parameters_number();

   assert_true(parameters_number >= 1 && parameters_number <= 10, LOG);
}


void IndependentParametersTest::test_initialize_parameters()
{      
   message += "test_initialize_parameters\n";

   IndependentParameters ip;
   Vector<double> independent_parameters;

   ip.set_parameters_number(1);
   ip.initialize_parameters(0.0);
   independent_parameters = ip.get_parameters();
   assert_true(independent_parameters == 0.0, LOG);
}


void IndependentParametersTest::test_randomize_parameters_uniform()
{
   message += "test_randomize_parameters_uniform\n";
   IndependentParameters ip;
   Vector<double> independent_parameters;

   // Test

   ip.set_parameters_number(1);      
   ip.randomize_parameters_uniform();
   independent_parameters = ip.get_parameters();   
   assert_true(independent_parameters >= -1.0, LOG);
   assert_true(independent_parameters <=  1.0, LOG);
}


void IndependentParametersTest::test_randomize_parameters_normal()
{
   message += "test_randomize_parameters_normal\n";
   IndependentParameters ip;
   Vector<double> independent_parameters;

   // Test

   ip.set_parameters_number(1);
   ip.randomize_parameters_normal(1.0, 0.0);
   independent_parameters = ip.get_parameters();
   assert_true(independent_parameters == 1.0, LOG);
}


void IndependentParametersTest::test_calculate_scaled_parameters()
{
   message += "test_calculate_scaled_parameters\n";

   IndependentParameters ip;
   Vector<double> independent_parameters;
   Vector<double> scaled_parameters;

   // Minimum and maximum scaling and unscaling

   ip.set_parameters_number(1);
   ip.initialize_parameters(0.0);

   ip.set_scaling_method(IndependentParameters::MinimumMaximum);

   ip.set_minimum(0, -1.0);
   ip.set_maximum(0,  1.0);

   independent_parameters = ip.get_parameters();

   scaled_parameters = ip.calculate_scaled_parameters();

   assert_true(scaled_parameters == independent_parameters, LOG);      

   // Mean and standard deviation scaling and unscaling

   ip.set_parameters_number(1);
   ip.initialize_parameters(0.0);

   ip.set_scaling_method(IndependentParameters::MeanStandardDeviation);
 
   ip.set_mean(0, 0.0);
   ip.set_standard_deviation(0, 1.0);

   independent_parameters = ip.get_parameters();

   scaled_parameters = ip.calculate_scaled_parameters();

   assert_true(scaled_parameters == independent_parameters, LOG);      

}


void IndependentParametersTest::test_unscale_parameters()
{
   message += "test_unscale_parameters\n";

   IndependentParameters ip;
   Vector<double> scaled_parameters;
   Vector<double> independent_parameters;

   // Minimum and maximum scaling and unscaling

   ip.set_parameters_number(1);
   ip.initialize_parameters(0.0);

   ip.set_scaling_method(IndependentParameters::MinimumMaximum);

   ip.set_minimum(0, -1.0);
   ip.set_maximum(0,  1.0);

   scaled_parameters = ip.calculate_scaled_parameters();

   ip.unscale_parameters(scaled_parameters);

   independent_parameters = ip.get_parameters();

   assert_true(independent_parameters == scaled_parameters, LOG);

   // Mean and standard deviation scaling and unscaling

   ip.set_parameters_number(1);
   ip.initialize_parameters(0.0);

   ip.set_scaling_method(IndependentParameters::MeanStandardDeviation);

   ip.set_mean(0, 0.0);
   ip.set_standard_deviation(0, 1.0);

   scaled_parameters = ip.calculate_scaled_parameters();

   ip.unscale_parameters(scaled_parameters);

   independent_parameters = ip.get_parameters();

   assert_true(independent_parameters == scaled_parameters, LOG);
}


void IndependentParametersTest::test_bound_parameters()
{
   message += "test_bound_parameters\n";

   IndependentParameters ip;

   Vector<double> independent_parameters;
   Vector<double> bounded_parameters;

   // Test

}


void IndependentParametersTest::test_bound_parameter()
{
   message += "test_bound_parameter\n";
}


void IndependentParametersTest::test_get_parameters_scaling_method()
{
   message += "test_get_parameters_scaling_method\n";

   IndependentParameters ip;

   ip.set_scaling_method(IndependentParameters::MeanStandardDeviation);

   assert_true(ip.get_scaling_method() == IndependentParameters::MeanStandardDeviation, LOG);

   ip.set_scaling_method(IndependentParameters::MinimumMaximum);

   assert_true(ip.get_scaling_method() == IndependentParameters::MinimumMaximum, LOG);
}


void IndependentParametersTest::test_get_parameters_scaling_method_name()
{
   message += "test_get_parameters_scaling_method_name\n";
}


void IndependentParametersTest::test_to_XML()
{
   message += "test_to_XML\n";

   IndependentParameters  ip;

   tinyxml2::XMLDocument* element = ip.to_XML();

   assert_true(element != NULL, LOG);
}


void IndependentParametersTest::test_from_XML()
{
   message += "test_from_XML\n";

   IndependentParameters  ip;

   tinyxml2::XMLDocument* document;

   // Test

   document = ip.to_XML();
   
   ip.from_XML(*document);

   delete document;
}


void IndependentParametersTest::run_test_case()
{
   message += "Running independent parameters test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Independent parameters

   test_count_parameters_number();
   
   test_get_parameters();   
   test_get_parameter();   

   // Independent parameters information

   test_get_parameters_name();
   test_get_parameter_name();

   test_get_parameters_units();
   test_get_parameter_units();

   test_get_parameters_description();
   test_get_parameter_description();

   test_get_parameters_information();

   // Independent parameters statistics

   test_get_parameters_minimum();
   test_get_parameter_minimum();

   test_get_parameters_maximum();
   test_get_parameter_maximum();

   test_get_parameters_mean();
   test_get_parameter_mean();

   test_get_parameters_standard_deviation();
   test_get_parameter_standard_deviation();

   test_get_parameters_mean_standard_deviation();
   test_get_parameters_minimum_maximum();

   test_get_parameters_statistics();

   // Independent parameters scaling and unscaling

   test_get_parameters_scaling_method();
   test_get_parameters_scaling_method_name();

   // Independent parameters bounds

   test_get_parameters_lower_bound();
   test_get_parameter_lower_bound();

   test_get_parameters_upper_bound();
   test_get_parameter_upper_bound();

   test_get_parameters_bounds();

   // Parameters methods

   test_count_parameters_number();
   test_get_parameters();   

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Independent parameters

   test_set_parameters_number();
   test_set_parameters();

   test_set_parameter();

   // Independent parameters information

   test_set_parameters_name();
   test_set_parameter_name();

   test_set_parameters_units();
   test_set_parameter_units();

   test_set_parameters_description();
   test_set_parameter_description();

   // Independent parameters statistics

   test_set_parameters_minimum();
   test_set_parameter_minimum();

   test_set_parameters_maximum();
   test_set_parameter_maximum();

   test_set_parameters_mean();
   test_set_parameter_mean();

   test_set_parameters_standard_deviation();
   test_set_parameter_standard_deviation();
   
   test_set_parameters_minimum_maximum();
   test_set_parameters_mean_standard_deviation();

   test_set_parameters_statistics();

   // Independent parameters scaling and unscaling

   test_set_parameters_scaling_method();

   // Independent parameters bounds

   test_set_parameters_lower_bound();
   test_set_parameter_lower_bound();

   test_set_parameters_upper_bound();
   test_set_parameter_upper_bound();

   test_set_parameters_bounds();

   // Parameters methods

   test_set_parameters();

   // Display messages

   test_set_display();

   // Neural network initialization methods

   test_initialize_random();

   // Independent parameters initialization methods

   test_initialize_parameters();
   test_randomize_parameters_uniform();
   test_randomize_parameters_normal();

   // Parameters initialization methods

   test_initialize_parameters();
   test_randomize_parameters_uniform();
   test_randomize_parameters_normal();

   // Independent parameters methods   

   test_calculate_scaled_parameters();
   test_unscale_parameters();

   test_bound_parameters();
   test_bound_parameter();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of independent parameters test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
