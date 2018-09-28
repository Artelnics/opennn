/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N D E P E N D E N T   P A R A M E T E R S   T E S T   C L A S S   H E A D E R                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INDEPENDENTPARAMETERSTEST_H__
#define __INDEPENDENTPARAMETERSTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class IndependentParametersTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit IndependentParametersTest();


   // DESTRUCTOR

   virtual ~IndependentParametersTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   // Independent parameters

   void test_count_parameters_number();

   void test_get_parameters();   
   void test_get_parameter();   

   // Independent parameters scaling and unscaling

   void test_get_parameters_scaling_method();
   void test_get_parameters_scaling_method_name();

   // Independent parameters information

   void test_get_parameters_name();
   void test_get_parameter_name();

   void test_get_parameters_units();
   void test_get_parameter_units();

   void test_get_parameters_description();
   void test_get_parameter_description();

   void test_get_parameters_information();

   // Independent parameters statistics

   void test_get_parameters_minimum();
   void test_get_parameter_minimum();

   void test_get_parameters_maximum();
   void test_get_parameter_maximum();

   void test_get_parameters_mean();
   void test_get_parameter_mean();

   void test_get_parameters_standard_deviation();
   void test_get_parameter_standard_deviation();

   void test_get_parameters_mean_standard_deviation();

   void test_get_parameters_minimum_maximum();

   void test_get_parameters_statistics();

   // Independent parameters bounds

   void test_get_parameters_lower_bound();
   void test_get_parameter_lower_bound();

   void test_get_parameters_upper_bound();
   void test_get_parameter_upper_bound();

   void test_get_parameters_bounds();

  // Display messages

   void test_get_display();

   // SET METHODS

   void test_set();
   void test_set_default();

   // Independent parameters

   void test_set_parameters_number();

   void test_set_parameters();
   void test_set_parameter();

   // Independent parameters information

   void test_set_parameters_name();
   void test_set_parameter_name();

   void test_set_parameters_units();
   void test_set_parameter_units();

   void test_set_parameters_description();
   void test_set_parameter_description();

   // Independent parameters statistics

   void test_set_parameters_mean();
   void test_set_parameter_mean();

   void test_set_parameters_standard_deviation();
   void test_set_parameter_standard_deviation();
   
   void test_set_parameters_minimum();
   void test_set_parameter_minimum();

   void test_set_parameters_maximum();
   void test_set_parameter_maximum();

   void test_set_parameters_mean_standard_deviation();
   void test_set_parameters_minimum_maximum();

   void test_set_parameters_statistics();

   // Independent parameters scaling and unscaling

   void test_set_parameters_scaling_method();

   // Independent parameters bounds

   void test_set_parameters_lower_bound();
   void test_set_parameter_lower_bound();

   void test_set_parameters_upper_bound();
   void test_set_parameter_upper_bound();

   void test_set_parameters_bounds();

   // Display messages

   void test_set_display();

   // Neural network initialization methods

   void test_initialize_random();

   // Independent parameters initialization methods

   void test_initialize_parameters();

   void test_randomize_parameters_uniform();
   void test_randomize_parameters_normal();

   // Independent parameters methods

   void test_calculate_scaled_parameters();
   void test_unscale_parameters();

   void test_bound_parameters();
   void test_bound_parameter();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Unit testing methods

   void run_test_case();
};


#endif



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
