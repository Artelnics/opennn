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

   explicit IndependentParametersTest(void);


   // DESTRUCTOR

   virtual ~IndependentParametersTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   // Independent parameters

   void test_count_parameters_number(void);

   void test_get_parameters(void);   
   void test_get_parameter(void);   

   // Independent parameters scaling and unscaling

   void test_get_parameters_scaling_method(void);
   void test_get_parameters_scaling_method_name(void);

   // Independent parameters information

   void test_get_parameters_name(void);
   void test_get_parameter_name(void);

   void test_get_parameters_units(void);
   void test_get_parameter_units(void);

   void test_get_parameters_description(void);
   void test_get_parameter_description(void);

   void test_get_parameters_information(void);

   // Independent parameters statistics

   void test_get_parameters_minimum(void);
   void test_get_parameter_minimum(void);

   void test_get_parameters_maximum(void);
   void test_get_parameter_maximum(void);

   void test_get_parameters_mean(void);
   void test_get_parameter_mean(void);

   void test_get_parameters_standard_deviation(void);
   void test_get_parameter_standard_deviation(void);

   void test_get_parameters_mean_standard_deviation(void);

   void test_get_parameters_minimum_maximum(void);

   void test_get_parameters_statistics(void);

   // Independent parameters bounds

   void test_get_parameters_lower_bound(void);
   void test_get_parameter_lower_bound(void);

   void test_get_parameters_upper_bound(void);
   void test_get_parameter_upper_bound(void);

   void test_get_parameters_bounds(void);

  // Display messages

   void test_get_display(void);

   // SET METHODS

   void test_set(void);
   void test_set_default(void);

   // Independent parameters

   void test_set_parameters_number(void);

   void test_set_parameters(void);
   void test_set_parameter(void);

   // Independent parameters information

   void test_set_parameters_name(void);
   void test_set_parameter_name(void);

   void test_set_parameters_units(void);
   void test_set_parameter_units(void);

   void test_set_parameters_description(void);
   void test_set_parameter_description(void);

   // Independent parameters statistics

   void test_set_parameters_mean(void);
   void test_set_parameter_mean(void);

   void test_set_parameters_standard_deviation(void);
   void test_set_parameter_standard_deviation(void);
   
   void test_set_parameters_minimum(void);
   void test_set_parameter_minimum(void);

   void test_set_parameters_maximum(void);
   void test_set_parameter_maximum(void);

   void test_set_parameters_mean_standard_deviation(void);
   void test_set_parameters_minimum_maximum(void);

   void test_set_parameters_statistics(void);

   // Independent parameters scaling and unscaling

   void test_set_parameters_scaling_method(void);

   // Independent parameters bounds

   void test_set_parameters_lower_bound(void);
   void test_set_parameter_lower_bound(void);

   void test_set_parameters_upper_bound(void);
   void test_set_parameter_upper_bound(void);

   void test_set_parameters_bounds(void);

   // Display messages

   void test_set_display(void);

   // Neural network initialization methods

   void test_initialize_random(void);

   // Independent parameters initialization methods

   void test_initialize_parameters(void);

   void test_randomize_parameters_uniform(void);
   void test_randomize_parameters_normal(void);

   // Independent parameters methods

   void test_calculate_scaled_parameters(void);
   void test_unscale_parameters(void);

   void test_bound_parameters(void);
   void test_bound_parameter(void);

   // Serialization methods

   void test_to_XML(void);
   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);
};


#endif



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
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
