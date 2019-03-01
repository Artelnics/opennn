/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   U N S C A L I N G   L A Y E R   T E S T   C L A S S   H E A D E R                                          */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __UNSCALINGLAYERTEST_H__
#define __UNSCALINGLAYERTEST_H__

// Unit testing includes

#include "unit_testing.h"


using namespace OpenNN;


class UnscalingLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit UnscalingLayerTest();


   // DESTRUCTOR

   virtual ~UnscalingLayerTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   // Multilayer perceptron architecture 

   void test_get_unscaling_neurons_number();

   // Statistics

   void test_get_minimums();
   void test_get_minimum();

   void test_get_maximums();
   void test_get_maximum();

   void test_get_means();
   void test_get_mean();

   void test_get_standard_deviations();
   void test_get_standard_deviation();

   void test_get_statistics();

   // Variables scaling and unscaling

   void test_get_unscaling_method();
   void test_get_unscaling_method_name();

   // Display warning 

   void test_get_display_warning();

   // Display messages

   void test_get_display();

   // SET METHODS

   void test_set();
   void test_set_default();

   // Output variables statistics

   void test_set_means();
   void test_set_mean();

   void test_set_standard_deviations();
   void test_set_standard_deviation();

   void test_set_minimums();
   void test_set_minimum();

   void test_set_maximums();
   void test_set_maximum();

   // Statistics

   void test_set_statistics();

   // Variables scaling and unscaling

   void test_set_unscaling_method();

   // Display messages

   void test_set_display_outputs_warning();
   void test_set_display();

   // Initialization methods

   void test_initialize_random();

   // Input range

   void test_check_outputs_range();

   // Outputs unscaling

   void test_calculate_outputs();
   void test_calculate_derivatives();
   void test_calculate_second_derivatives();

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
