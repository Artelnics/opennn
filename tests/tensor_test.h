//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TENSORTEST_H
#define TENSORTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace std;
using namespace OpenNN;

class TensorTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit TensorTest();

   virtual ~TensorTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Arithmetic operators

   void test_sum_operator();
   void test_rest_operator();
   void test_multiplication_operator();
   void test_division_operator();

   // Assignment operators methods

   void test_assignment_operator();   

   // Reference operator

   void test_reference_operator();

   // Operation an assignment operators
   
   void test_sum_assignment_operator();
   void test_rest_assignment_operator();
   void test_multiplication_assignment_operator();
   void test_division_assignment_operator();

   // Equality and relational operators

   void test_equal_to_operator();
   void test_not_equal_to_operator();
   void test_greater_than_operator();
   void test_less_than_operator();
   void test_greater_than_or_equal_to_operator();
   void test_less_than_or_equal_to_operator();

   // Output operator

   void test_output_operator();

   // File operation

    void test_tuck_in();

   // Get methods

   void test_get_size();
   void test_get_display();

   void test_get_subvector();
   void test_get_subvector_random();

   // Set

   void test_set();
   void test_set_display();


  // Resize methods

   void test_resize();

   void test_insert();
   void test_take_out();

   void test_insert_element();
   void test_split_element();

   void test_delete_index();
   void test_delete_indices();

   void test_delete_value();
   void test_delete_values();

   void test_assemble();

   void test_get_difference();
   void test_get_union();
   void test_get_intersection();

   void test_get_unique_elements();

   void test_count_unique();

   void test_remove_element();

   void test_get_assembly();
   void test_difference();

   void test_intersection();

   void test_get_unique();

   void test_calculate_top_string();
   void test_calculate_top_number();

   void test_print_top_string();

   void test_to_float_vector();

   // Initialization

   void test_initialize();
   void test_initialize_first();
   void test_initialize_sequential();
   void test_randomize_uniform();
   void test_randomize_normal();
   void test_randomize_binary();

   void test_fill_from();

   // Checking methods

   void test_contains();
   void test_contains_greater_than();

   void test_is_in();

   void test_is_constant();
   void test_is_constant_string();

   void test_is_crescent();
   void test_is_decrescent();

   void test_has_same_elements();

   void test_is_binary();
   void test_is_binary_0_1();

   void test_is_positive();
   void test_is_negative();

   void test_is_integer();

   void test_check_period();

   void test_get_reverse();

   // Replace methods

   void test_replace_value();

   // Mathematical operations

   void test_dot_vector();
   void test_dot_matrix();

   void test_calculate_sum();
   void test_calculate_partial_sum();
   void test_calculate_product();

   void test_index();


   // Scaling and unscaling

   void test_scale_minimum_maximum();
   void test_scale_mean_standard_deviation();

   void test_unscale_minimum_maximum();
   void test_unscale_mean_standard_deviation();

   // Parsing methods

   void test_parse();

   // Serizalization methods

   void test_load();
   void test_save();

   //Count methods

   void test_count_equal_to();
   void test_count_not_equal_to();

   void test_count_NAN();

   void test_count_negative();
   void test_count_positive();
   void test_count_integers();

   void test_filter_equal_to();
   void test_filter_not_equal_to();

   void test_get_positive_elements();
   void test_get_negative_elements();
   void test_get_between_indices();

   void test_get_indices_equal_to();
   void test_get_indices_not_equal_to();

   void test_filter_minimum_maximum();

   void test_get_indices_less_than();
   void test_get_indices_greater_than();

   void test_count_greater_than();
   void test_count_greater_equal_to();
   void test_count_less_than();
   void test_count_less_equal_to();
   void test_count_between();

   void test_get_indices_that_contains();
   void test_get_indices_less_equal_to();
   void test_get_indices_greater_equal_to();

   void test_perform_Box_Cox_transformation();

   void test_calculate_percentage();

   void test_count_contains();

   void test_merge();

   //Descriptive methods

   void test_get_first_index();
   void test_calculate_cumulative_index();


   // Ranks methods

   void test_sort_ascending_indices();
   void test_sort_ascending_values();
   void test_sort_descending_indices();
   void test_sort_descending_values();

   void test_calculate_lower_indices();
   void test_calculate_lower_values();


   void test_calculate_less_rank();
   void test_calculate_greater_rank();
   void test_calculate_sort_rank();

   // Filter methods

   void test_filter_positive();
   void test_filter_negative();

   void test_get_first();
   void test_get_last();
   void test_get_before_last();

   void test_delete_first();
   void test_delete_last();

   void test_get_integer_elements();
   void test_get_integers();

   void test_count_dates();


   //Matrix

   void test_matrix_without_NAN();


   // Unit testing methods

   void run_test_case();

};

#endif


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
