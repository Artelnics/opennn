/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V E C T O R   T E S T   C L A S S   H E A D E R                                                            */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __VECTORTEST_H__
#define __VECTORTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace std;
using namespace OpenNN;

class VectorTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit VectorTest();

   // DESTRUCTOR

   virtual ~VectorTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();   

   // Reference operator

   void test_reference_operator();

   // Arithmetic operators

   void test_sum_operator();
   void test_rest_operator();
   void test_multiplication_operator();
   void test_division_operator();

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

   // Get methods

   void test_get_size();
   void test_get_display();

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

   void test_remove_element();

   void test_get_assembly();
   void test_difference();

   void test_intersection();

   void test_get_unique();

   // Initialization

   void test_initialize();
   void test_initialize_sequential();
   void test_randomize_uniform();
   void test_randomize_normal();

   // Checking methods

   void test_contains();
   void test_is_in();
   void test_is_constant();
   void test_is_crescent();
   void test_is_decrescent();

   void test_impute_time_series_missing_values_mean();

   // Mathematical operations

   void test_dot_vector();
   void test_dot_matrix();

   void test_calculate_sum();
   void test_calculate_partial_sum();
   void test_calculate_product();

   void test_calculate_mean();
   void test_calculate_standard_deviation();
   void test_calculate_covariance();

   void test_calculate_mean_standard_deviation();

   void test_calculate_minimum();
   void test_calculate_maximum();

   void test_calculate_minimum_maximum();  

   void test_calculate_minimum_missing_values();
   void test_calculate_maximum_missing_values();

   void test_calculate_minimum_maximum_missing_values();

   void test_calculate_explained_variance();

   void test_calculate_statistics();

   void test_calculate_quartiles();

   void test_calculate_histogram();

   void test_calculate_bin();
   void test_calculate_frequency();
   void test_calculate_total_frequencies();

   void test_calculate_maximal_indices();
   void test_calculate_minimal_index();
   void test_index();
   void test_calculate_maximal_index();
   void test_calculate_minimal_indices();

   void test_calculate_minimal_maximal_index();

   void test_calculate_cumulative_index();
   void test_calculate_closest_index();

   void test_calculate_norm();
   void test_calculate_normalized();

   void test_calculate_sum_squared_error();
   void test_calculate_mean_squared_error();
   void test_calculate_root_mean_squared_error();
   
   void test_apply_absolute_value();

   void test_calculate_lower_bounded();
   void test_calculate_upper_bounded();

   void test_calculate_lower_upper_bounded();

   void test_apply_lower_bound();
   void test_apply_upper_bound();
   void test_apply_lower_upper_bounds();

   void test_calculate_less_rank();
   void test_calculate_greater_rank();

   void test_calculate_linear_correlation();
   void test_calculate_linear_correlation_missing_values();

   void test_calculate_linear_regression_parameters();

   void test_threshold();
   void test_symmetric_threshold();
   void test_logistic();
   void test_hyperbolic_tangent();

   void test_hyperbolic_tangent_derivatives();
   void test_hyperbolic_tangent_second_derivatives();
   void test_logistic_derivatives();
   void test_logistic_second_derivatives();
   void test_threshold_derivatives();
   void test_threshold_second_derivatives();
   void test_symmetric_threshold_derivatives();
   void test_symmetric_threshold_second_derivatives();

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

   // Unit testing methods

   void run_test_case();

private:

    static Vector<double> dot(const Vector<double>&, const Matrix<double>&);

    static double dot(const Vector<double>&, const Vector<double>&);
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
