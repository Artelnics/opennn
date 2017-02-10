/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V E C T O R   T E S T   C L A S S   H E A D E R                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __VECTORTEST_H__
#define __VECTORTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class VectorTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit VectorTest(void);

   // DESTRUCTOR

   virtual ~VectorTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);   

   // Reference operator

   void test_reference_operator(void);

   // Arithmetic operators

   void test_sum_operator(void);
   void test_rest_operator(void);
   void test_multiplication_operator(void);
   void test_division_operator(void);

   // Operation an assignment operators
   
   void test_sum_assignment_operator(void);
   void test_rest_assignment_operator(void);
   void test_multiplication_assignment_operator(void);
   void test_division_assignment_operator(void);

   // Equality and relational operators

   void test_equal_to_operator(void);
   void test_not_equal_to_operator(void);
   void test_greater_than_operator(void);
   void test_less_than_operator(void);
   void test_greater_than_or_equal_to_operator(void);
   void test_less_than_or_equal_to_operator(void);

   // Output operator

   void test_output_operator(void);

   // Get methods

   void test_get_size(void);
   void test_get_display(void);

   // Set

   void test_set(void);

   void test_set_display(void);

   // Resize methods

   void test_resize(void);

   void test_tuck_in(void);
   void test_take_out(void);

   void test_remove_element(void);

   void test_get_assembly(void);

   // Initialization

   void test_initialize(void);
   void test_initialize_sequential(void);
   void test_randomize_uniform(void);
   void test_randomize_normal(void);

   // Checking methods

   void test_contains(void);
   void test_is_in(void);
   void test_is_constant(void);
   void test_is_crescent(void);
   void test_is_decrescent(void);

   // Mathematical operations

   void test_dot_vector(void);
   void test_dot_matrix(void);

   void test_calculate_sum(void);
   void test_calculate_partial_sum(void);
   void test_calculate_product(void);

   void test_calculate_mean(void);
   void test_calculate_standard_deviation(void);
   void test_calculate_covariance(void);

   void test_calculate_mean_standard_deviation(void);

   void test_calculate_minimum(void);
   void test_calculate_maximum(void);

   void test_calculate_minimum_maximum(void);  

   void test_calculate_minimum_missing_values(void);
   void test_calculate_maximum_missing_values(void);

   void test_calculate_minimum_maximum_missing_values(void);

   void test_calculate_explained_variance(void);

   void test_calculate_statistics(void);

   void test_calculate_histogram(void);

   void test_calculate_bin(void);
   void test_calculate_frequency(void);
   void test_calculate_total_frequencies(void);

   void test_calculate_maximal_indices(void);
   void test_calculate_minimal_index(void);
   void test_index(void);
   void test_calculate_maximal_index(void);
   void test_calculate_minimal_indices(void);

   void test_calculate_minimal_maximal_index(void);

   void test_calculate_cumulative_index(void);
   void test_calculate_closest_index(void);

   void test_calculate_norm(void);
   void test_calculate_normalized(void);

   void test_calculate_sum_squared_error(void);
   void test_calculate_mean_squared_error(void);
   void test_calculate_root_mean_squared_error(void);
   
   void test_apply_absolute_value(void);

   void test_calculate_lower_bounded(void);
   void test_calculate_upper_bounded(void);

   void test_calculate_lower_upper_bounded(void);

   void test_apply_lower_bound(void);
   void test_apply_upper_bound(void);
   void test_apply_lower_upper_bounds(void);

   void test_calculate_less_rank(void);
   void test_calculate_greater_rank(void);

   void test_calculate_linear_correlation(void);
   void test_calculate_linear_correlation_missing_values(void);

   void test_calculate_linear_regression_parameters(void);

   // Scaling and unscaling

   void test_scale_minimum_maximum(void);
   void test_scale_mean_standard_deviation(void);

   void test_unscale_minimum_maximum(void);
   void test_unscale_mean_standard_deviation(void);

   // Parsing methods

   void test_parse(void);

   // Serizalization methods

   void test_load(void);
   void test_save(void);

   // Unit testing methods

   void run_test_case(void);

private:

    static Vector<double> dot(const Vector<double>&, const Matrix<double>&);

    static double dot(const Vector<double>&, const Vector<double>&);

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
