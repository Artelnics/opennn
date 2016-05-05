/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T R I X   T E S T   C L A S S   H E A D E R                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/


#ifndef __MATRIXTEST_H__
#define __MATRIXTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class MatrixTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit MatrixTest(void);

   // DESTRUCTOR

   virtual ~MatrixTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);   

   // Reference operator methods

   void test_reference_operator(void);   

   // Arithmetic operators

   void test_sum_operator(void);
   void test_rest_operator(void);
   void test_multiplication_operator(void);
   void test_division_operator(void);

   // Arithmetic and assignment operators

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

   // Output operators 

   void test_output_operator(void);

   // METHODS

   // Get methods

   void test_get_rows_number(void);
   void test_get_columns_number(void);  

   void test_arrange_row(void);
   void test_arrange_column(void);

   void test_arrange_submatrix(void);

   // Set methods

   void test_set(void);

   void test_set_rows_number(void);
   void test_set_columns_number(void);

   void test_set_row(void);
   void test_set_column(void);

   // Diagonal methods

   void test_get_diagonal(void);
   void test_set_diagonal(void);
   void test_sum_diagonal(void);

   // Resize methods

   void test_append_row(void);
   void test_append_column(void);

   void test_insert_row(void);
   void test_insert_column(void);

   void test_subtract_row(void);
   void test_subtract_column(void);

   void test_sort_less_rows(void);
   void test_sort_greater_rows(void);

   // Initialization methods

   void test_initialize(void);
   void test_randomize_uniform(void);
   void test_randomize_normal(void);

   void test_set_to_identity(void);

   // Mathematical methods

   void test_calculate_sum(void);
   void test_calculate_rows_sum(void);

   void test_dot_vector(void);
   void test_dot_matrix(void);

   void test_calculate_eigenvalues(void);
   void test_calculate_eigenvectors(void);

   void test_direct(void);

   void test_calculate_minimum_maximum(void);
   void test_calculate_mean_standard_deviation(void);

   void test_calculate_statistics(void);

   void test_calculate_histogram(void);

   void test_calculate_covariance_matrix(void);

   void test_calculate_minimal_indices(void);
   void test_calculate_maximal_indices(void);

   void test_calculate_minimal_maximal_indices(void);

   void test_calculate_sum_squared_error(void);
   void test_calculate_mean_squared_error(void);
   void test_calculate_root_mean_squared_error(void);

   void test_calculate_determinant(void);
   void test_calculate_transpose(void);
   void test_calculate_cofactor(void);
   void test_calculate_inverse(void);

   void test_is_symmetric(void);
   void test_is_antisymmetric(void);

   // Scaling methods
 
   void test_scale_mean_standard_deviation(void);
   void test_scale_rows_mean_standard_deviation(void);
   void test_scale_columns_mean_standard_deviation(void);
   void test_scale_rows_columns_mean_standard_deviation(void);

   void test_scale_minimum_maximum(void);
   void test_scale_rows_minimum_maximum(void);
   void test_scale_columns_minimum_maximum(void);
   void test_scale_rows_columns_minimum_maximum(void);

   // Unscaling methods

   void test_unscale_mean_standard_deviation(void);
   void test_unscale_rows_mean_standard_deviation(void);
   void test_unscale_columns_mean_standard_deviation(void);
   void test_unscale_rows_columns_mean_standard_deviation(void);


   void test_unscale_minimum_maximum(void);
   void test_unscale_rows_minimum_maximum(void);
   void test_unscale_columns_minimum_maximum(void);
   void test_unscale_rows_columns_minimum_maximum(void);

   void test_convert_angular_variables_degrees(void);
   void test_convert_angular_variables_radians(void);

   // Serialization methods

   void test_print(void);

   void test_load(void);

   void test_save(void);

   void test_parse(void);

   // Unit testing methods

   void run_test_case(void);

private:

   static Vector<double> dot(const Matrix<double>&, const Vector<double>&);

   static Matrix<double> dot(const Matrix<double>&, const Matrix<double>&);
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
