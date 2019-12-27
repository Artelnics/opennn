//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T R I X   T E S T   C L A S S   H E A D E R                       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MATRIXTEST_H
#define MATRIXTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class MatrixTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit MatrixTest();

   virtual ~MatrixTest();
   
   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Arithmetic operators

   void test_sum_operator();
   void test_rest_operator();
   void test_multiplication_operator();
   void test_division_operator();

   // Equality and relational operators

   void test_equal_to_operator();
   void test_not_equal_to_operator();
   void test_greater_than_operator();
   void test_less_than_operator();
   void test_greater_than_or_equal_to_operator();
   void test_less_than_or_equal_to_operator();

   // Output operators 

   void test_output_operator();

   // Get methods

   void test_get_rows_number();
   void test_get_columns_number();

   void test_get_row();
   void test_get_rows();
   void test_get_column();
   void test_get_columns();

   void test_get_rows_equal_to();

   void test_get_header();

   void test_get_column_index();
   void test_get_columns_indices();
   void test_get_binary_columns_indices();

   void test_get_submatrix();
   void test_get_submatrix_rows();
   void test_get_submatrix_columns();

   void test_get_first();
   void test_get_last();

   void test_get_constant_columns_indices();

   void test_get_first_rows();
   void test_get_last_rows();
   void test_get_first_columns();
   void test_get_last_columns();

   // Set methods

   void test_set();

   void test_set_identity();

   void test_set_rows_number();
   void test_set_columns_number();

   void test_set_header();

   void test_set_row();
   void test_set_column();

   void test_set_submatrix_rows();

   //Check methods

   void test_empty();

   void test_is_square();

   void test_is_symmetric();
   void test_is_antisymmetric();

   void test_is_diagonal();

   void test_is_scalar();

   void test_is_identity();

   void test_is_binary();
   void test_is_column_binary();

   void test_is_column_constant();

   void test_is_positive();

   void test_is_row_equal_to();

   void test_has_column_value();

   // Count methods

   void test_count_diagonal_elements();

   void test_count_off_diagonal_elements();

   void test_count_equal_to();
   void test_count_not_equal_to();

   void test_count_equal_to_by_rows();

   void test_count_rows_equal_to();
   void test_count_rows_not_equal_to();

   // Not a number

   void test_has_nan();

   void test_count_nan();

   void test_count_rows_with_nan();
   void test_count_columns_with_nan();

   void test_count_nan_rows();
   void test_count_nan_columns();

   //Filter

   void test_filter_column_equal_to();
   void test_filter_column_not_equal_to();

   void test_filter_column_less_than();
   void test_filter_column_greater_than();

   void test_filter_column_minimum_maximum();

   // Diagonal methods

   void test_get_diagonal();
   void test_set_diagonal();
   void test_sum_diagonal();

   // Resize methods

   void test_append_row();
   void test_append_column();

   void test_insert_row();
   void test_insert_column();

   void test_insert_row_values();

   void test_add_column();
   void test_add_column_first();

   void test_swap_columns();

   void test_delete_row();
   void test_delete_column();

   void test_delete_rows_wiht_value();
   void test_delete_columns_with_value();

   void test_delete_first_rows();
   void test_delete_last_rows();

   void test_delete_first_columns();
   void test_delete_last_columns();

   void test_delete_columns_name_contain();

   void test_delete_constant_rows();
   void test_delete_constant_columns();

   void test_delete_binary_rows();
   void test_delete_binary_columns();

   void test_assembla_rows();
   void test_assembla_columns();

   void test_subtract_row();

   // Sorting methods

   void test_sort_ascending();
   void test_sort_descending();

   //Replace methods

   void test_replace();

   void test_replace_header();

   void test_replace_in_row();
   void test_replace_in_column();

   void test_replace_substring();

   void test_replace_contains();
   void test_replace_contains_in_row();

   void test_replace_column_equal_to();
   void test_replace_column_not_equal_to();

   void test_replace_column_less_than_string();

   void test_replace_column_contain();

   // Initialization methods

   void test_initialize();

   void test_randomize_uniform();
   void test_randomize_normal();

   void test_initialize_identity();
   void test_initialize_diagonal();

   void test_append_header();

   void test_tuck_in();

   void test_set_to_identity();

   // Mathematical methods

   void test_calculate_sum();
   void test_calculate_rows_sum();
   void test_calculate_columns_sum();

   void test_sum_row();
   void test_sum_rows();

   void test_substract_rows();
   void test_multiply_rows();
   void test_divide_rows();

   void test_calculate_trace();

   void test_get_indices_less_than();

   void test_calculate_reverse_columns();

   void test_compare_rows();

   void test_dot_vector();
   void test_dot_matrix();

   void test_eigenvalues();
   void test_eigenvectors();

   void test_direct();

   void test_sum_squared_error();
   void test_calculate_mean_squared_error();
   void test_calculate_root_mean_squared_error();

   void test_determinant();
   void test_calculate_transpose();
   void test_cofactor();
   void test_calculate_inverse();   

   //Conversion

   void test_matrix_to_string();
   void test_to_zeros();

   // Scaling methods
 
   void test_scale_mean_standard_deviation();
   void test_scale_rows_mean_standard_deviation();
   void test_scale_columns_mean_standard_deviation();
   void test_scale_rows_columns_mean_standard_deviation();

   void test_scale_minimum_maximum();
   void test_scale_rows_minimum_maximum();
   void test_scale_columns_minimum_maximum();
   void test_scale_rows_columns_minimum_maximum();

   // Unscaling methods

   void test_to_time_t();
   void test_get_tensor();

   // Serialization methods

   void test_print();

   void test_load();

   void test_save();

   void test_parse();

   // Unit testing methods

   void run_test_case();

private:

   static Vector<double> dot(const Matrix<double>&, const Vector<double>&);

   static Matrix<double> dot(const Matrix<double>&, const Matrix<double>&);
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
