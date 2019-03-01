/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T R I X   T E S T   C L A S S   H E A D E R                                                            */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
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

   explicit MatrixTest();

   // DESTRUCTOR

   virtual ~MatrixTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Reference operator methods

   void test_reference_operator();

   // Arithmetic operators

   void test_sum_operator();
   void test_rest_operator();
   void test_multiplication_operator();
   void test_division_operator();

   // Arithmetic and assignment operators

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

   // Output operators 

   void test_output_operator();

   // METHODS

   // Get methods

   void test_get_rows_number();
   void test_get_columns_number();

   void test_get_row();
   void test_get_column();

   void test_get_submatrix();

   // Set methods

   void test_set();

   void test_set_rows_number();
   void test_set_columns_number();

   void test_set_row();
   void test_set_column();

   // Diagonal methods

   void test_get_diagonal();
   void test_set_diagonal();
   void test_sum_diagonal();

   // Resize methods

   void test_append_row();
   void test_append_column();

   void test_insert_row();
   void test_insert_column();

   void test_subtract_row();
   void test_subtract_column();

   void test_sort_less_rows();
   void test_sort_greater_rows();

   // Initialization methods

   void test_initialize();
   void test_randomize_uniform();
   void test_randomize_normal();

   void test_set_to_identity();

   // Mathematical methods

   void test_calculate_sum();
   void test_calculate_rows_sum();

   void test_dot_vector();
   void test_dot_matrix();

   void test_calculate_eigenvalues();
   void test_calculate_eigenvectors();

   void test_direct();

   void test_calculate_minimum_maximum();
   void test_calculate_mean_standard_deviation();

   void test_calculate_statistics();

   void test_calculate_histogram();

   void test_calculate_covariance_matrix();

   void test_calculate_minimal_indices();
   void test_calculate_maximal_indices();

   void test_calculate_minimal_maximal_indices();

   void test_calculate_sum_squared_error();
   void test_calculate_mean_squared_error();
   void test_calculate_root_mean_squared_error();

   void test_calculate_determinant();
   void test_calculate_transpose();
   void test_calculate_cofactor();
   void test_calculate_inverse();
   void test_calculate_lu_inverse();

   void test_is_symmetric();
   void test_is_antisymmetric();

   void test_calculate_k_means();

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

   void test_unscale_mean_standard_deviation();
   void test_unscale_rows_mean_standard_deviation();
   void test_unscale_columns_mean_standard_deviation();
   void test_unscale_rows_columns_mean_standard_deviation();


   void test_unscale_minimum_maximum();
   void test_unscale_rows_minimum_maximum();
   void test_unscale_columns_minimum_maximum();
   void test_unscale_rows_columns_minimum_maximum();

   void test_convert_angular_variables_degrees();
   void test_convert_angular_variables_radians();

   void test_to_time_t();

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
