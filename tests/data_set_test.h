/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   D A T A   S E T   T E S T   C L A S S   H E A D E R                                                        */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __DATASETTEST_H__
#define __DATASETTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class DataSetTest : public UnitTesting 
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:  

   // GENERAL CONSTRUCTOR

   explicit DataSetTest(void);


   // DESTRUCTOR

   virtual ~DataSetTest(void);

    // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   void test_get_instances_number(void);
   void test_get_variables_number(void);

   void test_get_variables(void);

   void test_get_display(void);

   // Data methods

   void test_get_data(void);

   void test_arrange_training_data(void);
   void test_arrange_selection_data(void);
   void test_arrange_testing_data(void);

   void test_arrange_input_data(void);
   void test_arrange_target_data(void);
  
   // Instance methods

   void test_get_instance(void);

   // Set methods

   void test_set(void);

   void test_set_instances_number(void);
   void test_set_variables_number(void);

   void test_set_display(void);

   // Data methods

   void test_set_data(void);

   // Instance methods

   void test_set_instance(void);

   void test_set_training_instance(void);
   void test_set_selection_instance(void);
   void test_set_testing_instance(void);

   void test_set_input_instance(void);
   void test_set_target_instance(void);

   void test_set_training_input_instance(void);
   void test_set_training_target_instance(void);

   void test_set_selection_input_instance(void); 
   void test_set_selection_target_instance(void);

   void test_set_testing_input_instance(void);
   void test_set_testing_target_instance(void);

   // Data resizing methods

   void test_add_instance(void);
   void test_subtract_instance(void); 

   void test_subtract_constant_variables(void);
   void test_subtract_repeated_instances(void);

   // Initialization methods

   void test_initialize_data(void);

   // Statistics methods

   void test_calculate_data_statistics(void);
   void test_calculate_data_statistics_missing_values(void);

   void test_calculate_training_instances_statistics(void);
   void test_calculate_selection_instances_statistics(void);
   void test_calculate_testing_instances_statistics(void);

   void test_calculate_input_variables_statistics(void);
   void test_calculate_targets_statistics(void);

   // Correlation methods

   void test_calculate_linear_correlations(void);

   void test_calculate_autocorrelation(void);
   void test_calculate_cross_correlation(void);

   // Histrogram methods

   void test_calculate_data_histograms(void);

   // Filtering methods

   void test_filter_data(void);

   // Data scaling

   void test_scale_data_mean_standard_deviation(void);  
   void test_scale_data_minimum_maximum(void); 

   // Input variables scaling

   void test_scale_inputs_mean_standard_deviation(void);
   void test_scale_inputs_minimum_maximum(void);

   // Target variables scaling

   void test_scale_targets_mean_standard_deviation(void);
   void test_scale_targets_minimum_maximum(void);

   // Input-target variables scaling

   void test_scale_variables_mean_standard_deviation(void);
   void test_scale_variables_minimum_maximum(void);

   // Data unscaling

   void test_unscale_data_mean_standard_deviation(void);
   void test_unscale_data_minimum_maximum(void);

   // Input variables unscaling

   void test_unscale_inputs_mean_standard_deviation(void);
   void test_unscale_inputs_minimum_maximum(void);

   // Target variables unscaling

   void test_unscale_targets_mean_standard_deviation(void);
   void test_unscale_targets_minimum_maximum(void);

   // Input-target variables unscaling

   void test_unscale_variables_mean_standard_deviation(void);
   void test_unscale_variables_minimum_maximum(void);

   // Pattern recognition methods

   void test_calculate_target_distribution(void);

   void test_unuse_most_populated_target(void);

   void test_balance_binary_targets_distribution(void);
   void test_balance_multiple_targets_distribution(void);
   void test_balance_function_regression_targets_distribution(void);

   // Outlier detection

   //void test_calculate_instances_distances(void);
   //void test_calculate_k_distances(void);
   //void test_calculate_reachability_distances(void);
   //void test_calculate_reachability_density(void);
   //void test_calculate_local_outlier_factor(void);

   //void test_clean_local_outlier_factor(void);
   void test_clean_Tukey_outliers(void);

   // Data generation

   void test_generate_data_function_regression(void);

   void test_generate_data_binary_classification(void);
   void test_generate_data_multiple_classification(void);

   // Serialization methods

   void test_to_XML(void);
   void test_from_XML(void);

   void test_print(void);

   void test_save(void);
   void test_load(void);

   void test_print_data(void);
   void test_save_data(void);
   void test_load_data(void);

   void test_get_data_statistics(void);
   void test_print_data_statistics(void);

   void test_get_training_instances_statistics(void);
   void test_print_training_instances_statistics(void);
   void test_save_training_instances_statistics(void);

   void test_get_selection_instances_statistics(void);
   void test_print_selection_instances_statistics(void);
   void test_save_selection_instances_statistics(void);

   void test_get_testing_instances_statistics(void);
   void test_print_testing_instances_statistics(void);
   void test_save_testing_instances_statistics(void);

   void test_get_instances_statistics(void);
   void test_print_instances_statistics(void);
   void test_save_instances_statistics(void);

   void test_convert_time_series(void);
   void test_convert_autoassociation(void);

   void test_convert_angular_variable_degrees(void);
   void test_convert_angular_variable_radians(void);

   void test_convert_angular_variables_degrees(void);
   void test_convert_angular_variables_radians(void);

   void test_convert_angular_variables(void);

   void test_scrub_missing_values(void);

   // String utilities

   void test_trim(void);
   void test_get_trimmed(void);

   void test_count_tokens(void);
   void test_get_tokens(void);

   void test_is_numeric(void);

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
