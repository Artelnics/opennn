//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   T E S T   C L A S S   H E A D E R                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DATASETTEST_H
#define DATASETTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class DataSetTest : public UnitTesting 
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:  

   explicit DataSetTest();

   virtual ~DataSetTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   void test_get_instances_number();
   void test_get_variables_number();
   void test_get_variables();
   void test_get_display();
   void test_write_first_cell();
   void test_is_binary_classification();
   void test_is_multiple_classification();

   // Data methods

   void test_get_data();
   void test_get_training_data();
   void test_get_selection_data();
   void test_get_testing_data();
   void test_get_inputs();
   void test_get_targets();
  
   // Instance methods

   void test_get_instance();

   // Set methods
   void test_set();
   void test_set_instances_number();
   void test_set_variables_number();
   void test_set_display();

   // Data methods
   void test_set_data();
   void test_empty();

   // Instance methods
   void test_set_instance();
   void test_set_training_instance();
   void test_set_selection_instance();
   void test_set_testing_instance();
   void test_set_input_instance();
   void test_set_target_instance();
   void test_set_training_input_instance();
   void test_set_training_target_instance();
   void test_set_selection_input_instance(); 
   void test_set_selection_target_instance();
   void test_set_testing_input_instance();
   void test_set_testing_target_instance();

   // Data resizing methods

   void test_add_instance();
   void test_append_variable();
   void test_remove_variable();
   void test_subtract_instance(); 
   void test_unuse_constant_columns();
   void test_unuse_repeated_instances();
   void test_unuse_non_significant_inputs();
   void test_unuse_columns_missing_values();

   // Initialization methods
   void test_initialize_data();

   // Statistics methods
   void test_calculate_data_descriptives();
   void test_calculate_data_descriptives_missing_values();
   void test_calculate_training_instances_descriptives();
   void test_calculate_selection_instances_descriptives();
   void test_calculate_testing_instances_descriptives();
   void test_calculate_inputs_descriptives();
   void test_calculate_variables_means();
   void test_calculate_training_targets_mean();
   void test_calculate_selection_targets_mean();
   void test_calculate_testing_targets_mean();

   // Correlation methods
   void test_calculate_linear_correlations();
   void test_calculate_autocorrelations();
   void test_calculate_cross_correlations();
   void test_calculate_input_target_correlations();
   void test_calculate_total_input_correlations();

   // Trending methods
   void test_calculate_trends();

   // Histrogram methods
   void test_calculate_data_histograms();

   // Filtering methods
   void test_filter_data();
   void test_filter_variable();

   // Data scaling
   void test_scale_data_mean_standard_deviation();  
   void test_scale_data_minimum_maximum(); 

   // Input variables scaling
   void test_scale_inputs_mean_standard_deviation();
   void test_scale_inputs_minimum_maximum();

   // Target variables scaling
   void test_scale_targets_mean_standard_deviation();
   void test_scale_targets_minimum_maximum();

   // Input-target variables scaling
   void test_scale_variables_mean_standard_deviation();
   void test_scale_variables_minimum_maximum();

   // Data unscaling
   void test_unscale_data_mean_standard_deviation();
   void test_unscale_data_minimum_maximum();

   // Input variables unscaling
   void test_unscale_inputs_mean_standard_deviation();
   void test_unscale_inputs_minimum_maximum();

   // Target variables unscaling
   void test_unscale_targets_mean_standard_deviation();
   void test_unscale_targets_minimum_maximum();

   // Input-target variables unscaling

 //  void test_unscale_variables_minimum_maximum();

   // Pattern recognition methods

   void test_calculate_target_columns_distribution();
   void test_unuse_most_populated_target();
   void test_balance_binary_targets_distribution();
   void test_balance_multiple_targets_distribution();
   void test_balance_function_regression_targets_distribution();

   // Outlier detection

   void test_calculate_instances_distances();
   //void test_calculate_k_distances();
   //void test_calculate_reachability_distances();
   //void test_calculate_reachability_density();
   //void test_calculate_local_outlier_factor();

   //void test_clean_local_outlier_factor();
   void test_clean_Tukey_outliers();

   // Data generation
   void test_generate_constant_data();
   void test_generate_data_binary_classification();
   void test_generate_data_multiple_classification();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   void test_print();

   void test_read_csv();

   void test_read_adult_csv();
   void test_read_airline_passengers_csv();
   void test_read_car_csv();
   void test_read_empty_csv();
   void test_read_heart_csv();
   void test_read_iris_csv();
   void test_read_mnsit_csv();
   void test_read_one_variable_csv();
   void test_read_pollution_csv();
   void test_read_urinary_inflammations_csv();
   void test_read_wine_csv();
   void test_read_binary_csv();

   //Trasform methods

   void test_convert_time_series();
   void test_convert_autoassociation();

   //Principal components mehtod

   void test_covariance_matrix();
   void test_perform_principal_components_analysis();

   void test_calculate_training_negatives();
   void test_calculate_selection_negatives();

   void test_scrub_missing_values();
   void test_impute_missing_values_mean();

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
