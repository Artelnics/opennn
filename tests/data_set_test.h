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

#include "../opennn/unit_testing.h"

class DataSetTest : public UnitTesting 
{

public:  

   explicit DataSetTest();

   virtual ~DataSetTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();  

   // Data resizing methods

   void test_add_sample();
   void test_append_variable();
   void test_remove_variable();
   void test_unuse_constant_columns();
   void test_unuse_repeated_samples();
   void test_unuse_uncorrelated_columns();

   // Statistics methods

   void test_calculate_variables_descriptives();
   void test_calculate_input_variables_descriptives();
   void test_calculate_variables_means();

   void test_calculate_used_targets_mean();
   void test_calculate_selection_targets_mean();

   // Correlation methods

   void test_calculate_linear_correlations();
   void test_calculate_autocorrelations();
   void test_calculate_cross_correlations();
   void test_calculate_input_target_correlations();
   void test_calculate_input_columns_correlations();

   // Histrogram methods

   void test_calculate_data_distributions();

   // Filtering methods

   void test_filter_data();

   // Data scaling

   void test_scale_data();

   void test_unscale_data();

   // Classification methods

   void test_calculate_target_distribution();

   void test_calculate_Tukey_outliers();

   // Data generation

   void test_generate_constant_data();

   // Serialization methods

   void test_read_csv();

   void test_read_bank_churn_csv();
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

   // Trasform methods

   void test_transform_time_series();

   // Principal components mehtod

   void test_calculate_training_negatives();
   void test_calculate_selection_negatives();
   void test_scrub_missing_values();
   void test_impute_missing_values_mean();   

   // Data set batch methods

   void test_fill();

   // Unit testing methods

   void run_test_case();

  private:

   std::ofstream file;

   string data_string;

   string data_source_path;


   Index inputs_number;
   Index targets_number;
   Index samples_number;

   Tensor<type, 2> data;

   DataSet data_set;

   Tensor<Index, 1> training_indices;
   Tensor<Index, 1> selection_indices;
   Tensor<Index, 1> testing_indices;

   Tensor<Index, 1> input_variables_indices;
   Tensor<Index, 1> target_variables_indices;

   DataSetBatch data_set_batch;

};

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
