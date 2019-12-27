//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   T E S T   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STATISTICSTEST_H
#define STATISTICSTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class StatisticsTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit StatisticsTest();  

   virtual ~StatisticsTest();

   // Constructor and destructor methods
   void test_constructor();
   void test_destructor();

   // Descriptives
   void test_set_mean();
   void test_set_standard_deviation();
   void test_has_mean_zero_standard_deviation_one();
   void test_has_minimum_minus_one_maximum_one();

   // Minimum
   void test_set_minimum();
   void test_minimum();
   void test_minimum_missing_values();
   void test_minimum_matrix();

   // Maximun
   void test_set_maximum();
   void test_maximum_missing_values();
   void test_maximum_matrix();


   // Mean
   void test_calculate_mean();
   void test_calculate_mean_missing_values();
   void test_weighted_mean();

   // Mean binary
   void test_calculate_means_binary_column();
   void test_means_binary_columns();
   void test_means_binary_columns_missing_values();


   // Median
   void test_calculate_median();
   void test_calculate_median_missing_values();

   // Variance
   void test_calculate_variance_missing_values();
   void test_variance();

   // Assymetry
   void test_calculate_asymmetry();
   void test_calculate_asymmetry_missing_values();

   // Kurtosis
   void test_calculate_kurtosis();
   void test_calculate_kurtosis_missing_values();

   // Standard deviation
   void test_standard_deviation();
   void test_standard_deviation_missing_values();

   // Quartiles
   void test_quartiles();
   void test_calculate_quartiles_missing_values();

   // Box plot
   void test_calculate_box_plot();
   void test_calculate_box_plot_missing_values();

   // Descriptives struct
   void test_descriptives_missing_values();

   // Histogram
   void test_get_bins_number();
   void test_count_empty_bins();
   void test_calculate_minimum_frequency();
   void test_calculate_maximum_frequency();
   void test_calculate_most_populated_bin();
   void test_calculate_minimal_centers();
   void test_calculate_maximal_centers();
   void test_calculate_bin();
   void test_calculate_frequency();
   void test_calculate_histogram();
   void test_total_frequencies();
   void test_calculate_histograms();
   void test_calculate_histogram_missing_values();
   void test_histograms_missing_values();

   // Minimal indices
   void test_calculate_minimal_index();
   void test_calculate_minimal_indices();

   // Maximal indices
   void test_calculate_maximal_index();
   void test_calculate_maximal_indices();

   // Normality
   void test_calculate_norm();

   // Percentiles
   void test_percentiles();
   void test_percentiles_missing_values();

   // Means by categories
   void test_means_by_categories();
   void test_means_by_categories_missing_values();


   // Unit testing methods

   void run_test_case();

private:

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
