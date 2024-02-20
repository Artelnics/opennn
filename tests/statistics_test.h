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

#include "../opennn/unit_testing.h"

class StatisticsTest : public UnitTesting
{

public:

   // CONSTRUCTOR

   explicit StatisticsTest();  

   virtual ~StatisticsTest();

   // Minimum

   void test_minimum();

   // Maximun

   void test_maximum();

   // Mean

   void test_mean();

   // Median

   void test_median();

   // Variance
   void test_variance();

   // Assymetry
   void test_asymmetry();

   // Kurtosis
   void test_kurtosis();

   // Standard deviation
   void test_standard_deviation();

   // Quartiles
   void test_quartiles();

   // Box plot
   void test_box_plot();

   // Descriptives struct

   // Histogram

   void test_count_empty_bins();
   void test_calculate_minimum_frequency();
   void test_calculate_maximum_frequency();
   void test_calculate_most_populated_bin();
   void test_calculate_minimal_centers();
   void test_calculate_maximal_centers();
   void test_calculate_bin();
   void test_calculate_frequency();
   void test_histogram();
   void test_total_frequencies();
   void test_histograms();

   // Minimal indices
   void test_calculate_minimal_index();
   void test_calculate_minimal_indices();

   // Maximal indices
   void test_calculate_maximal_index();
   void test_calculate_maximal_indices();

   // Percentiles
   void test_percentiles();

   // Unit testing methods

   void run_test_case();

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
