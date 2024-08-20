//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A   S E T   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TIMESERIESDATASETTEST_H
#define TIMESERIESDATASETTEST_H

#include <fstream>

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/batch.h"
#include "../opennn/time_series_data_set.h"

namespace opennn
{

class TimeSeriesDataSetTest : public UnitTesting
{

public:  

   explicit TimeSeriesDataSetTest();

   virtual ~TimeSeriesDataSetTest();

   // Constructor and destructor

   void test_constructor();
   void test_destructor();  

   // Correlation

   void test_calculate_autocorrelations();
   void test_calculate_cross_correlations();

   // Trasform

   void test_transform_time_series();

   // Set

   void test_set_time_series_data();
   void test_set_steps_ahead_number();
   void test_set_lags_number();

   //void test_has_time_raw_variables();

   // Saving

   void test_save_time_series_data_binary();  

   // Unit testing

   void run_test_case();

  private:

   ofstream file;

   string data_string;

   string data_source_path;


   Index inputs_number;
   Index targets_number;
   Index samples_number;

   Tensor<type, 2> data;

   TimeSeriesDataSet data_set;

   Tensor<Index, 1> training_indices;
   Tensor<Index, 1> selection_indices;
   Tensor<Index, 1> testing_indices;

   Tensor<Index, 1> input_variables_indices;
   Tensor<Index, 1> target_variables_indices;

   Batch batch;

};

}

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
