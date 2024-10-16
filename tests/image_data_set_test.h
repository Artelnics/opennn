//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A   S E T   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef IMAGEDATASETTEST_H
#define IMAGEDATASETTEST_H

#include <fstream>

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/image_data_set.h"
#include "../opennn/batch.h"

namespace opennn
{

class ImageDataSetTest : public UnitTesting
{

public: 

   explicit ImageDataSetTest();

   void test_constructor();

   // Unit testing

   void run_test_case();

  private:

   std::ofstream file;

   string data_string;

   string data_path;

   Index inputs_number;
   Index targets_number;
   Index samples_number;

   Tensor<type, 2> data;

   ImageDataSet data_set;

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
