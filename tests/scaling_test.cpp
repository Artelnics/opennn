//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_test.h"


ScalingTest::ScalingTest() : UnitTesting()
{
}


ScalingTest::~ScalingTest()
{
}


// @todo std abs

void ScalingTest::test_scale_mean_standard_deviation()
{
   cout << "test_scale_inputs_mean_standard_deviation\n";

   Tensor<type, 2> matrix;

   Tensor<Descriptives, 1> matrix_descriptives;

   Tensor<type, 0> mean_abs;
   Tensor<type, 0> std_abs;

   // Test

   matrix.resize(10, 1);
   matrix.setRandom();

   matrix_descriptives = descriptives(matrix);

   scale_mean_standard_deviation(matrix, 0, matrix_descriptives(0));

   mean_abs = matrix.mean().abs();

   assert_true(mean_abs(0) < static_cast<type>(1e-3) , LOG);
}




void ScalingTest::test_scale_minimum_maximum()
{
   cout << "test_scale_minimum_maximum\n";

   Tensor<type, 2> matrix;

   Tensor<Descriptives, 1> matrix_descriptives;

   Tensor<type, 0> min;
   Tensor<type, 0> max;

   // Test

   matrix.resize(10, 1);
   matrix.setRandom();

   matrix_descriptives = descriptives(matrix);

   scale_minimum_maximum(matrix, 0, matrix_descriptives(0));

   min = matrix.minimum();

//   assert_true(min_abs(0) < static_cast<type>(1e-3) , LOG);
}


void ScalingTest::test_unscale_data_mean_standard_deviation()
{
   cout << "test_unscale_data_mean_standard_deviation\n";

   Tensor<type, 2> matrix(3,1);
   matrix(0,0) = 2.0;
   matrix(1,0) = 5.0;
   matrix(2,0) = 77.0;

   DataSet data(matrix);

   Tensor<Descriptives, 1> descrriptives(1);
   Descriptives descriptives;
   descriptives.set_minimum(5.0);
   descriptives.set_maximum(9.0);
   descriptives.set_mean(8.0);
   descriptives.set_standard_deviation(2.0);
   descrriptives[0] = descriptives ;

   Tensor<type, 2> unescale_matrix(3,1);
   DataSet data_unscaled;
   data.set(unescale_matrix);

   //data.unscale_data_mean_standard_deviation(descrriptives);

   Tensor<type, 2> matrix_solution (3,1);
   matrix_solution(0,0) = descriptives.mean;
   matrix_solution(1,0) = descriptives.mean;
   matrix_solution(2,0) = descriptives.mean;

   //assert_true(data.get_data() == matrix_solution, LOG);
}


void ScalingTest::test_unscale_data_minimum_maximum()
{
   cout << "test_unscale_data_minimum_maximum\n";

   Tensor<type, 2> matrix (3,1);
   matrix(0,0) = 2.0;
   matrix(1,0) = 5.0;
   matrix(2,0) = 77.0;

   DataSet data;
   data.set(matrix);

   Tensor<Descriptives, 1> descrriptives(1);
   Descriptives descriptives;
   descriptives.set_minimum(5.0);
   descriptives.set_maximum(9.0);
   descriptives.set_mean(8.0);
   descriptives.set_standard_deviation(2.0);
   descrriptives[0] = descriptives ;

   Tensor<type, 2> unescale_matrix(3,1);
   DataSet data_unscaled;
   data.set_data(unescale_matrix);

   //data.unscale_data_minimum_maximum(descriptives);

   Tensor<type, 2> matrix_solution (3,1);
   matrix_solution(0,0) = 7.0;
   matrix_solution(1,0) = 7.0;
   matrix_solution(2,0) = 7.0;

   //assert_true(data.get_data() == matrix_solution, LOG);
}


void ScalingTest::run_test_case()
{
   cout << "Running scaling test case...\n";

   // Scaling

   test_scale_mean_standard_deviation();
   test_scale_minimum_maximum();

   // Unscaling

   test_unscale_data_mean_standard_deviation();
   test_unscale_data_minimum_maximum();

   cout << "End of scaling test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library sl free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library sl distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
