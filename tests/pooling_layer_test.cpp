//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pooling_layer_test.h"


PoolingLayerTest::PoolingLayerTest() : UnitTesting()
{
}


PoolingLayerTest::~PoolingLayerTest()
{
}


void PoolingLayerTest::test_constructor()
{
    cout << "test_constructor\n";

}

void PoolingLayerTest::test_destructor()
{
   cout << "test_destructor\n";

}

void PoolingLayerTest::test_calculate_average_pooling_outputs()
{
    cout << "test_calculate_average_pooling_outputs\n";

    PoolingLayer pooling_layer;

    Tensor<double> inputs;
    Tensor<double> outputs;

    // Test

    inputs.set(Vector<size_t>({6,6,6,6}));

    pooling_layer.set_pool_size(1,1);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 6 &&
                outputs.get_dimension(1) == 6 &&
                outputs.get_dimension(2) == 6 &&
                outputs.get_dimension(3) == 6, LOG);

    // Test

    inputs.set(Vector<size_t>({10,3,20,20}));

    pooling_layer.set_pool_size(2,2);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 10 &&
                outputs.get_dimension(1) == 3 &&
                outputs.get_dimension(2) == 19 &&
                outputs.get_dimension(3) == 19, LOG);

    // Test

    inputs.set(Vector<size_t>({1,1,4,4}));
    inputs(0,0,0,0) = 1.0;
    inputs(0,0,0,1) = 2.0;
    inputs(0,0,0,2) = 3.0;
    inputs(0,0,0,3) = 4.0;
    inputs(0,0,1,0) = 16.0;
    inputs(0,0,1,1) = 9.0;
    inputs(0,0,1,2) = 4.0;
    inputs(0,0,1,3) = 1.0;
    inputs(0,0,2,0) = 1.0;
    inputs(0,0,2,1) = 8.0;
    inputs(0,0,2,2) = 27.0;
    inputs(0,0,2,3) = 64.0;
    inputs(0,0,3,0) = 256.0;
    inputs(0,0,3,1) = 81.0;
    inputs(0,0,3,2) = 16.0;
    inputs(0,0,3,3) = 1.0;

    pooling_layer.set_pool_size(2, 2);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 1 &&
                outputs.get_dimension(1) == 1 &&
                outputs.get_dimension(2) == 3 &&
                outputs.get_dimension(3) == 3 &&
                outputs(0,0,0,0) == 7.0 &&
                outputs(0,0,0,1) == 4.5 &&
                outputs(0,0,0,2) == 3.0 &&
                outputs(0,0,1,0) == 8.5 &&
                outputs(0,0,1,1) == 12.0 &&
                outputs(0,0,1,2) == 24.0 &&
                outputs(0,0,2,0) == 86.5 &&
                outputs(0,0,2,1) == 33.0 &&
                outputs(0,0,2,2) == 27.0, LOG);

    // Test

    inputs.set(Vector<size_t>({1,1,4,4}));
    inputs(0,0,0,0) = 1.0;
    inputs(0,0,0,1) = 2.0;
    inputs(0,0,0,2) = 3.0;
    inputs(0,0,0,3) = 4.0;
    inputs(0,0,1,0) = 16.0;
    inputs(0,0,1,1) = 9.0;
    inputs(0,0,1,2) = 4.0;
    inputs(0,0,1,3) = 1.0;
    inputs(0,0,2,0) = 1.0;
    inputs(0,0,2,1) = 8.0;
    inputs(0,0,2,2) = 27.0;
    inputs(0,0,2,3) = 64.0;
    inputs(0,0,3,0) = 256.0;
    inputs(0,0,3,1) = 81.0;
    inputs(0,0,3,2) = 16.0;
    inputs(0,0,3,3) = 1.0;

    pooling_layer.set_pool_size(3, 3);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 1 &&
                outputs.get_dimension(1) == 1 &&
                outputs.get_dimension(2) == 2 &&
                outputs.get_dimension(3) == 2 &&
                outputs(0,0,0,0) - 7.8888 < 0.001 &&
                outputs(0,0,0,1) - 13.5555 < 0.001 &&
                outputs(0,0,1,0) - 46.4444 < 0.001 &&
                outputs(0,0,1,1) - 23.4444 < 0.001, LOG);
}

void PoolingLayerTest::test_calculate_max_pooling_outputs()
{
    cout << "test_calculate_max_pooling_outputs\n";

    PoolingLayer pooling_layer;

    Tensor<double> inputs;
    Tensor<double> outputs;

    // Test

    inputs.set(Vector<size_t>({6,6,6,6}));

    pooling_layer.set_pool_size(1,1);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 6 &&
                outputs.get_dimension(1) == 6 &&
                outputs.get_dimension(2) == 6 &&
                outputs.get_dimension(3) == 6, LOG);

    // Test

    inputs.set(Vector<size_t>({10,3,20,20}));

    pooling_layer.set_pool_size(2,2);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 10 &&
                outputs.get_dimension(1) == 3 &&
                outputs.get_dimension(2) == 19 &&
                outputs.get_dimension(3) == 19, LOG);

    // Test

    inputs.set(Vector<size_t>({1,1,4,4}));
    inputs(0,0,0,0) = 1.0;
    inputs(0,0,0,1) = 2.0;
    inputs(0,0,0,2) = 3.0;
    inputs(0,0,0,3) = 4.0;
    inputs(0,0,1,0) = 16.0;
    inputs(0,0,1,1) = 9.0;
    inputs(0,0,1,2) = 4.0;
    inputs(0,0,1,3) = 1.0;
    inputs(0,0,2,0) = 1.0;
    inputs(0,0,2,1) = 8.0;
    inputs(0,0,2,2) = 27.0;
    inputs(0,0,2,3) = 64.0;
    inputs(0,0,3,0) = 256.0;
    inputs(0,0,3,1) = 81.0;
    inputs(0,0,3,2) = 16.0;
    inputs(0,0,3,3) = 1.0;

    pooling_layer.set_pool_size(2, 2);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 1 &&
                outputs.get_dimension(1) == 1 &&
                outputs.get_dimension(2) == 3 &&
                outputs.get_dimension(3) == 3 &&
                outputs(0,0,0,0) == 16.0 &&
                outputs(0,0,0,1) == 9.0 &&
                outputs(0,0,0,2) == 4.0 &&
                outputs(0,0,1,0) == 16.0 &&
                outputs(0,0,1,1) == 27.0 &&
                outputs(0,0,1,2) == 64.0 &&
                outputs(0,0,2,0) == 256.0 &&
                outputs(0,0,2,1) == 81.0 &&
                outputs(0,0,2,2) == 64.0, LOG);

    // Test

    inputs.set(Vector<size_t>({1,1,4,4}));
    inputs(0,0,0,0) = 1.0;
    inputs(0,0,0,1) = 2.0;
    inputs(0,0,0,2) = 3.0;
    inputs(0,0,0,3) = 4.0;
    inputs(0,0,1,0) = -16.0;
    inputs(0,0,1,1) = -9.0;
    inputs(0,0,1,2) = -4.0;
    inputs(0,0,1,3) = -1.0;
    inputs(0,0,2,0) = 1.0;
    inputs(0,0,2,1) = 8.0;
    inputs(0,0,2,2) = 27.0;
    inputs(0,0,2,3) = 64.0;
    inputs(0,0,3,0) = -256.0;
    inputs(0,0,3,1) = -81.0;
    inputs(0,0,3,2) = -16.0;
    inputs(0,0,3,3) = -1.0;

    pooling_layer.set_pool_size(3, 3);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

    assert_true(outputs.get_dimension(0) == 1 &&
                outputs.get_dimension(1) == 1 &&
                outputs.get_dimension(2) == 2 &&
                outputs.get_dimension(3) == 2 &&
                outputs(0,0,0,0) == 27.0 &&
                outputs(0,0,0,1) == 64.0 &&
                outputs(0,0,1,0) == 27.0 &&
                outputs(0,0,1,1) == 64.0, LOG);
}

void PoolingLayerTest::run_test_case()
{
   cout << "Running pooling layer test case...\n";

   // Constructor and destructor

    test_constructor();
    test_destructor();

    // Outputs

    test_calculate_average_pooling_outputs();
    test_calculate_max_pooling_outputs();

   cout << "End of pooling layer test case.\n";
}


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
