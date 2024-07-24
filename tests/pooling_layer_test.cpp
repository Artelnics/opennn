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

/*
void PoolingLayerTest::test_calculate_average_pooling_outputs()
{
    cout << "test_calculate_average_pooling_outputs\n";

    Tensor<type, 4> inputs;
    Tensor<type, 4> outputs;

    // Test

    inputs.resize(6,6,6,6);

    pooling_layer.set_pool_size(1, 1);
    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 6 &&
//                outputs.dimension(1) == 6 &&
//                outputs.dimension(2) == 6 &&
//                outputs.dimension(3) == 6, LOG);

    // Test

//    inputs.resize(({10,3,20,20}));

//    pooling_layer.set_pool_size(2,2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 10 &&
//                outputs.dimension(1) == 3 &&
//                outputs.dimension(2) == 19 &&
//                outputs.dimension(3) == 19, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = 16.0;
//    inputs(0,0,1,1) = 9.0;
//    inputs(0,0,1,2) = 4.0;
//    inputs(0,0,1,3) = type(1);
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = 256.0;
//    inputs(0,0,3,1) = 81.0;
//    inputs(0,0,3,2) = 16.0;
//    inputs(0,0,3,3) = type(1);

//    pooling_layer.set_pool_size(2, 2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 3 &&
//                outputs.dimension(3) == 3 &&
//                outputs(0,0,0,0) == 7.0 &&
//                outputs(0,0,0,1) == 4.5 &&
//                outputs(0,0,0,2) == 3.0 &&
//                outputs(0,0,1,0) == 8.5 &&
//                outputs(0,0,1,1) == 12.0 &&
//                outputs(0,0,1,2) == 24.0 &&
//                outputs(0,0,2,0) == 86.5 &&
//                outputs(0,0,2,1) == 33.0 &&
//                outputs(0,0,2,2) == 27.0, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = 16.0;
//    inputs(0,0,1,1) = 9.0;
//    inputs(0,0,1,2) = 4.0;
//    inputs(0,0,1,3) = type(1);
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = 256.0;
//    inputs(0,0,3,1) = 81.0;
//    inputs(0,0,3,2) = 16.0;
//    inputs(0,0,3,3) = type(1);

//    pooling_layer.set_pool_size(3, 3);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 2 &&
//                outputs.dimension(3) == 2 &&
//                outputs(0,0,0,0) - 7.8888 < 0.001 &&
//                outputs(0,0,0,1) - 13.5555 < 0.001 &&
//                outputs(0,0,1,0) - 46.4444 < 0.001 &&
//                outputs(0,0,1,1) - 23.4444 < 0.001, LOG);

    // input_dimensions

    const Index input_images = 1;
    const Index channels = 1;

    const Index input_heigth = 4;
    const Index input_width = 4;

    //pooling dimensions

    const Index pool_height = 2;
    const Index pool_width = 2;

    //stride

    const Index row_stride = 1;
    const Index column_stride = 1;

    //output dimensions

    const Index output_height = (input_heigth - pool_height)/row_stride + 1;
    const Index output_width = (input_width - pool_width)/column_stride +1;

    inputs.resize(input_heigth, input_width, channels, input_images);
    outputs.resize(output_height, output_width, channels, input_images);

    inputs.setRandom();

    //pooling average

    Index column = 0;
    Index row = 0;

    for(int i = 0; i<input_images; i++)
    {
        for(int c = 0; c < channels; c++)
        {
            for(int k = 0; k < output_width; k++)
            {
                for(int l = 0; l < output_height; l++)
                {
                    float tmp_result = 0;

                    for(int m = 0; m < pool_width; m++)
                    {
                        column = m*column_stride + k;

                        for(int n = 0; n < pool_height; n++)
                        {
                            row = n*row_stride + l;

                            tmp_result += inputs(row,column,c,i);
                        }
                    }

                    outputs(l,k,c,i) = tmp_result/(pool_width*pool_height);
                }
            }
        }
    }
}
*/


void PoolingLayerTest::test_forward_propagate_max_pooling()
{
    cout << "test_forward_propagate_max_pooling" << endl;
}


void PoolingLayerTest::test_forward_propagate_average_pooling()
{
    cout << "test_forward_propagate_average_pooling" << endl;

    const Index batch_samples_number = 1;

    const Index input_channels = 3;
    const Index input_height = 5;
    const Index input_width = 5;

    const Index pool_height = 2;
    const Index pool_width = 2;

    const Index targets_number = 1;

    dimensions input_dimensions;
    dimensions pool_dimensions;
    dimensions output_dimensions;

    pair<type*, dimensions> outputs_pair;

    ImageDataSet image_data_set(batch_samples_number,
                                input_channels,
                                input_height,
                                input_width,
                                targets_number);

    image_data_set.set_data_constant(type(1));

    input_dimensions = {input_channels,
                        input_height,
                        input_width};

    pool_dimensions = {pool_height,
                       pool_width};

    bool is_training = true;

    PoolingLayer pooling_layer(input_dimensions, pool_dimensions);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation(batch_samples_number, &pooling_layer);

    Tensor<type, 4> inputs(batch_samples_number,
                           input_channels,
                           input_height,
                           input_width);

    inputs.setValues({{
        {{0.0, 1.0, 2.0, 2.0, 2.0},
         {0.0, 1.0, 2.0, 2.0, 2.0},
         {0.0, 1.0, 2.0, 2.0, 2.0},
         {0.0, 1.0, 2.0, 2.0, 2.0},
         {0.0, 1.0, 2.0, 2.0, 2.0}
        },
        {
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0}
        },
        {
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 1.0, 2.0, 2.0, 2.0}
        }
    }});

    pooling_layer.forward_propagate_average_pooling(inputs,
                                                    &pooling_layer_forward_propagation,
                                                    is_training);

    outputs_pair = pooling_layer_forward_propagation.get_outputs_pair();

    assert_true(outputs_pair.second.size() == input_dimensions.size(), LOG);

    for(Index i = 0; i < static_cast<Index>(output_dimensions.size()); i++)
    {
//        assert_true(outputs_pair.second.dimensions(i) <= input_dimensions(i), LOG);
    }
/*
    type* outputs_data = pooling_layer_forward.outputs_data(0);

    TensorMap<Tensor<type, 4>> outputs(outputs_data,
                                       output_dimensions[0],
                                       output_dimensions(1),
                                       output_dimensions(2),
                                       output_dimensions(3));

    Tensor<type, 3> batch = outputs.chip(0,0);

    cout << "1 single channel: " << endl << batch.chip(0,0) << endl;

    cout << "outputs: " << endl << outputs << endl;
*/
}

/*
void PoolingLayerTest::test_calculate_max_pooling_outputs()
{
    cout << "test_calculate_max_pooling_outputs\n";

//    Tensor<type, 2> inputs;
//    Tensor<type, 2> outputs;

    // Test

//    inputs.resize(({6,6,6,6}));

//    pooling_layer.set_pool_size(1,1);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 6 &&
//                outputs.dimension(1) == 6 &&
//                outputs.dimension(2) == 6 &&
//                outputs.dimension(3) == 6, LOG);

    // Test

//    inputs.resize(({10,3,20,20}));

//    pooling_layer.set_pool_size(2,2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 10 &&
//                outputs.dimension(1) == 3 &&
//                outputs.dimension(2) == 19 &&
//                outputs.dimension(3) == 19, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = 16.0;
//    inputs(0,0,1,1) = 9.0;
//    inputs(0,0,1,2) = 4.0;
//    inputs(0,0,1,3) = type(1);
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = 256.0;
//    inputs(0,0,3,1) = 81.0;
//    inputs(0,0,3,2) = 16.0;
//    inputs(0,0,3,3) = type(1);

//    pooling_layer.set_pool_size(2, 2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 3 &&
//                outputs.dimension(3) == 3 &&
//                outputs(0,0,0,0) == 16.0 &&
//                outputs(0,0,0,1) == 9.0 &&
//                outputs(0,0,0,2) == 4.0 &&
//                outputs(0,0,1,0) == 16.0 &&
//                outputs(0,0,1,1) == 27.0 &&
//                outputs(0,0,1,2) == 64.0 &&
//                outputs(0,0,2,0) == 256.0 &&
//                outputs(0,0,2,1) == 81.0 &&
//                outputs(0,0,2,2) == 64.0, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = -16.0;
//    inputs(0,0,1,1) = -9.0;
//    inputs(0,0,1,2) = -4.0;
//    inputs(0,0,1,3) = -1.0;
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = -256.0;
//    inputs(0,0,3,1) = -81.0;
//    inputs(0,0,3,2) = -16.0;
//    inputs(0,0,3,3) = -1.0;

//    pooling_layer.set_pool_size(3, 3);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 2 &&
//                outputs.dimension(3) == 2 &&
//                outputs(0,0,0,0) == 27.0 &&
//                outputs(0,0,0,1) == 64.0 &&
//                outputs(0,0,1,0) == 27.0 &&
//                outputs(0,0,1,1) == 64.0, LOG);

    cout << "test_calculate_max_pooling_outputs\n";

    //input_dimensions
    const Index input_images = 1;
    const Index channels = 1;

    const Index input_heigth = 4;
    const Index input_width = 4;

    //pooling dimensions
    const Index pool_height = 2;
    const Index pool_width = 2;

    //stride
    const Index row_stride = 1;
    const Index column_stride = 1;

    //output dimensions

    const Index output_height = (input_heigth - pool_height)/row_stride + 1;
    const Index output_width = (input_width - pool_width)/column_stride +1;

    Tensor<type, 4> inputs(input_heigth, input_width, channels, input_images);
    Tensor<type, 4> outputs(output_height, output_width, channels, input_images);

    inputs.setRandom();

    //pooling average

    Index column = 0;
    Index row = 0;

    for(int i = 0; i < input_images; i++)
    {
        for(int c = 0; c < channels; c++)
        {
            for(int k = 0; k < output_width; k++)
            {
                for(int l = 0; l < output_height; l++)
                {
                    float tmp_result = 0;

                    float final_result = 0;

                    for(int m = 0; m < pool_width; m++)
                    {
                        column = m*column_stride + k;

                        for(int n = 0; n < pool_height; n++)
                        {
                            row = n*row_stride + l;

                            tmp_result = inputs(row,column,c,i);

                            if(tmp_result > final_result) final_result = tmp_result;
                        }
                    }

                    outputs(l,k,c,i) = final_result;
                }
            }
        }
    }
}
*/

void PoolingLayerTest::run_test_case()
{
   cout << "Running pooling layer test case...\n";

   // Constructor and destructor

    test_constructor();
    test_destructor();

    // Outputs

    test_forward_propagate_average_pooling();
    test_forward_propagate_max_pooling();

   cout << "End of pooling layer test case.\n\n";
}


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
