//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "pooling_layer_test.h"

namespace opennn
{

PoolingLayerTest::PoolingLayerTest() : UnitTesting()
{
}


void PoolingLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    dimensions input_dimensions;
    dimensions pool_dimensions;

    input_dimensions = { 28, 29, 1 };
    pool_dimensions = { 3, 2 };

    PoolingLayer pooling_layer(input_dimensions, pool_dimensions);

    assert_true(pooling_layer.get_input_height() == 28 
                && pooling_layer.get_input_width() == 29 
                && pooling_layer.get_channels_number() == 1, LOG);

    assert_true(pooling_layer.get_pool_height() == 3 
                && pooling_layer.get_pool_width() == 2, LOG);
}


void PoolingLayerTest::test_forward_propagate_max_pooling()
{
    cout << "test_forward_propagate_max_pooling" << endl;

    // 2 images 1 channel

    bool is_training = true;

    const Index images_number = 2;

    Tensor<unsigned char, 3> bmp_image_1;
    Tensor<unsigned char, 3> bmp_image_2;

    PoolingLayer pooling_layer;
    PoolingLayer pooling_layer_2;

    dimensions input_dimensions;
    dimensions pool_dimensions;
/*
    bmp_image_1 = read_bmp_image("../examples/mnist/data/images/one/1_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/images/one/1_1.bmp");

    Index input_height = bmp_image_1.dimension(0); // 28
    Index input_width = bmp_image_1.dimension(1); // 28
    Index input_channels = bmp_image_1.dimension(2); // 1

    Index pool_height = 27;
    Index pool_width = 27;

    pair<type*, dimensions> outputs_pair;
    pair<type*, dimensions> outputs_pair_2;

    input_dimensions = { input_height, input_width, input_channels };

    pool_dimensions = { pool_height, pool_width };

    pooling_layer.set(input_dimensions, pool_dimensions);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation(images_number, &pooling_layer);

    Tensor<type, 4> inputs(images_number,
        input_height,
        input_width,
        input_channels);

    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < input_channels; ++c)
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));

    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < input_channels; ++c)
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));

    pooling_layer.forward_propagate_max_pooling(inputs,
                                                &pooling_layer_forward_propagation,
                                                is_training);

    outputs_pair = pooling_layer_forward_propagation.get_outputs_pair();

    assert_true(outputs_pair.second.size() == input_dimensions.size() + 1, LOG);

    type* outputs_data = outputs_pair.first;

    TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    assert_true(outputs(0, 0, 0, 0) == type(255)
                && outputs(1, 0, 0, 0) == type(254), LOG);

    // 2 images 3 channels

    bmp_image_1 = read_bmp_image("../examples/mnist/data/test/4x4_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/test/4x4_1.bmp");

    input_height = bmp_image_1.dimension(0); // 4
    input_width = bmp_image_1.dimension(1); // 4
    input_channels = bmp_image_1.dimension(2); // 3

    input_dimensions = { input_height, input_width, input_channels };

    pool_height = 2;
    pool_width = 2;

    pool_dimensions = { pool_height, pool_width };

    pooling_layer_2.set(input_dimensions, pool_dimensions);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation_2(images_number, &pooling_layer_2);

    inputs.resize(images_number,
        input_height,
        input_width,
        input_channels);

    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < input_channels; ++c)
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));

    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < input_channels; ++c)
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));

    pooling_layer_2.forward_propagate_max_pooling(inputs,
        &pooling_layer_forward_propagation_2,
        is_training);

    outputs_pair_2 = pooling_layer_forward_propagation_2.get_outputs_pair();

    assert_true(outputs_pair_2.second.size() == input_dimensions.size() + 1, LOG);

    type* outputs_data_2 = outputs_pair_2.first;

    TensorMap<Tensor<type, 4>> outputs_2(outputs_data_2,
        outputs_pair_2.second[0],
        outputs_pair_2.second[1],
        outputs_pair_2.second[2],
        outputs_pair_2.second[3]);
    //Image 1:
    assert_true(outputs_2(0, 0, 0, 0) == type(255)
        && outputs_2(0, 0, 0, 1) == type(255)
        && outputs_2(0, 0, 0, 2) == type(255)
        && outputs_2(0, 0, 1, 0) == type(255)
        && outputs_2(0, 0, 1, 1) == type(255)
        && outputs_2(0, 0, 1, 2) == type(255)
        && outputs_2(0, 0, 2, 0) == type(0)
        && outputs_2(0, 0, 2, 1) == type(0)
        && outputs_2(0, 0, 2, 2) == type(0)
        // Image 2:
        && outputs_2(1, 0, 0, 0) == type(0)
        && outputs_2(1, 0, 0, 1) == type(0)
        && outputs_2(1, 0, 0, 2) == type(0)
        && outputs_2(1, 0, 1, 0) == type(255)
        && outputs_2(1, 0, 1, 1) == type(255)
        && outputs_2(1, 0, 1, 2) == type(255)
        && outputs_2(1, 0, 2, 0) == type(255)
        && outputs_2(1, 0, 2, 1) == type(255)
        && round(outputs_2(1, 0, 2, 2)) == type(255), LOG);
*/
}


void PoolingLayerTest::test_forward_propagate_average_pooling()
{
    cout << "test_forward_propagate_average_pooling" << endl;

    // 2 images 1 channel

    bool is_training = true;

    const Index images_number = 2;

    Tensor<unsigned char, 3> bmp_image_1;
    Tensor<unsigned char, 3> bmp_image_2;

    PoolingLayer pooling_layer;
    PoolingLayer pooling_layer_2;

    dimensions input_dimensions;
    dimensions pool_dimensions;
/*
    bmp_image_1 = read_bmp_image("../examples/mnist/data/images/one/1_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/images/one/1_1.bmp");

    Index input_height = bmp_image_1.dimension(0); // 28
    Index input_width = bmp_image_1.dimension(1); // 28
    Index input_channels = bmp_image_1.dimension(2); // 1

    Index pool_height = 27;
    Index pool_width = 27;

    pair<type*, dimensions> outputs_pair;
    pair<type*, dimensions> outputs_pair_2;

    input_dimensions = { input_height, input_width, input_channels };

    pool_dimensions = { pool_height, pool_width };

    pooling_layer.set(input_dimensions, pool_dimensions);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation(images_number, &pooling_layer);

    Tensor<type, 4> inputs(images_number,
                           input_height,
                           input_width,
                           input_channels);

    // Copy bmp_image data into inputs
    for (int h = 0; h < input_height; ++h)
    {
        for (int w = 0; w < input_width; ++w)
        {
            for (int c = 0; c < input_channels; ++c)
            {
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));
            }
        }
    }

    // Copy bmp_image_2 data into inputs
    for (int h = 0; h < input_height; ++h)
    {
        for (int w = 0; w < input_width; ++w)
        {
            for (int c = 0; c < input_channels; ++c)
            {
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));
            }
        }
    }

    pooling_layer.forward_propagate_average_pooling(inputs,
                                                    &pooling_layer_forward_propagation,
                                                    is_training);

    outputs_pair = pooling_layer_forward_propagation.get_outputs_pair();

    assert_true(outputs_pair.second.size() == input_dimensions.size() + 1, LOG);

    type* outputs_data = outputs_pair.first;

    TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    assert_true(round(outputs(0, 0, 0, 0)) == type(14)
                && round(outputs(1, 0, 0, 0)) == type(19), LOG);

    //cout << "outputs:" << endl << "Image 1 (0,0,0): " << round(outputs(0, 0, 0, 0)) << endl << "Image 2 (0,0,0): " << round(outputs(1, 0, 0, 0)) << endl;

    // 2 images 3 channels

    bmp_image_1 = read_bmp_image("../examples/mnist/data/test/4x4_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/test/4x4_1.bmp");

    input_height = bmp_image_1.dimension(0); // 4
    input_width = bmp_image_1.dimension(1); // 4
    input_channels = bmp_image_1.dimension(2); // 3

    input_dimensions = { input_height, input_width, input_channels };

    pool_height = 2;
    pool_width = 2;

    pool_dimensions = { pool_height, pool_width };

    pooling_layer_2.set(input_dimensions, pool_dimensions);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation_2(images_number, &pooling_layer_2);

    inputs.resize(images_number,
                  input_height,
                  input_width,
                  input_channels);

    // Copy bmp_image data into inputs
    for (int h = 0; h < input_height; ++h)
    {
        for (int w = 0; w < input_width; ++w)
        {
            for (int c = 0; c < input_channels; ++c)
            {
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));
            }
        }
    }

    // Copy bmp_image_2 data into inputs
    for (int h = 0; h < input_height; ++h)
    {
        for (int w = 0; w < input_width; ++w)
        {
            for (int c = 0; c < input_channels; ++c)
            {
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));
            }
        }
    }

    pooling_layer_2.forward_propagate_average_pooling(inputs,
                                                      &pooling_layer_forward_propagation_2,
                                                      is_training);

    outputs_pair_2 = pooling_layer_forward_propagation_2.get_outputs_pair();

    assert_true(outputs_pair_2.second.size() == input_dimensions.size() + 1, LOG);

    type* outputs_data_2 = outputs_pair_2.first;

    TensorMap<Tensor<type, 4>> outputs_2(outputs_data_2,
                                         outputs_pair_2.second[0],
                                         outputs_pair_2.second[1],
                                         outputs_pair_2.second[2],
                                         outputs_pair_2.second[3]);
                //Image 1:
    assert_true(outputs_2(0, 0, 0, 0) == type(255)
                && outputs_2(0, 0, 0, 1) == type(255)
                && outputs_2(0, 0, 0, 2) == type(255)
                && outputs_2(0, 0, 1, 0) == type(127.5)
                && outputs_2(0, 0, 1, 1) == type(127.5)
                && outputs_2(0, 0, 1, 2) == type(127.5)
                && outputs_2(0, 0, 2, 0) == type(0)
                && outputs_2(0, 0, 2, 1) == type(0)
                && outputs_2(0, 0, 2, 2) == type(0)
                // Image 2:
                && outputs_2(1, 0, 0, 0) == type(0)
                && outputs_2(1, 0, 0, 1) == type(0)
                && outputs_2(1, 0, 0, 2) == type(0)
                && outputs_2(1, 0, 1, 0) == type(127.5)
                && outputs_2(1, 0, 1, 1) == type(127.5)
                && outputs_2(1, 0, 1, 2) == type(127.5)
                && outputs_2(1, 0, 2, 0) == type(255)
                && outputs_2(1, 0, 2, 1) == type(255)
                && round(outputs_2(1, 0, 2, 2)) == type(255), LOG);
*/
}


void PoolingLayerTest::run_test_case()
{
   cout << "Running pooling layer test case...\n";

    test_constructor();

    // Outputs

    test_forward_propagate_average_pooling();
    test_forward_propagate_max_pooling();

   cout << "End of pooling layer test case.\n\n";
}
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
