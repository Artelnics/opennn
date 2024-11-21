#include "pch.h"

#include "../opennn/pooling_layer.h"

struct PoolingLayerConfig {
    dimensions input_dimensions;
    dimensions pool_dimensions;
    dimensions stride_dimensions;
    dimensions padding_dimensions;
    PoolingLayer::PoolingMethod pooling_method;
    string test_name;
    Tensor<type, 4> input_data;
    Tensor<type, 4> expected_output;
};


class PoolingLayerConstructorTest : public ::testing::TestWithParam<PoolingLayerConfig> {};


TEST_P(PoolingLayerConstructorTest, Constructor) 
{
    auto param = GetParam();

    PoolingLayer pooling_layer(param.input_dimensions,
                               param.pool_dimensions,
                               param.stride_dimensions,
                               param.padding_dimensions,
                               param.pooling_method,
                               param.test_name);

    EXPECT_EQ(pooling_layer.get_type(), Layer::Type::Pooling);
    EXPECT_EQ(pooling_layer.get_input_dimensions(), param.input_dimensions);
    EXPECT_EQ(pooling_layer.get_pool_height(), param.pool_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_pool_width(), param.pool_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_row_stride(), param.stride_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_column_stride(), param.stride_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_padding_height(), param.padding_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_padding_width(), param.padding_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_pooling_method(), param.pooling_method);
}


INSTANTIATE_TEST_CASE_P(PoolingLayerConstructorTests,PoolingLayerConstructorTest,::testing::Values(
        PoolingLayerConfig{ {4, 4, 1}, {2, 2}, {2, 2}, {0, 0}, PoolingLayer::PoolingMethod::MaxPooling, "MaxPoolingNoPadding1Channel" },
        PoolingLayerConfig{ {4, 4, 3}, {2, 2}, {2, 2}, {0, 0}, PoolingLayer::PoolingMethod::MaxPooling, "MaxPoolingNoPadding3Channels" },
        PoolingLayerConfig{ {4, 4, 1}, {2, 2}, {2, 2}, {0, 0}, PoolingLayer::PoolingMethod::AveragePooling, "AveragePoolingNoPadding1Channel" },
        PoolingLayerConfig{ {4, 4, 3}, {2, 2}, {2, 2}, {0, 0}, PoolingLayer::PoolingMethod::AveragePooling, "AveragePoolingNoPadding3Channels" }
    )
);


/*
TEST(PoolingLayerTest, ForwardPropagateMaxPooling)
{
/*
    // 2 images 1 channel

    bool is_training = true;

    const Index images_number = 2;

    Tensor<unsigned char, 3> bmp_image_1;
    Tensor<unsigned char, 3> bmp_image_2;

    PoolingLayer pooling_layer;
    PoolingLayer pooling_layer_2;

    dimensions input_dimensions;
    dimensions pool_dimensions;

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

    EXPECT_EQ(outputs_pair.second.size() == input_dimensions.size() + 1);

    type* outputs_data = outputs_pair.first;

    TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    EXPECT_EQ(outputs(0, 0, 0, 0) == type(255)
        && outputs(1, 0, 0, 0) == type(254));

}


/*
void PoolingLayerTest::test_forward_propagate_max_pooling()
{
    cout << "test_forward_propagate_max_pooling" << endl;

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

    EXPECT_EQ(outputs_pair_2.second.size() == input_dimensions.size() + 1);

    type* outputs_data_2 = outputs_pair_2.first;

    TensorMap<Tensor<type, 4>> outputs_2(outputs_data_2,
        outputs_pair_2.second[0],
        outputs_pair_2.second[1],
        outputs_pair_2.second[2],
        outputs_pair_2.second[3]);
    //Image 1:
    EXPECT_EQ(outputs_2(0, 0, 0, 0) == type(255)
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
        && round(outputs_2(1, 0, 2, 2)) == type(255));
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

    EXPECT_EQ(outputs_pair.second.size() == input_dimensions.size() + 1);

    type* outputs_data = outputs_pair.first;

    TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    EXPECT_EQ(round(outputs(0, 0, 0, 0)) == type(14)
                && round(outputs(1, 0, 0, 0)) == type(19));

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

    EXPECT_EQ(outputs_pair_2.second.size() == input_dimensions.size() + 1);

    type* outputs_data_2 = outputs_pair_2.first;

    TensorMap<Tensor<type, 4>> outputs_2(outputs_data_2,
                                         outputs_pair_2.second[0],
                                         outputs_pair_2.second[1],
                                         outputs_pair_2.second[2],
                                         outputs_pair_2.second[3]);
                //Image 1:
    EXPECT_EQ(outputs_2(0, 0, 0, 0) == type(255)
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
                && round(outputs_2(1, 0, 2, 2)) == type(255));
}

}
*/
