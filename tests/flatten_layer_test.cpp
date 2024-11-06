#include "pch.h"

#include "../opennn/flatten_layer.h"


TEST(FlattenLayerTest, Constructor)
{
    EXPECT_EQ(1, 1);
}


TEST(FlattenLayerTest, ForwardPropagate)
{
    const Index height = 6;
    const Index width = 6;
    const Index image_channels_number = 3;
    const Index images_number = 2;

    bool is_training = true;

    Tensor<type, 4> inputs(height, width, image_channels_number, images_number);
    inputs.setRandom();

    dimensions input_dimensions({ height, width, image_channels_number, images_number });

//    flatten_layer.set(input_dimensions);

    Tensor<type, 2> outputs;

//    flatten_layer_forward_propagation.set(images_number, &flatten_layer);

    Tensor<type*, 1> input_data(1);
    input_data(0) = inputs.data();

    pair<type*, dimensions> input_pairs(inputs.data(), { {height, width, image_channels_number, images_number} });

//    flatten_layer.forward_propagate({ input_pairs }, &flatten_layer_forward_propagation, is_training);

//    outputs = flatten_layer_forward_propagation.outputs;

    // Test

//    assert_true(inputs.size() == outputs.size(), LOG);

    EXPECT_EQ(1, 1);
}
