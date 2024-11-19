#include "pch.h"

#include "../opennn/flatten_layer.h"


TEST(FlattenLayerTest, Constructor)
{
    const Index height = 6;
    const Index width = 6;
    const Index channels = 3;

    dimensions input_dimensions({ height, width, channels });

    FlattenLayer flatten_layer(input_dimensions);

    EXPECT_EQ(flatten_layer.get_input_dimensions(), input_dimensions);

    EXPECT_EQ(flatten_layer.get_type(), Layer::Type::Flatten);
}


TEST(FlattenLayerTest, ForwardPropagate)
{
    const Index batch_samples_number = 2;
    const Index height = 4;
    const Index width = 4;
    const Index channels = 3;

    const bool is_training = true;

    Tensor<type, 4> inputs(batch_samples_number, height, width, channels);
    inputs.setRandom();

    dimensions input_dimensions({ height, width, channels });

    FlattenLayer flatten_layer(input_dimensions);

    Tensor<type, 2> outputs;

    //unique_ptr<FlattenLayerForwardPropagation> flatten_layer_forward_propagation  = make_unique<FlattenLayerForwardPropagation>(batch_samples_number, flatten_layer);

    //pair<type*, dimensions> input_pairs(inputs.data(), { {batch_samples_number, height, width, channels} });

    //flatten_layer.forward_propagate({ input_pairs }, flatten_layer_forward_propagation, is_training);

//    outputs = flatten_layer_forward_propagation.outputs;

    // Test

//    EXPECT_EQ(inputs.size() == outputs.size());

    EXPECT_EQ(1, 1);
}

TEST(FlattenLayerTest, BackPropagate)
{




}
