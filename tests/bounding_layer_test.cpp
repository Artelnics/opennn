#include "pch.h"

#include "../opennn/bounding_layer.h"


TEST(BoundingLayerTest, Constructor) 
{
    BoundingLayer bounding_layer;

    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{0});
}


TEST(BoundingLayerTest, ForwardPropagate)
{
    BoundingLayer bounding_layer({1});

    bounding_layer.set_lower_bound(0, type(-1.0));
    bounding_layer.set_upper_bound(0, type(1));
    bounding_layer.set_bounding_method("BoundingLayer");

    BoundingLayerForwardPropagation bounding_layer_forward_propagation(1, &bounding_layer);

    Tensor<type, 2> inputs(1, 1);
    inputs.setConstant(-2.0);
    Tensor<type, 2> outputs(1, 1);

    const pair<type*, dimensions> input_pairs = { inputs.data(), {{1, 1}} };

//    bounding_layer.forward_propagate({ input_pairs }, &bounding_layer_forward_propagation, true);

//    outputs = bounding_layer_forward_propagation.outputs;

//    EXPECT_NEAR(outputs(0), type(-1.0), NUMERIC_LIMITS_MIN);
//    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{ 0 });

}
