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
/*
    bounding_layer.set_lower_bound(0, type(-1.0));
    bounding_layer.set_upper_bound(0, type(1));
    bounding_layer.set_bounding_method("BoundingLayer");

    Tensor<type, 2> inputs(1,1);
    Tensor<type, 2> outputs(1,1);

    BoundingLayerForwardPropagation bounding_layer_forward_propagation(1, &bounding_layer);

    // Test


    inputs.resize(1, 1);
    inputs(0) = type(-2.0);

    const pair<type*, dimensions> input_pairs = { inputs.data(), {{1, 1}} };
    
//    bounding_layer.forward_propagate({ input_pairs }, &bounding_layer_forward_propagation, is_training);

//    outputs = bounding_layer_forward_propagation.outputs;

//    assert_true(outputs(0) - type(-1.0) < type(NUMERIC_LIMITS_MIN), LOG);


    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{ 0 });
*/
}


/*
namespace opennn
{


void BoundingLayerTest::test_forward_propagate()
{
    BoundingLayer bounding_layer;

    // Test

    inputs(0) = type(2.0);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    bounding_layer_forward_propagation.set(samples_number, &bounding_layer);
    bounding_layer.forward_propagate({ input_pairs }, &bounding_layer_forward_propagation, is_training);

    outputs = bounding_layer_forward_propagation.outputs;

    assert_true(outputs(0) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
}

}
*/
