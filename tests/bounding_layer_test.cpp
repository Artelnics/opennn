#include "pch.h"

#include "../opennn/bounding_layer.h"


TEST(BoundingLayerTest, Constructor) 
{
    BoundingLayer bounding_layer;

//    EXPECT_EQ(bounding_layer_1.get_output_dimensions(), dimensions{});
}


/*
namespace opennn
{


void BoundingLayerTest::test_forward_propagate()
{
    BoundingLayer bounding_layer;
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    pair<type*, dimensions> input_pairs;

    BoundingLayerForwardPropagation bounding_layer_forward_propagation;

    // Test

    Index samples_number = 1;
    Index inputs_number = 1;
    bool is_training = false;

    bounding_layer.set(inputs_number);

    bounding_layer.set_lower_bound(0, type(-1.0));
    bounding_layer.set_upper_bound(0, type(1));
    bounding_layer.set_bounding_method("BoundingLayer");

    inputs.resize(1, 1);
    inputs(0) = type(-2.0);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};
    input_pairs.first = inputs.data();
    input_pairs.second = {{samples_number, inputs_number}};

    bounding_layer_forward_propagation.set(samples_number, &bounding_layer);
    bounding_layer.forward_propagate({ input_pairs }, &bounding_layer_forward_propagation, is_training);

    outputs = bounding_layer_forward_propagation.outputs;

    assert_true(outputs(0) - type(-1.0) < type(NUMERIC_LIMITS_MIN), LOG);

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
