#include "pch.h"

#include "../opennn/bounding_layer.h"


TEST(BoundingTest, Constructor) 
{
    Bounding bounding_layer;

    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{0});
}


TEST(BoundingTest, ForwardPropagate)
{
    Bounding bounding_layer({1});

    bounding_layer.set_lower_bound(0, type(-1.0));
    bounding_layer.set_upper_bound(0, type(1));
    bounding_layer.set_bounding_method("Bounding");

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<BoundingForwardPropagation>(1, &bounding_layer);

    Tensor<type, 2> inputs(1, 1);
    inputs.setConstant(-2.0);
    Tensor<type, 2> outputs(1, 1);

    const pair<type*, dimensions> input_pairs = { inputs.data(), {{1, 1}} };



    bounding_layer.forward_propagate({ input_pairs },
                                          forward_propagation,
                                          true);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_NEAR(outputs(0), type(-1.0), NUMERIC_LIMITS_MIN);
    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{ 1 });

}
