#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/perceptron_layer_3d.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Dense3dTest, DefaultConstructor)
{
    Dense3d dense_3d;

    EXPECT_EQ(dense_3d.get_output_dimensions(), dimensions({0, 0}));
}


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = get_random_index(1, 10);
    const Index input_embedding = get_random_index(1, 10);
    const Index output_embedding = get_random_index(1, 10);

    Dense3d dense_3d(sequence_length, input_embedding, output_embedding);

    EXPECT_EQ(dense_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(dense_3d.get_input_embedding(), input_embedding);
    EXPECT_EQ(dense_3d.get_output_embedding(), output_embedding);
}


TEST(Dense3dTest, ForwardPropagate)
{
    const Index sequence_length = get_random_index(1, 10);
    const Index input_embedding = get_random_index(1, 10);
    const Index output_embedding = get_random_index(1, 10);
    const Index batch_size = get_random_index(1, 10);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense3d>(sequence_length, input_embedding, output_embedding));

/*
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

    outputs = tensor_map<2>(output_pair);

    EXPECT_NEAR(outputs(0), type(-1.0), NUMERIC_LIMITS_MIN);
    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{ 1 });
*/
}
