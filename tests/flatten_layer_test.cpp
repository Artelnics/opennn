#include "pch.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

class FlattenLayerTest : public ::testing::Test
{
protected:
    const Index height = 4;
    const Index width = 4;
    const Index channels = 3;
    const Shape input_shape{ height, width, channels };

    unique_ptr<Flatten<4>> flatten_layer;

    void SetUp() override
    {
        flatten_layer = make_unique<Flatten<4>>(input_shape);
    }
};


TEST_F(FlattenLayerTest, Constructor)
{
    EXPECT_EQ(flatten_layer->get_input_shape(), input_shape);
    EXPECT_EQ(flatten_layer->get_output_shape(), Shape{ height * width * channels });
    EXPECT_EQ(flatten_layer->get_name(), "Flatten4d");
}


TEST_F(FlattenLayerTest, ForwardPropagate)
{
    const Index batch_size = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Flatten<4>>(input_shape));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(1.23f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(), {batch_size, height, width, channels}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const Shape& output_dims = output_view.shape;

    ASSERT_EQ(output_dims.rank, 2);
    EXPECT_EQ(output_dims[0], batch_size);
    EXPECT_EQ(output_dims[1], height * width * channels);

    for(Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], 1.23f, 1e-6f);
}


TEST_F(FlattenLayerTest, BackPropagate)
{
    const Index batch_size = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Flatten<4>>(input_shape));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(1.0f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(), {batch_size, height, width, channels}) };
    neural_network.forward_propagate(input_views, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], height * width * channels);

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], 1.0f, 1e-7f);
}
