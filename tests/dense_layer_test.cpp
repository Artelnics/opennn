#include "pch.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Dense2dTest, DefaultConstructor)
{
    opennn::Dense<2> dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense2d");
    EXPECT_EQ(dense_layer.get_input_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_output_shape().size(), 1);
}


TEST(Dense2dTest, GeneralConstructor)
{
    opennn::Dense<2> dense_layer({10}, {3}, "Linear");

    EXPECT_EQ(dense_layer.get_activation_function(), "Linear");

    ASSERT_EQ(dense_layer.get_input_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_input_shape()[0], 10);

    ASSERT_EQ(dense_layer.get_output_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 3);

    EXPECT_EQ(dense_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;
    const bool is_training = true;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_number}, Shape{outputs_number}, "Linear"));
    neural_network.set_parameters_random();

    opennn::ForwardPropagation forward_propagation(batch_size, &neural_network);

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(type(1.0));

    TensorView input_view(input_data.data(), {batch_size, inputs_number});
    vector<TensorView> input_views = {input_view};

    ASSERT_NO_THROW(neural_network.forward_propagate(input_views, forward_propagation, is_training));

    EXPECT_EQ(neural_network.get_layers()[0]->get_name(), "Dense2d");

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.size(), 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], outputs_number);
}


TEST(Dense3dTest, DefaultConstructor)
{
    opennn::Dense<3> dense_3d({1, 1}, {1});

    EXPECT_EQ(dense_3d.get_name(), "Dense3d");
    EXPECT_EQ(dense_3d.get_input_shape().size(), 1);
    EXPECT_EQ(dense_3d.get_output_shape().size(), 1);
}


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = 5;
    const Index input_embedding = 4;
    const Index output_embedding = 3;

    opennn::Dense<3> dense_3d({sequence_length, input_embedding}, {output_embedding}, "HyperbolicTangent");

    const Shape input_dims = dense_3d.get_input_shape();
    const Shape output_dims = dense_3d.get_output_shape();

    EXPECT_EQ(input_dims[0], sequence_length);
    EXPECT_EQ(output_dims[0], output_embedding);
    EXPECT_EQ(dense_3d.get_name(), "Dense3d");
}

/*
TEST(Dense3dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index input_embedding = 4;
    const Index output_embedding = 5;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<3>>(
        Shape{sequence_length, input_embedding}, Shape{output_embedding}));
    neural_network.set_parameters_random();

    opennn::ForwardPropagation forward_propagation(batch_size, &neural_network);

    MatrixR input_data(batch_size * sequence_length, input_embedding);
    input_data.setConstant(type(0.5));

    TensorView input_view(input_data.data(), {batch_size, sequence_length, input_embedding});
    vector<TensorView> input_views = {input_view};

    ASSERT_NO_THROW(neural_network.forward_propagate(input_views, forward_propagation, false));

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.size(), 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], sequence_length);
    EXPECT_EQ(output_view.shape[2], output_embedding);
}*/
