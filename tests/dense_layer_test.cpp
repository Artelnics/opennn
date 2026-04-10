#include "pch.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

using Dense2d = opennn::Dense<2>;
using Dense3d = opennn::Dense<3>;


TEST(Dense2dTest, DefaultConstructor)
{
    Dense2d dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense2d");
}


TEST(Dense2dTest, GeneralConstructor)
{
    Dense2d dense_layer({10}, {3}, "Linear");

    EXPECT_EQ(dense_layer.get_activation_function(), ActivationFunction::Linear);

    EXPECT_EQ(dense_layer.get_input_shape()[0], 10);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 3);
    // 10*3 weights + 3 biases = 33 raw, but aligned to 8-element boundary
    EXPECT_GE(dense_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(Shape{inputs_number}, Shape{outputs_number}, "Linear"));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(type(1.0));

    MatrixR result = neural_network.calculate_outputs(input_data);

    EXPECT_EQ(result.rows(), batch_size);
    EXPECT_EQ(result.cols(), outputs_number);

    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Dense2d");
}


TEST(Dense2dTest, BackPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(Shape{inputs_number}, Shape{outputs_number}, "Linear"));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(type(1.0));

    ForwardPropagation forward_propagation(batch_size, &neural_network);

    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], outputs_number);
}


TEST(Dense3dTest, DefaultConstructor)
{
    Dense3d dense_layer({1, 1}, {1});

    EXPECT_EQ(dense_layer.get_name(), "Dense3d");
    EXPECT_EQ(dense_layer.get_input_shape()[0], 1);
    EXPECT_EQ(dense_layer.get_input_shape()[1], 1);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 1);
    EXPECT_EQ(dense_layer.get_output_shape()[1], 1);
}


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = 5;
    const Index input_embedding = 4;
    const Index output_embedding = 3;

    Dense3d dense_layer({sequence_length, input_embedding}, {output_embedding}, "Linear");

    EXPECT_EQ(dense_layer.get_activation_function(), ActivationFunction::Linear);

    EXPECT_EQ(dense_layer.get_input_shape()[0], sequence_length);
    EXPECT_EQ(dense_layer.get_input_shape()[1], input_embedding);

    EXPECT_EQ(dense_layer.get_output_shape()[0], sequence_length);
    EXPECT_EQ(dense_layer.get_output_shape()[1], output_embedding);

    // input_embedding*output_embedding weights + output_embedding biases, aligned
    EXPECT_GE(dense_layer.get_parameters_number(), input_embedding * output_embedding + output_embedding);
}


TEST(Dense3dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index input_embedding = 4;
    const Index output_embedding = 5;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense3d>(Shape{sequence_length, input_embedding}, Shape{output_embedding}, "Linear"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 input_data(batch_size, sequence_length, input_embedding);
    input_data.setConstant(type(0.5));

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, sequence_length, input_embedding}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    EXPECT_EQ(output_view.shape[0], batch_size);
    // Output total elements = batch_size * sequence_length * output_embedding
    EXPECT_EQ(output_view.size(), batch_size * sequence_length * output_embedding);
}


TEST(Dense3dTest, BackPropagate)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index input_embedding = 4;
    const Index output_embedding = 5;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense3d>(Shape{sequence_length, input_embedding}, Shape{output_embedding}, "Linear"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 input_data(batch_size, sequence_length, input_embedding);
    input_data.setConstant(type(0.5));

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, sequence_length, input_embedding}) };
    neural_network.forward_propagate(inputs, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.size(), batch_size * sequence_length * output_embedding);
}
