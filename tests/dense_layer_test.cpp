#include "pch.h"
#include "../opennn/tensor_types.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Dense2dTest, DefaultConstructor)
{
    opennn::Dense dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense");
}


TEST(Dense2dTest, GeneralConstructor)
{
    opennn::Dense dense_layer({10}, {3}, "Identity");

    EXPECT_EQ(dense_layer.get_activation_function(), ActivationOp::Function::Identity);

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
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(type(1.0));

    MatrixR result = neural_network.calculate_outputs(input_data);

    EXPECT_EQ(result.rows(), batch_size);
    EXPECT_EQ(result.cols(), outputs_number);

    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Dense");
}


TEST(Dense2dTest, BackPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity"));
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
    opennn::Dense dense_layer({1, 1}, {1});

    EXPECT_EQ(dense_layer.get_name(), "Dense");
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

    opennn::Dense dense_layer({sequence_length, input_embedding}, {output_embedding}, "Identity");

    EXPECT_EQ(dense_layer.get_activation_function(), ActivationOp::Function::Identity);

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
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{sequence_length, input_embedding}, Shape{output_embedding}, "Identity"));
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
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{sequence_length, input_embedding}, Shape{output_embedding}, "Identity"));
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


TEST(ActivationTest, LeakyReLUStringMapRoundTrip)
{
    EXPECT_EQ(activation_function_from_string("LeakyReLU"), ActivationFunction::LeakyReLU);
    EXPECT_EQ(activation_function_to_string(ActivationFunction::LeakyReLU), "LeakyReLU");
}


TEST(ActivationTest, LeakyReLUForwardPassesPositiveAndScalesNegative)
{
    // f(x) = x         if x >= 0
    //      = slope * x if x <  0      (slope = LEAKY_RELU_SLOPE = 0.1)
    vector<float> buffer = { -2.0f, -0.5f, 0.0f, 0.5f, 2.0f };
    TensorView view(buffer.data(), {Index(buffer.size())});

    activation_forward(view, ActivationFunction::LeakyReLU);

    EXPECT_FLOAT_EQ(buffer[0], -0.2f);  // -2.0 * 0.1
    EXPECT_FLOAT_EQ(buffer[1], -0.05f); // -0.5 * 0.1
    EXPECT_FLOAT_EQ(buffer[2],  0.0f);  // gate is >= 0
    EXPECT_FLOAT_EQ(buffer[3],  0.5f);
    EXPECT_FLOAT_EQ(buffer[4],  2.0f);
}


TEST(ActivationTest, LeakyReLUBackwardGatesByOutputSign)
{
    // f'(y) = 1        if y >= 0  (because positive slope preserves sign of x)
    //       = slope    if y <  0
    // Inputs: outputs (post-activation), incoming delta.
    vector<float> outputs = { -0.2f, -0.05f, 0.0f, 0.5f, 2.0f };
    vector<float> delta   = {  1.0f,   2.0f, 3.0f, 4.0f, 5.0f };

    TensorView outputs_view(outputs.data(), {Index(outputs.size())});
    TensorView delta_view  (delta.data(),   {Index(delta.size())});

    activation_backward(outputs_view, delta_view, ActivationFunction::LeakyReLU);

    EXPECT_FLOAT_EQ(delta[0], 0.1f);  // negative side: 1.0 * 0.1
    EXPECT_FLOAT_EQ(delta[1], 0.2f);  // negative side: 2.0 * 0.1
    EXPECT_FLOAT_EQ(delta[2], 3.0f);  // zero counts as positive side
    EXPECT_FLOAT_EQ(delta[3], 4.0f);
    EXPECT_FLOAT_EQ(delta[4], 5.0f);
}


TEST(ActivationTest, LeakyReLUFlowsThroughDenseLayer)
{
    // End-to-end smoke test: a Dense layer constructed with "LeakyReLU"
    // resolves the string, propagates through compile(), and produces output
    // of the expected shape. Validates the wiring beyond the kernel itself.
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{4}, Shape{3}, "LeakyReLU"));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input(2, 4);
    input.setConstant(type(1.0));

    const MatrixR output = neural_network.calculate_outputs(input);
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 3);
}
