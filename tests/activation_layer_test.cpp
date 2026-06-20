#include "pch.h"
#include "numerical_derivatives.h"

#include "../opennn/tensor_types.h"
#include "../opennn/tensor_operations.h"
#include "../opennn/activation_layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/loss.h"

using namespace opennn;


TEST(ActivationLayerTest, DefaultConstructor)
{
    Activation activation_layer;

    EXPECT_EQ(activation_layer.get_name(), "Activation");
    EXPECT_EQ(activation_layer.get_output_activation(), ActivationFunction::ReLU);
}


TEST(ActivationLayerTest, GeneralConstructor)
{
    const Shape input_shape{ 5 };

    Activation activation_layer(input_shape, "Tanh", "act");

    EXPECT_EQ(activation_layer.get_input_shape(), input_shape);
    EXPECT_EQ(activation_layer.get_output_shape(), input_shape);
    EXPECT_EQ(activation_layer.get_label(), "act");
    EXPECT_EQ(activation_layer.get_output_activation(), ActivationFunction::Tanh);
}


TEST(ActivationLayerTest, ForwardPropagateReLU)
{
    const Index batch_size = 2;
    const Index features = 3;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Activation>(Shape{ features }, "ReLU", "act"));
    neural_network.compile();

    MatrixR input_data(batch_size, features);
    input_data(0, 0) = type(-2.0); input_data(0, 1) = type(0.0); input_data(0, 2) = type(3.0);
    input_data(1, 0) = type(1.5);  input_data(1, 1) = type(-0.5); input_data(1, 2) = type(-4.0);

    MatrixR expected(batch_size, features);
    expected(0, 0) = type(0.0); expected(0, 1) = type(0.0); expected(0, 2) = type(3.0);
    expected(1, 0) = type(1.5); expected(1, 1) = type(0.0); expected(1, 2) = type(0.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, features}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], features);
    ASSERT_EQ(output_view.size(), batch_size * features);

    const type* output_data = output_view.as<type>();
    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_data[i], expected.data()[i], 1e-6f);
}


TEST(ActivationLayerTest, ForwardPropagateTanh)
{
    const Index batch_size = 2;
    const Index features = 3;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Activation>(Shape{ features }, "Tanh", "act"));
    neural_network.compile();

    MatrixR input_data(batch_size, features);
    input_data(0, 0) = type(-1.0); input_data(0, 1) = type(0.0); input_data(0, 2) = type(2.0);
    input_data(1, 0) = type(0.5);  input_data(1, 1) = type(-0.25); input_data(1, 2) = type(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, features}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], features);
    ASSERT_EQ(output_view.size(), batch_size * features);

    const type* output_data = output_view.as<type>();
    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_data[i], std::tanh(input_data.data()[i]), 1e-6f);
}


TEST(ActivationLayerTest, ForwardPropagateSigmoid)
{
    const Index batch_size = 1;
    const Index features = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Activation>(Shape{ features }, "Sigmoid", "act"));
    neural_network.compile();

    MatrixR input_data(batch_size, features);
    input_data(0, 0) = type(0.0); input_data(0, 1) = type(2.0);
    input_data(0, 2) = type(-2.0); input_data(0, 3) = type(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, features}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.size(), batch_size * features);

    const type* output_data = output_view.as<type>();
    for (Index i = 0; i < output_view.size(); ++i)
    {
        const float expected = 1.0f / (1.0f + std::exp(-input_data.data()[i]));
        EXPECT_NEAR(output_data[i], expected, 1e-6f);
    }
}


TEST(ActivationLayerTest, ForwardPropagateSoftmax)
{
    const Index batch_size = 2;
    const Index features = 3;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Activation>(Shape{ features }, "Softmax", "act"));
    neural_network.compile();

    MatrixR input_data(batch_size, features);
    input_data(0, 0) = type(1.0); input_data(0, 1) = type(2.0); input_data(0, 2) = type(3.0);
    input_data(1, 0) = type(0.0); input_data(1, 1) = type(0.0); input_data(1, 2) = type(0.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, features}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], features);
    ASSERT_EQ(output_view.size(), batch_size * features);

    const type* output_data = output_view.as<type>();

    const float e1 = std::exp(1.0f);
    const float e2 = std::exp(2.0f);
    const float e3 = std::exp(3.0f);
    const float denom = e1 + e2 + e3;

    EXPECT_NEAR(output_data[0], e1 / denom, 1e-6f);
    EXPECT_NEAR(output_data[1], e2 / denom, 1e-6f);
    EXPECT_NEAR(output_data[2], e3 / denom, 1e-6f);

    EXPECT_NEAR(output_data[3], 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(output_data[4], 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(output_data[5], 1.0f / 3.0f, 1e-6f);

    for (Index i = 0; i < batch_size; ++i)
    {
        float row_sum = 0.0f;
        for (Index j = 0; j < features; ++j)
            row_sum += output_data[i * features + j];
        EXPECT_NEAR(row_sum, 1.0f, 1e-6f);
    }
}


TEST(ActivationLayerTest, BackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index inputs_number = 4;
    const Index features = 3;
    const Index targets_number = 3;

    const Shape input_shape{ inputs_number };

    TabularDataset dataset(samples_number, input_shape, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<opennn::Dense>(input_shape, Shape{ features }, "Identity"),
                             { -1 });
    const Index dense_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Activation>(Shape{ features }, "Tanh", "act"),
                             { dense_index });

    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
