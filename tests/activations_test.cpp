#include "pch.h"
#include "numerical_derivatives.h"

#include <cmath>

#include "opennn/tensor_types.h"
#include "opennn/tensor_operations.h"
#include "opennn/configuration.h"
#include "opennn/tabular_dataset.h"
#include "opennn/dense_layer.h"
#include "opennn/activation_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/activation_operator.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

namespace
{
double gelu_ref(double x)
{
    return 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));
}

double gelu_tanh_ref(double x)
{
    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}

double silu_ref(double x)
{
    return x / (1.0 + exp(-x));
}
}

TEST(ActivationsTest, DefaultConstructor)
{
    Activation activation_layer;

    EXPECT_EQ(activation_layer.get_name(), "Activation");
    EXPECT_EQ(activation_layer.get_output_activation(), ActivationFunction::ReLU);
}


TEST(ActivationsTest, GeneralConstructor)
{
    const Shape input_shape{ 5 };

    Activation activation_layer(input_shape, "Tanh", "act");

    EXPECT_EQ(activation_layer.get_input_shape(), input_shape);
    EXPECT_EQ(activation_layer.get_output_shape(), input_shape);
    EXPECT_EQ(activation_layer.get_label(), "act");
    EXPECT_EQ(activation_layer.get_output_activation(), ActivationFunction::Tanh);
}


TEST(ActivationsTest, ForwardPropagateReLU)
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


TEST(ActivationsTest, ForwardPropagateTanh)
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
        EXPECT_NEAR(output_data[i], tanh(input_data.data()[i]), 1e-6f);
}


TEST(ActivationsTest, ForwardPropagateSigmoid)
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
        const float expected = 1.0f / (1.0f + exp(-input_data.data()[i]));
        EXPECT_NEAR(output_data[i], expected, 1e-6f);
    }
}


TEST(ActivationsTest, ForwardPropagateSoftmax)
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

    const float e1 = exp(1.0f);
    const float e2 = exp(2.0f);
    const float e3 = exp(3.0f);
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


TEST(ActivationsTest, BackwardGradientMatchesNumerical)
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


TEST(ActivationsTest, LeakyReLUForwardPassesPositiveAndScalesNegative)
{
    VectorR buffer(5);
    buffer << -2.0f, -0.5f, 0.0f, 0.5f, 2.0f;
    TensorView view(buffer.data(), {buffer.size()});

    activation_forward(view, ActivationFunction::LeakyReLU);

    EXPECT_FLOAT_EQ(buffer[0], -0.2f);
    EXPECT_FLOAT_EQ(buffer[1], -0.05f);
    EXPECT_FLOAT_EQ(buffer[2],  0.0f);
    EXPECT_FLOAT_EQ(buffer[3],  0.5f);
    EXPECT_FLOAT_EQ(buffer[4],  2.0f);
}


TEST(ActivationsTest, LeakyReLUBackwardGatesByOutputSign)
{
    VectorR outputs(5);
    outputs << -0.2f, -0.05f, 0.0f, 0.5f, 2.0f;
    VectorR delta(5);
    delta << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f;

    TensorView outputs_view(outputs.data(), {outputs.size()});
    TensorView delta_view  (delta.data(),   {delta.size()});

    activation_backward(outputs_view, delta_view, ActivationFunction::LeakyReLU);

    EXPECT_FLOAT_EQ(delta[0], 0.1f);
    EXPECT_FLOAT_EQ(delta[1], 0.2f);
    EXPECT_FLOAT_EQ(delta[2], 3.0f);
    EXPECT_FLOAT_EQ(delta[3], 4.0f);
    EXPECT_FLOAT_EQ(delta[4], 5.0f);
}


TEST(ActivationsTest, GeluForwardClosedForm)
{
    VectorR x(11);
    x << -4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f;
    VectorR y = x;

    TensorView view(y.data(), {y.size()});
    activation_forward(view, ActivationFunction::GELU);

    for (Index i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], float(gelu_ref(x[i])), 1e-5f);
}


TEST(ActivationsTest, GeluBackwardMatchesFiniteDifference)
{
    vector<float> x = {-4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f};

    for (float xi : x)
    {
        float xv = xi;
        float dv = 1.0f;
        TensorView xview(&xv, {1});
        TensorView dview(&dv, {1});
        activation_backward(xview, dview, ActivationFunction::GELU);

        const double h = 1e-3;
        const double numeric = (gelu_ref(xi + h) - gelu_ref(xi - h)) / (2.0 * h);

        EXPECT_NEAR(dv, float(numeric), 2e-3f) << "x = " << xi;
    }
}


TEST(ActivationsTest, GeluTanhForwardClosedForm)
{
    VectorR x(11);
    x << -4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f;
    VectorR y = x;

    TensorView view(y.data(), {y.size()});
    activation_forward(view, ActivationFunction::GELUTanh);

    for (Index i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], float(gelu_tanh_ref(x[i])), 1e-5f);
}


TEST(ActivationsTest, GeluTanhBackwardMatchesFiniteDifference)
{
    vector<float> x = {-4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f};

    for (float xi : x)
    {
        float xv = xi;
        float dv = 1.0f;
        TensorView xview(&xv, {1});
        TensorView dview(&dv, {1});
        activation_backward(xview, dview, ActivationFunction::GELUTanh);

        const double h = 1e-3;
        const double numeric = (gelu_tanh_ref(xi + h) - gelu_tanh_ref(xi - h)) / (2.0 * h);

        EXPECT_NEAR(dv, float(numeric), 2e-3f) << "x = " << xi;
    }
}


TEST(ActivationsTest, GeluTanhFromString)
{
    EXPECT_EQ(ActivationFunction::GELUTanh, ActivationOperator::from_string("GELUTanh"));
}


TEST(ActivationsTest, GeluActivationLayerForward)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 2;
    const Index features = 5;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Activation>(Shape{features}, "GELU"));
    neural_network.compile();

    MatrixR input_data(batch_size, features);
    input_data << -2.0f, -0.5f, 0.0f, 0.5f, 2.0f,
                   1.0f, -1.0f, 3.0f, -3.0f, 0.25f;

    MatrixR result = neural_network.calculate_outputs(input_data);

    ASSERT_EQ(result.rows(), batch_size);
    ASSERT_EQ(result.cols(), features);

    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < features; ++j)
            EXPECT_NEAR(result(i, j), float(gelu_ref(input_data(i, j))), 1e-5f);

    Configuration::instance().set();
}


TEST(ActivationsTest, GeluGradientCheckThroughActivationLayer)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = 6;
    const Index inputs_number = 4;
    const Index hidden_number = 5;
    const Index targets_number = 3;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "Identity"));
    neural_network.add_layer(make_unique<opennn::Activation>(Shape{hidden_number}, "GELU"));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-3));

    Configuration::instance().set();
}


TEST(ActivationsTest, GeluDenseFusedGradientCheck)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = 6;
    const Index inputs_number  = 4;
    const Index hidden_number  = 5;
    const Index targets_number = 3;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELU"));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-3));

    Configuration::instance().set();
}


TEST(ActivationsTest, GeluDenseFusedRejectsBatchNorm)
{
    EXPECT_THROW(opennn::Dense(Shape{4}, Shape{5}, "GELU", true), exception);
}


TEST(ActivationsTest, ConvolutionalRejectsInputDerivativeActivations)
{
    EXPECT_THROW(Convolutional(Shape{8, 8, 1}, Shape{3, 3, 1, 2}, "GELU"), exception);
    EXPECT_THROW(Convolutional(Shape{8, 8, 1}, Shape{3, 3, 1, 2}, "GELUTanh"), exception);
    EXPECT_THROW(Convolutional(Shape{8, 8, 1}, Shape{3, 3, 1, 2}, "SiLU"), exception);

    Convolutional convolutional(Shape{8, 8, 1}, Shape{3, 3, 1, 2}, "ReLU");
    EXPECT_THROW(convolutional.set_activation_function("SiLU"), exception);
}


TEST(ActivationsTest, GeluTanhDenseGradientCheck)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = 6;
    const Index inputs_number  = 4;
    const Index hidden_number  = 8;
    const Index targets_number = 3;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELUTanh"));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-3));

    Configuration::instance().set();
}


TEST(ActivationsTest, SiluForwardClosedForm)
{
    VectorR x(11);
    x << -4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f;
    VectorR y = x;

    TensorView view(y.data(), {y.size()});
    activation_forward(view, ActivationFunction::SiLU);

    for (Index i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], float(silu_ref(x[i])), 1e-5f);
}


TEST(ActivationsTest, SiluBackwardMatchesFiniteDifference)
{
    vector<float> x = {-4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f};

    for (float xi : x)
    {
        float xv = xi;
        float dv = 1.0f;
        TensorView xview(&xv, {1});
        TensorView dview(&dv, {1});
        activation_backward(xview, dview, ActivationFunction::SiLU);

        const double h = 1e-3;
        const double numeric = (silu_ref(xi + h) - silu_ref(xi - h)) / (2.0 * h);

        EXPECT_NEAR(dv, float(numeric), 2e-3f) << "x = " << xi;
    }
}


TEST(ActivationsTest, SiluFromString)
{
    EXPECT_EQ(ActivationFunction::SiLU, ActivationOperator::from_string("SiLU"));
}


TEST(ActivationsTest, SiluActivationLayerForward)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 2;
    const Index features = 5;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Activation>(Shape{features}, "SiLU"));
    neural_network.compile();

    MatrixR input_data(batch_size, features);
    input_data << -2.0f, -0.5f, 0.0f, 0.5f, 2.0f,
                   1.0f, -1.0f, 3.0f, -3.0f, 0.25f;

    MatrixR result = neural_network.calculate_outputs(input_data);

    ASSERT_EQ(result.rows(), batch_size);
    ASSERT_EQ(result.cols(), features);

    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < features; ++j)
            EXPECT_NEAR(result(i, j), float(silu_ref(input_data(i, j))), 1e-5f);

    Configuration::instance().set();
}


TEST(ActivationsTest, SiluGradientCheckThroughActivationLayer)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = 6;
    const Index inputs_number = 4;
    const Index hidden_number = 5;
    const Index targets_number = 3;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "Identity"));
    neural_network.add_layer(make_unique<opennn::Activation>(Shape{hidden_number}, "SiLU"));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-3));

    Configuration::instance().set();
}


TEST(ActivationsTest, SiluDenseFusedGradientCheck)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = 6;
    const Index inputs_number  = 4;
    const Index hidden_number  = 5;
    const Index targets_number = 3;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "SiLU"));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-3));

    Configuration::instance().set();
}
