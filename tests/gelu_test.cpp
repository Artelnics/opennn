#include "pch.h"
#include "numerical_derivatives.h"

#include <cmath>

#include "opennn/tensor_types.h"
#include "opennn/tensor_operations.h"
#include "opennn/configuration.h"
#include "opennn/tabular_dataset.h"
#include "opennn/dense_layer.h"
#include "opennn/activation_layer.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

namespace
{
double gelu_ref(double x)
{
    return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
}
}

TEST(GeluTest, ForwardClosedForm)
{
    vector<float> x = {-4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f};
    vector<float> y = x;

    TensorView view(y.data(), {Index(y.size())});
    activation_forward(view, ActivationFunction::GELU);

    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], float(gelu_ref(x[i])), 1e-5f);
}

TEST(GeluTest, BackwardMatchesFiniteDifference)
{
    vector<float> x = {-4.f, -2.f, -1.f, -0.5f, -0.1f, 0.f, 0.1f, 0.5f, 1.f, 2.f, 4.f};

    for (float xi : x)
    {
        float xv = xi;
        float dv = 1.0f;                       // upstream gradient = 1
        TensorView xview(&xv, {1});
        TensorView dview(&dv, {1});
        activation_backward(xview, dview, ActivationFunction::GELU);  // dv <- gelu'(xi)

        const double h = 1e-3;
        const double numeric = (gelu_ref(xi + h) - gelu_ref(xi - h)) / (2.0 * h);

        EXPECT_NEAR(dv, float(numeric), 2e-3f) << "x = " << xi;
    }
}

TEST(GeluTest, ActivationLayerForward)
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

TEST(GeluTest, GradientCheckThroughActivationLayer)
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

TEST(GeluTest, DenseFusedGradientCheck)
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

TEST(GeluTest, DenseFusedRejectsBatchNorm)
{
    EXPECT_THROW(opennn::Dense(Shape{4}, Shape{5}, "GELU", true), std::exception);
}
