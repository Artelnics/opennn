#include "tests/common/pch.h"
#include "opennn/core/random_utilities.h"
#include "tests/common/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training/loss.h"
#include "opennn/models/models.h"

using namespace opennn;

TEST(MinkowskiErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_network() == nullptr, true);
    EXPECT_EQ(loss.get_dataset() == nullptr, true);
}

TEST(MinkowskiErrorTest, GeneralConstructor)
{
    Network network;
    TabularDataset dataset;

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MinkowskiError);

    EXPECT_EQ(loss.get_network() != nullptr, true);
    EXPECT_EQ(loss.get_dataset() != nullptr, true);
}

TEST(MinkowskiErrorTest, BackPropagate)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index outputs_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({ inputs_number }, { neurons_number }, { outputs_number });

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MinkowskiError);

    const VectorR gradient = calculate_gradient(loss);

    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(5.0e-2));
}

// Minkowski was the only error without a CUDA implementation. The same network
// must give the same error and gradient on the GPU as on the CPU for each power
// in use, p = 1 included, where the gradient is the sign of the residual.
TEST(MinkowskiErrorTest, GpuMatchesCpu)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    const ScopeExit reset_configuration([] { Configuration::instance().set(Device::CPU, Type::FP32); });

    constexpr Index samples_number = 6;
    TabularDataset dataset(samples_number, {3}, {2});
    MatrixR data(samples_number, 5);
    data << -0.8f,  0.4f,  0.1f,  0.3f, -1.2f,
             0.3f, -0.9f,  0.7f, -0.5f,  0.2f,
             1.2f,  0.1f, -0.4f,  0.9f,  0.6f,
            -0.2f,  0.6f,  0.5f, -1.1f, -0.3f,
             0.7f, -0.5f, -0.9f,  0.4f,  1.0f,
             0.0f,  0.8f,  0.3f, -0.7f,  0.5f;
    dataset.set_data(data);
    dataset.set_variable_indices({0, 1, 2}, {3, 4});
    dataset.set_sample_roles("Training");

    for (const float power : {1.0f, 1.5f, 2.0f, 3.0f})
    {
        SCOPED_TRACE(power);

        float errors[2];
        VectorR gradients[2];

        for (const Device device : {Device::CPU, Device::CUDA})
        {
            const size_t index = device == Device::CUDA ? 1 : 0;

            Configuration::instance().set(device, Type::FP32);

            Network network;
            network.add_layer(make_unique<opennn::Dense>(Shape{3}, Shape{2}, "Identity"));
            network.compile(device);
            network.get_parameters_map().setLinSpaced(-0.4f, 0.6f);

            if (device == Device::CUDA)
                network.copy_parameters_device();

            Loss loss(&network, &dataset);
            loss.set_error(Loss::Error::MinkowskiError);
            loss.set_minkowski_parameter(power);
            loss.set_regularization(Loss::Regularization::NoRegularization);

            errors[index] = calculate_numerical_error(loss);
            gradients[index] = calculate_gradient(loss);
        }

        ASSERT_TRUE(std::isfinite(errors[0]));
        ASSERT_GT(errors[0], 0.0f);
        EXPECT_NEAR(errors[1], errors[0], 1.0e-5f * max(1.0f, errors[0]));

        ASSERT_EQ(gradients[1].size(), gradients[0].size());
        EXPECT_LT((gradients[1] - gradients[0]).cwiseAbs().maxCoeff(),
                  1.0e-5f * max(1.0f, gradients[0].cwiseAbs().maxCoeff()));
    }
}
