// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "tests/common/pch.h"
#include "tests/common/numerical_derivatives.h"

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/network.h"
#include "opennn/training/loss.h"

using namespace opennn;

namespace
{

// Six samples of three classes in proportion 4:1:1, whose balanced weights are 0.5, 2 and 2.
void set_imbalanced_data(TabularDataset& dataset)
{
    MatrixR data(6, 5);
    data <<  0.3f, -0.6f, 1.0f, 0.0f, 0.0f,
            -0.8f,  0.2f, 1.0f, 0.0f, 0.0f,
             0.5f,  0.9f, 1.0f, 0.0f, 0.0f,
            -0.1f, -0.4f, 1.0f, 0.0f, 0.0f,
             0.7f, -0.3f, 0.0f, 1.0f, 0.0f,
            -0.5f,  0.8f, 0.0f, 0.0f, 1.0f;

    dataset.set_data(data);
    dataset.set_sample_roles("Training");
}

}

TEST(WeightedCrossEntropyTest, CalculateError)
{
    TabularDataset dataset(6, {2}, {3});
    set_imbalanced_data(dataset);

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Softmax"));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::WeightedCrossEntropy);
    loss.set_normalization_coefficient();

    const MatrixR outputs = network.calculate_outputs(MatrixR(dataset.get_data().leftCols(2)));
    const float class_weights[] = {0.5f, 2.0f, 2.0f};
    const Index labels[] = {0, 0, 0, 0, 1, 2};

    float expected_error = 0.0f;
    for (Index i = 0; i < 6; ++i)
        expected_error -= class_weights[labels[i]] * log(outputs(i, labels[i]) + EPSILON) / 6.0f;

    EXPECT_NEAR(calculate_numerical_error(loss), expected_error, 1.0e-5f);
}

TEST(WeightedCrossEntropyTest, BackPropagate)
{
    TabularDataset dataset(6, {2}, {3});
    set_imbalanced_data(dataset);

    ClassificationNetwork network({2}, {4}, {3});
    network.set_parameters_random();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::WeightedCrossEntropy);
    loss.set_normalization_coefficient();

    const VectorR analytical_gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((analytical_gradient - numerical_gradient).array().abs().maxCoeff(), 1.0e-3f);
}

#ifdef OPENNN_HAS_CUDA
TEST(WeightedCrossEntropyTest, GpuMatchesCpu)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    const ScopeExit reset_configuration([] { Configuration::instance().set(Device::CPU, Type::FP32); });

    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(6, {2}, {3});
    set_imbalanced_data(dataset);

    ClassificationNetwork cpu_network({2}, {4}, {3});
    cpu_network.set_parameters_random();

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::WeightedCrossEntropy);
    cpu_loss.set_normalization_coefficient();

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ClassificationNetwork gpu_network({2}, {4}, {3});
    gpu_network.set_parameters(cpu_network.get_parameters_map());

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::WeightedCrossEntropy);
    gpu_loss.set_normalization_coefficient();

    EXPECT_NEAR(calculate_numerical_error(gpu_loss), calculate_numerical_error(cpu_loss), 1.0e-5f);
    EXPECT_LT((calculate_gradient(gpu_loss) - calculate_gradient(cpu_loss)).array().abs().maxCoeff(), 1.0e-5f);
}
#endif
