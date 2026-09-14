#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <utility>

#include "opennn/core/tensor_types.h"
#include "opennn/core/configuration.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/network/layers/embedding_layer.h"
#include "opennn/network/layers/flatten_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/network.h"
#include "opennn/training/loss.h"

using namespace opennn;

TEST(LearnedPositionalTest, EmbeddingGradientCheck)
{
    const Index samples_number  = 6;
    const Index vocabulary_size = 12;
    const Index sequence_length = 5;
    const Index embedding_dim   = 8;
    const Index targets_number  = 3;
    const Index flattened       = sequence_length * embedding_dim;

    TabularDataset dataset(samples_number, { sequence_length }, { targets_number });
    dataset.set_data_integer(vocabulary_size);
    dataset.set_sample_roles("Training");

    auto embedding = make_unique<Embedding>(Shape{vocabulary_size, sequence_length}, embedding_dim);
    embedding->set_learned_positional(true);
    ASSERT_TRUE(embedding->get_learned_positional());

    Network network;
    network.add_layer(std::move(embedding));
    network.add_layer(make_unique<Flatten>(network.get_layer(0)->get_output_shape()));
    network.add_layer(make_unique<opennn::Dense>(Shape{flattened}, Shape{targets_number}));
    network.compile();
    network.set_parameters_random();

    EXPECT_GT(network.get_parameters_number(),
              vocabulary_size * embedding_dim + flattened * targets_number + targets_number);

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-2));
}

TEST(LearnedPositionalTest, CudaPrecisionPreservesTokenIdsAndMaskedPositionGradients)
{
    if (!device::has_cuda_device() || device::cuda_compute_capability() < 80)
        GTEST_SKIP() << "An Ampere or newer CUDA device is required.";
    const ScopeExit reset_configuration([]
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
    });

    // Token IDs have the same exact-input contract as the language datasets.
    class TokenIdDataset final : public TabularDataset
    {
    public:
        using TabularDataset::TabularDataset;
        bool supports_bf16_inputs() const override { return false; }
    };
    constexpr Index samples = 2;
    constexpr Index vocabulary_size = 258;
    constexpr Index sequence_length = 4;
    constexpr Index embedding_dimension = 4;
    Configuration::instance().set(Device::CPU, Type::FP32);
    TokenIdDataset dataset(samples, Shape{sequence_length}, Shape{1});
    MatrixR data(samples, sequence_length + 1);
    data << 257.0f, 0.0f, 258.0f, -1.0f, 0.0f,
            257.0f, 257.0f, 0.0f, 258.0f, 0.25f;
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
    const MatrixR inputs = data.leftCols(sequence_length);

    const auto build = [=](Network& network)
    {
        auto embedding = make_unique<Embedding>(Shape{vocabulary_size, sequence_length}, embedding_dimension);
        embedding->set_learned_positional(true);
        embedding->set_weights_follow_compute_dtype(true);
        network.add_layer(std::move(embedding));
        network.add_layer(make_unique<Flatten>(network.get_output_shape()));
        network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(), Shape{1}, "Identity"));
        network.compile();
    };
    Network cpu;
    build(cpu);
    auto& embedding_parameters = cpu.get_layer(0)->get_parameter_views();
    ASSERT_EQ(embedding_parameters.size(), 2);
    MatrixMap table = embedding_parameters[0].as_matrix();
    table.setConstant(0.125f);
    table.row(0).setZero();
    table.row(256).setConstant(0.25f);
    table.row(257).setConstant(0.5f); // Rounding the identifier to BF16 must not select row 256.
    MatrixMap positional = embedding_parameters[1].as_matrix();
    for (Index position = 0; position < sequence_length; ++position)
        positional.row(position).setConstant(0.125f * float(position + 1));
    auto& dense_parameters = cpu.get_layer(2)->get_parameter_views();
    ASSERT_EQ(dense_parameters.size(), 2);
    dense_parameters[0].as_vector().setConstant(0.125f);
    dense_parameters[1].as_vector().setConstant(0.0625f);
    const VectorR parameters = cpu.get_parameters_map();
    cpu.set_parameters(parameters);

    const Index table_offset = embedding_parameters[0].as<float>() - cpu.get_parameters_data();
    const Index positional_offset = embedding_parameters[1].as<float>() - cpu.get_parameters_data();
    const MatrixR expected_outputs = cpu.calculate_outputs(inputs);
    EXPECT_FLOAT_EQ(expected_outputs(0, 0), 0.28125f);
    EXPECT_FLOAT_EQ(expected_outputs(1, 0), 0.46875f);
    Loss cpu_loss(&cpu, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR expected_gradient = calculate_gradient(cpu_loss);
    ASSERT_GT(expected_gradient.segment(table_offset + 257 * embedding_dimension, embedding_dimension).norm(), 0.0f);
    ASSERT_GT(expected_gradient.segment(positional_offset, 2 * embedding_dimension).norm(), 0.0f);

    const auto expect_masked_gradients = [&](const VectorR& gradient)
    {
        EXPECT_TRUE(gradient.segment(table_offset, embedding_dimension).isZero());
        EXPECT_TRUE(gradient.segment(table_offset + 256 * embedding_dimension, embedding_dimension).isZero());
        // Positions 2 and 3 contain only padding, negative IDs, or OOV IDs.
        EXPECT_TRUE(gradient.segment(positional_offset + 2 * embedding_dimension, 2 * embedding_dimension).isZero());
    };
    expect_masked_gradients(expected_gradient);

    for (const Type precision : {Type::FP32, Type::BF16})
    {
        SCOPED_TRACE(int(precision));
        Configuration::instance().set(Device::CUDA, precision);
        Network gpu;
        build(gpu);
        gpu.set_parameters(parameters);
        const MatrixR actual_outputs = gpu.calculate_outputs(inputs);
        ASSERT_EQ(actual_outputs.rows(), expected_outputs.rows());
        ASSERT_EQ(actual_outputs.cols(), expected_outputs.cols());
        EXPECT_LT((actual_outputs - expected_outputs).cwiseAbs().maxCoeff(), 1.0e-4f);
        Loss gpu_loss(&gpu, &dataset);
        gpu_loss.set_error(Loss::Error::MeanSquaredError);
        const VectorR actual_gradient = calculate_gradient(gpu_loss);
        ASSERT_EQ(actual_gradient.size(), expected_gradient.size());
        EXPECT_TRUE(actual_gradient.allFinite());
        EXPECT_LT((actual_gradient - expected_gradient).cwiseAbs().maxCoeff(), 1.0e-4f);
        expect_masked_gradients(actual_gradient);
    }
}
