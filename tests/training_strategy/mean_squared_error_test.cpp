#include "tests/pch.h"
#include "opennn/core/random_utilities.h"
#include "tests/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include <iomanip>

using namespace opennn;

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_neural_network(), nullptr);
    EXPECT_EQ(loss.get_dataset(), nullptr);
}

TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    TabularDataset dataset;
    Loss loss(&neural_network, &dataset);

    EXPECT_NE(loss.get_neural_network(), nullptr);
    EXPECT_NE(loss.get_dataset(), nullptr);
}

TEST(MeanSquaredErrorTest, GpuWorkspaceIsForwardPropagationOwned)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);

    constexpr Index samples_number = 2;
    TabularDataset dataset(samples_number, {1}, {1});
    MatrixR data(samples_number, 2);
    data << 0.25f, 0.5f,
            0.75f, 1.0f;
    dataset.set_data(data);
    dataset.set_variable_indices({0}, {1});
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(
        make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    neural_network.compile(Device::CUDA);
    neural_network.get_parameters_map().setConstant(0.25f);

    // get_parameters_map writes the fp32 master; the device mirror the forward
    // pass actually reads is only updated here. Without this the mirror holds
    // whatever the allocator handed over, which is harmless only while that
    // happens to be zero.
    neural_network.copy_parameters_device();
    neural_network.copy_parameters_device();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    Batch batch(samples_number, &dataset, neural_network.get_config());
    batch.fill({0, 1}, {0}, {}, {1});
    batch.wait_h2d_on_compute_stream();

    ForwardPropagation first(samples_number, &neural_network);
    ForwardPropagation second(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_inputs(), first, false);
    neural_network.forward_propagate(batch.get_inputs(), second, false);

    const Loss::EvaluationResult first_result = loss.calculate_error(batch, first);
    const Loss::EvaluationResult second_result = loss.calculate_error(batch, second);

    ASSERT_FALSE(first.loss_workspace.empty());
    ASSERT_FALSE(second.loss_workspace.empty());
    EXPECT_NE(first.loss_workspace.data(), second.loss_workspace.data());
    EXPECT_FLOAT_EQ(first_result.error, second_result.error);
}

TEST(MeanSquaredErrorTest, BackPropagateDense2d)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ inputs_number }, Shape{ dataset.get_target_shape()}));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(MeanSquaredErrorTest, BackPropagateRecurrent)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(3, 10);
    const Index time_steps = random_integer(1, 10);

    TabularDataset dataset(samples_number, {time_steps, inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{targets_number}));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(MeanSquaredErrorTest, BackPropagateConvolutional)
{
    const Index samples_number = 6;
    const Index targets_number = 1;

    const Shape input_shape = { 21, 21, 3 };
    const Shape kernel_shape = { 3, 3, 3, 1 };

    TabularDataset dataset(samples_number, input_shape, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(input_shape, kernel_shape));
    const Shape flatten_layer_input_dimensions = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten>(flatten_layer_input_dimensions));
    const Shape dense_layer_input_dimensions = neural_network.get_layer(1)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(dense_layer_input_dimensions, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(MeanSquaredErrorTest, BackPropagatePooling)
{
    const Index samples_number = 6;
    const Index targets_number = 1;

    const Shape input_shape = { 21, 21, 3 };
    const Shape kernel_shape = { 3, 3, 3, 1 };

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Convolutional>(input_shape, kernel_shape));
    const Shape conv_output_dimensions = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Pooling>(conv_output_dimensions));
    const Shape pool_output_dimensions = neural_network.get_layer(1)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten>(pool_output_dimensions));
    const Shape flatten_output_dimensions = neural_network.get_layer(2)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(flatten_output_dimensions, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(MeanSquaredErrorTest, BackPropagateEmbedding)
{
    const Index samples_number = random_integer(5, 10);
    const Index inputs_number = random_integer(10, 20);
    const Index targets_number = random_integer(3, 10);

    const Index embeding_dim = inputs_number;
    const Index sequence_length = random_integer(1, 10);
    const Index flattened_size = sequence_length * embeding_dim;

    TabularDataset dataset(samples_number, { sequence_length }, { targets_number });
    dataset.set_data_integer(inputs_number);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Embedding>(Shape{ inputs_number, sequence_length }, embeding_dim));
    const Shape flatten_layer_input_dimensions = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten>(Shape{ flatten_layer_input_dimensions }));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ flattened_size }, Shape{ targets_number }));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);

    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    const type max_abs_diff = (gradient - numerical_gradient).array().abs().maxCoeff();
    const type gradient_scale = max(type(1), numerical_gradient.array().abs().maxCoeff());

    EXPECT_LT(max_abs_diff / gradient_scale, type(2.0e-2));
}

TEST(MeanSquaredErrorTest, BackPropagateMultiheadAttention)
{
    const Index batch_size = random_integer(1, 10);
    const Index sequence_length = random_integer(3, 10);
    const Index heads_number = random_integer(1, 10);
    const Index head_dimension = random_integer(1, 10);
    const Index embedding_dimension = heads_number * head_dimension;

    const Shape sample_input_dimensions = { sequence_length, embedding_dimension };

    const Shape sample_target_shape = { sequence_length * embedding_dimension };

    TabularDataset dataset(batch_size, sample_input_dimensions, sample_target_shape);
    dataset.set_data_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(dataset.get_input_shape(), heads_number));
    neural_network.add_layer(make_unique<Flatten>(neural_network.get_output_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    const VectorR analytical_gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_GE(error, 0.0) << "MSE must be positive";
    EXPECT_LT((analytical_gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
