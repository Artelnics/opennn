#include "tests/pch.h"
#include "opennn/core/random_utilities.h"
#include "tests/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/pooling_layer.h"
#include "opennn/network/layers/convolutional_layer.h"
#include "opennn/network/network.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/training/loss.h"
#include "opennn/network/layers/recurrent_layer.h"
#include "opennn/network/layers/flatten_layer.h"
#include "opennn/network/layers/embedding_layer.h"
#include "opennn/network/layers/multihead_attention_layer.h"
#include <iomanip>

using namespace opennn;

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_network(), nullptr);
    EXPECT_EQ(loss.get_dataset(), nullptr);
}

TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    Network network;
    TabularDataset dataset;
    Loss loss(&network, &dataset);

    EXPECT_NE(loss.get_network(), nullptr);
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

    Network network;
    network.add_layer(
        make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    network.compile(Device::CUDA);
    network.get_parameters_map().setConstant(0.25f);

    // get_parameters_map writes the fp32 master; the device mirror the forward
    // pass actually reads is only updated here. Without this the mirror holds
    // whatever the allocator handed over, which is harmless only while that
    // happens to be zero.
    network.copy_parameters_device();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    Batch batch(samples_number, &dataset, network.get_config());
    batch.fill({0, 1}, FeatureSelection{{0}, {}, {1}});
    batch.upload_to_device_batch_async(batch, device::get_transfer_stream());
    batch.wait_h2d_on_compute_stream();

    ForwardPropagation first(samples_number, &network);
    ForwardPropagation second(samples_number, &network);
    network.forward_propagate(batch.get_inputs(), first, ForwardPropagationMode::Inference);
    network.forward_propagate(batch.get_inputs(), second, ForwardPropagationMode::Inference);

    const Loss::EvaluationResult first_result = loss.calculate_error(batch, first);
    const Loss::EvaluationResult second_result = loss.calculate_error(batch, second);

    ASSERT_FALSE(first.loss_workspace.empty());
    ASSERT_FALSE(second.loss_workspace.empty());
    EXPECT_NE(first.loss_workspace.data(), second.loss_workspace.data());
    EXPECT_FLOAT_EQ(first_result.error, second_result.error);
}

TEST(LossDeviceMetricsTest, MatchesHostMetricsAndAccumulatesAcrossBatches)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    const ScopeExit reset_configuration([] { Configuration::instance().set(Device::CPU, Type::FP32); });
    const vector<Loss::Error> errors{
        Loss::Error::MeanAbsoluteError, Loss::Error::MeanSquaredError,
        Loss::Error::NormalizedSquaredError, Loss::Error::WeightedSquaredError,
        Loss::Error::CrossEntropy};
    const vector<vector<Index>> sample_batches{{0, 3, 1}, {2, 4}};

    for (const Type precision : {Type::FP32, Type::BF16})
    {
        if (precision == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        SCOPED_TRACE(precision == Type::FP32 ? "FP32" : "BF16");
        Configuration::instance().set(Device::CUDA, precision);

        for (const Index targets_number : {Index(1), Index(3)})
        {
            SCOPED_TRACE(targets_number);
            TabularDataset dataset(5, {2}, {targets_number});
            MatrixR data = MatrixR::Zero(5, 2 + targets_number);
            data.col(0) << -0.8f, 0.3f, 1.2f, -0.2f, 0.7f;
            data.col(1) << 0.4f, -0.9f, 0.1f, 0.6f, -0.5f;
            for (Index row = 0; row < 5; ++row)
            {
                if (targets_number == 1)
                    data(row, 2) = row == 3 ? 1.0f : 0.0f;
                else
                    data(row, 2 + row % targets_number) = 1.0f;
            }
            dataset.set_data(data);
            dataset.set_sample_roles("Training");

            Network network;
            network.add_layer(make_unique<opennn::Dense>(
                Shape{2}, Shape{targets_number}, targets_number == 1 ? "Sigmoid" : "Softmax"));
            network.compile(Device::CUDA);
            network.get_parameters_map().setLinSpaced(-0.4f, 0.6f);
            network.copy_parameters_device();

            Loss loss(&network, &dataset);
            Buffer accumulated_error(Device::CUDA);
            float* const error_sum_device = accumulated_error.ensure<float>(1);
            const DeviceStream stream = device::get_compute_stream();

            for (const Loss::Error error : errors)
            {
                loss.set_error(error);
                SCOPED_TRACE(loss.get_name());
                // Five training rows versus batches of three and two exercise
                // normalization; the 4:1 binary imbalance gives unequal weights.
                loss.set_normalization_coefficient();
                device::set_zero_async(error_sum_device, sizeof(float), stream);
                float expected_sum = 0.0f;

                for (const vector<Index>& samples : sample_batches)
                {
                    SCOPED_TRACE(samples.size());
                    Batch batch(Index(samples.size()), &dataset, network.get_config());
                    batch.fill(samples, dataset.get_feature_selection());
                    batch.upload_to_device_batch_async(batch, device::get_transfer_stream());
                    batch.wait_h2d_on_compute_stream();

                    MatrixR uploaded_targets(Index(samples.size()), targets_number);
                    const TensorView& targets = batch.get_targets();
                    copy_device_to_host_float(targets.get_data(), targets.get_type(), targets.size(),
                                              uploaded_targets.data(), stream);
                    device::synchronize(stream);
                    for (Index row = 0; row < Index(samples.size()); ++row)
                        for (Index column = 0; column < targets_number; ++column)
                            EXPECT_FLOAT_EQ(uploaded_targets(row, column), data(samples[size_t(row)], 2 + column));

                    ForwardPropagation forward(Index(samples.size()), &network);
                    network.forward_propagate(batch.get_inputs(), forward, ForwardPropagationMode::Inference);

                    const float expected = loss.calculate_error(batch, forward).error;
                    ASSERT_TRUE(std::isfinite(expected));
                    ASSERT_GT(expected, 0.0f);
                    expected_sum += expected;
                    ASSERT_TRUE(loss.calculate_error_device_metrics(batch, forward, error_sum_device, nullptr));

                    float actual_sum = 0.0f;
                    copy_device_to_host_float(error_sum_device, Type::FP32, 1, &actual_sum, stream);
                    device::synchronize(stream);
                    EXPECT_NEAR(actual_sum, expected_sum, 2.0e-6f * max(1.0f, expected_sum));
                }
            }
        }
    }
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

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{ inputs_number }, Shape{ dataset.get_target_shape()}));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
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

    Network network;
    network.add_layer(make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{targets_number}));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
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

    Network network;
    network.add_layer(make_unique<Convolutional>(input_shape, kernel_shape));
    const Shape flatten_layer_input_dimensions = network.get_layer(0)->get_output_shape();
    network.add_layer(make_unique<Flatten>(flatten_layer_input_dimensions));
    const Shape dense_layer_input_dimensions = network.get_layer(1)->get_output_shape();
    network.add_layer(make_unique<opennn::Dense>(dense_layer_input_dimensions, dataset.get_target_shape()));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
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

    Network network;

    network.add_layer(make_unique<Convolutional>(input_shape, kernel_shape));
    const Shape conv_output_dimensions = network.get_layer(0)->get_output_shape();
    network.add_layer(make_unique<Pooling>(conv_output_dimensions));
    const Shape pool_output_dimensions = network.get_layer(1)->get_output_shape();
    network.add_layer(make_unique<Flatten>(pool_output_dimensions));
    const Shape flatten_output_dimensions = network.get_layer(2)->get_output_shape();
    network.add_layer(make_unique<opennn::Dense>(flatten_output_dimensions, dataset.get_target_shape()));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
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

    Network network;

    network.add_layer(make_unique<Embedding>(Shape{ inputs_number, sequence_length }, embeding_dim));
    const Shape flatten_layer_input_dimensions = network.get_layer(0)->get_output_shape();
    network.add_layer(make_unique<Flatten>(Shape{ flatten_layer_input_dimensions }));
    network.add_layer(make_unique<opennn::Dense>(Shape{ flattened_size }, Shape{ targets_number }));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
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

    Network network;
    network.add_layer(make_unique<MultiHeadAttention>(dataset.get_input_shape(), heads_number));
    network.add_layer(make_unique<Flatten>(network.get_output_shape()));
    network.compile();
    network.set_parameters_random();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    const VectorR analytical_gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_GE(error, 0.0) << "MSE must be positive";
    EXPECT_LT((analytical_gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
