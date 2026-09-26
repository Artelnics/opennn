#include "tests/pch.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "tests/numerical_derivatives.h"

#include "opennn/training/loss.h"
#include "opennn/training/error_functions.h"
#include "opennn/training/training_context.h"
#include "opennn/core/tensor_types.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/convolutional_layer.h"
#include "opennn/network/network.h"
#include "opennn/dataset/batch.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/network/back_propagation.h"

using namespace opennn;

TEST(CrossEntropyError2d, DefaultConstructor)
{
    Loss loss;

    EXPECT_TRUE(loss.get_network() == nullptr);
    EXPECT_TRUE(loss.get_dataset() == nullptr);
}

TEST(CrossEntropyError2d, BackPropagate)
{
    const Index samples_number = random_integer(2, 10);

    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = 1;
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });

    dataset.set_data_random();

    dataset.set_sample_roles("Training");

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{ inputs_number }, Shape{ targets_number }, "Sigmoid"));
    network.compile();

    network.set_parameters_random();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    const VectorR gradient = calculate_gradient(loss);

    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));

    type difference = 0;
    for (int i = 0; i < gradient.size(); i++)
    {
        difference += ((gradient[i] - numerical_gradient[i]) * (gradient[i] - numerical_gradient[i]));
    }
    type error = std::sqrt(difference);

    EXPECT_NEAR(error,0, type(1.0e-1));

}

TEST(CrossEntropyError2d, CalculateError)
{
    TabularDataset dataset(5, { 3 }, { 1 });

    MatrixR data(5, 4);
    data << type(0), type(1), type(0), type(1),
            type(1), type(1), type(0), type(0),
            type(0), type(1), type(1), type(1),
            type(0), type(1), type(1), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);

    const vector<Index> input_features_indices = { 0, 1, 2 };
    const vector<Index> target_features_indices = { 3 };

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{ 3 }, Shape{ 1 }, "Sigmoid"));
    network.compile();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    Batch batch(5, &dataset, network.get_config());

    const vector<Index> training_indices = dataset.get_sample_indices("Training");
    batch.fill(training_indices,
               FeatureSelection{input_features_indices, {}, target_features_indices});

    ForwardPropagation forward_propagation(5, &network);
    network.forward_propagate(batch.get_inputs(), forward_propagation, ForwardPropagationMode::Inference);

    const type error = loss.calculate_error(batch, forward_propagation).error;

    EXPECT_FALSE(isnan(error));
    EXPECT_GE(error, type(0));
}

TEST(CrossEntropyError2d, get_name)
{
    MatrixR data;
    TabularDataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data << type(2), type(5), type(0), type(0),
            type(2), type(9), type(1), type(0),
            type(2), type(9), type(1), type(1),
            type(6), type(5), type(0), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);

    vector<Index> input_features_indices(2);
    input_features_indices = { 0,1 };

    vector<Index> target_features_indices(2);
    target_features_indices = { 2,3 };

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{ 2 }, Shape{ 2 }, "Sigmoid"));
    network.compile();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    string name = loss.get_name();

    EXPECT_EQ(name, "CrossEntropy");
}

namespace
{
struct RestoreBinaryLossConfiguration
{
    EffectiveConfig previous = Configuration::instance().resolve();
    ~RestoreBinaryLossConfiguration()
    {
        Configuration::instance().set(previous.device, previous.training_type);
    }
};

vector<float> binary_loss_host_values(const TensorView& view)
{
    vector<float> values(size_t(view.size()));
    if (view.get_device() == Device::CPU)
        std::copy_n(view.as<float>(), values.size(), values.data());
    else
        copy_device_to_host_float(view.get_data(), view.get_type(), view.size(),
            values.data(), device::get_compute_stream());
    return values;
}

void expect_saturated_binary_logits(Device device_kind, Type precision)
{
    RestoreBinaryLossConfiguration restore;
    Configuration::instance().set(device_kind, precision);
    TabularDataset dataset(4, {1}, {1});
    MatrixR data(4, 2);
    data << -20.0f, 1.0f, 20.0f, 0.0f, -1000.0f, 1.0f, 1000.0f, 0.0f;
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Sigmoid"));
    network.compile();
    auto* combination = dynamic_cast<CombinationOperator*>(network.get_layer(0)->get_operators()[0]);
    ASSERT_NE(combination, nullptr);
    {
        const Network::HostParametersGuard guard(network);
        combination->weights.as_vector().setOnes();
        combination->bias.setZero();
        network.mark_parameters_changed();
    }
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);
    loss.set_regularization(Loss::Regularization::NoRegularization);
    Batch batch(4, &dataset, network.get_config());
    batch.fill({0, 1, 2, 3}, dataset.get_feature_selection());
#ifdef OPENNN_HAS_CUDA
    if (device_kind == Device::CUDA)
    {
        batch.upload_to_device_batch_async(batch, device::get_compute_stream());
        batch.wait_h2d_on_compute_stream();
    }
#endif
    TrainingContext context(4, loss);
    ASSERT_FALSE(context.forward.binary_logits.empty());
    network.forward_propagate(batch.get_inputs(), context.forward, ForwardPropagationMode::Training);
    const auto probabilities = binary_loss_host_values(context.forward.get_outputs());
    EXPECT_NEAR(probabilities[0], 0.0f, 1.0e-7f);
    EXPECT_NEAR(probabilities[1], 1.0f, 1.0e-7f);
    EXPECT_EQ(probabilities[2], 0.0f);
    EXPECT_NEAR(probabilities[3], 1.0f, 1.0e-7f);
    loss.back_propagate(batch, context.forward, context.backward);
    EXPECT_NEAR(context.backward.metrics.loss_value, 510.0f, 1.0e-4f);
    EXPECT_NEAR(binary_loss_host_values(combination->weight_gradient)[0], 510.0f, 1.0e-4f);
    EXPECT_NEAR(binary_loss_host_values(combination->bias_gradient)[0], 0.0f, 1.0e-6f);

    InferenceShapePolicy validation_policy;
    validation_policy.capture_binary_logits = true;
    ForwardPropagation validation(4, &network, ForwardPropagationMode::Inference, validation_policy);
    network.forward_propagate(batch.get_inputs(), validation, ForwardPropagationMode::Inference);
    EXPECT_NEAR(loss.calculate_error(batch, validation).error, 510.0f, 1.0e-4f);

    ForwardPropagation inference(4, &network, ForwardPropagationMode::Inference);
    EXPECT_TRUE(inference.binary_logits.empty());
    EXPECT_EQ(inference.binary_logits_storage.byte_size(), 0);
    network.forward_propagate(batch.get_inputs(), inference, ForwardPropagationMode::Inference);
    const auto regular_probabilities = binary_loss_host_values(inference.get_outputs());
    for (size_t i = 0; i < probabilities.size(); ++i)
        EXPECT_NEAR(regular_probabilities[i], probabilities[i], 1.0e-7f);

    TrainingContext tail(2, loss, false, &context);
    EXPECT_FALSE(tail.forward.binary_logits_storage.owns_memory());
    EXPECT_EQ(tail.forward.binary_logits_storage.data(), context.forward.binary_logits_storage.data());

#ifdef OPENNN_HAS_CUDA
    if (device_kind == Device::CUDA)
    {
        Buffer metric(Device::CUDA);
        metric.resize_bytes(sizeof(float), Device::CUDA);
        metric.setZero();
        network.forward_propagate(batch.get_inputs(), context.forward, ForwardPropagationMode::Training);
        ASSERT_TRUE(loss.back_propagate_device_metrics(batch, context.forward, context.backward,
            metric.as<float>(), nullptr));
        EXPECT_NEAR(binary_loss_host_values(TensorView(metric.data(), {1}, Type::FP32, Device::CUDA))[0],
            510.0f, 1.0e-4f);
        EXPECT_NEAR(binary_loss_host_values(combination->weight_gradient)[0], 510.0f, 1.0e-4f);
    }
#endif

    // Correct predictions must retain their tiny gradients after sigmoid
    // rounds to one: at +8 in BF16 and at +20 even in FP32.
    for (const float magnitude : {8.0f, 20.0f})
    {
        data << -magnitude, 0.0f, magnitude, 1.0f, -magnitude, 0.0f, magnitude, 1.0f;
        dataset.set_data(data);
        dataset.set_sample_roles("Training");
        batch.fill({0, 1, 2, 3}, dataset.get_feature_selection());
#ifdef OPENNN_HAS_CUDA
        if (device_kind == Device::CUDA)
        {
            batch.upload_to_device_batch_async(batch, device::get_compute_stream());
            batch.wait_h2d_on_compute_stream();
        }
#endif
        network.forward_propagate(batch.get_inputs(), context.forward, ForwardPropagationMode::Training);
        loss.back_propagate(batch, context.forward, context.backward);
        const float expected_loss = log1p(exp(-magnitude));
        const float expected_gradient = -magnitude / (1.0f + exp(magnitude));
        const float actual_gradient = binary_loss_host_values(combination->weight_gradient)[0];
        EXPECT_NEAR(context.backward.metrics.loss_value, expected_loss, expected_loss * 1.0e-5f);
        EXPECT_LT(actual_gradient, 0.0f);
        EXPECT_NEAR(actual_gradient, expected_gradient,
            abs(expected_gradient) * (precision == Type::BF16 ? 0.01f : 1.0e-5f));
    }
}
}

TEST(CrossEntropyError2d, SaturatedSigmoidTrainingKeepsLogitsLossAndGradient)
{
    expect_saturated_binary_logits(Device::CPU, Type::FP32);
}

TEST(CrossEntropyError2d, BinaryLogitsCaptureRequiresEligibleHeadAndLoss)
{
    RestoreBinaryLossConfiguration restore;
    Configuration::instance().set(Device::CPU, Type::FP32);
    for (const bool normalize : {false, true})
    {
        Network network;
        auto head = make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Sigmoid",
            normalize ? BatchNormalization::Yes : BatchNormalization::No);
        if (!normalize) head->set_dropout_rate(0.2f);
        network.add_layer(std::move(head));
        network.compile();
        Loss loss(&network);
        loss.set_error(Loss::Error::CrossEntropy);
        TrainingContext context(4, loss);
        EXPECT_TRUE(context.forward.binary_logits.empty());
    }
    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Sigmoid"));
    network.compile();
    Loss mse(&network);
    mse.set_error(Loss::Error::MeanSquaredError);
    TrainingContext unrelated(4, mse);
    EXPECT_TRUE(unrelated.forward.binary_logits.empty());
}

TEST(CrossEntropyError2d, DirectProbabilityLossRetainsItsContract)
{
    MatrixR probabilities(2, 1), targets(2, 1);
    probabilities << 0.2f, 0.8f;
    targets << 1.0f, 0.0f;
    float error = 0.0f;
    binary_cross_entropy(TensorView(probabilities.data(), {2, 1}),
        TensorView(targets.data(), {2, 1}), error, nullptr);
    EXPECT_NEAR(error, -log(0.2f + EPSILON), 1.0e-6f);
}

#ifdef OPENNN_HAS_CUDA
TEST(CrossEntropyError2d, SaturatedSigmoidTrainingKeepsLogitsLossAndGradientCUDA)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    expect_saturated_binary_logits(Device::CUDA, Type::FP32);
}

TEST(CrossEntropyError2d, SaturatedSigmoidTrainingKeepsLogitsLossAndGradientBF16CUDA)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    expect_saturated_binary_logits(Device::CUDA, Type::BF16);
}
#endif
