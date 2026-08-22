#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <utility>

#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"

using namespace opennn;

namespace {

constexpr float tol = 1e-5f;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}

TEST(Detection, ConstructorInfersClassesNumber)
{
    const Index grid = 4;
    const Index B = 2;
    const Index C = 3;
    const Index channels = B * (5 + C);

    const vector<std::array<float, 2>> anchors{{0.2f, 0.2f}, {0.5f, 0.5f}};

    Detection layer(Shape{grid, grid, channels}, anchors, "detection");

    EXPECT_EQ(layer.get_output_shape(), (Shape{grid, grid, channels}));
    ASSERT_EQ(layer.get_anchors().size(), size_t(B));
    EXPECT_FLOAT_EQ(layer.get_anchors()[0][0], 0.2f);
    EXPECT_FLOAT_EQ(layer.get_anchors()[1][1], 0.5f);
}

TEST(Detection, DefaultConstructorAndClassActivationDefault)
{
    Detection default_layer;

    EXPECT_EQ(default_layer.get_output_shape(), (Shape{}));
    EXPECT_EQ(default_layer.get_class_activation(), Detection::ClassActivation::Softmax);

    default_layer.set_class_activation(Detection::ClassActivation::Sigmoid);
    EXPECT_EQ(default_layer.get_class_activation(), Detection::ClassActivation::Sigmoid);

    default_layer.set_class_activation(Detection::ClassActivation::Softmax);
    EXPECT_EQ(default_layer.get_class_activation(), Detection::ClassActivation::Softmax);
}

TEST(Detection, ImplementsDetectionHeadContract)
{
    const vector<std::array<float, 2>> anchors{{0.2f, 0.3f}, {0.4f, 0.5f}};
    Detection anchored(Shape{2, 2, 16}, anchors);
    anchored.set_class_activation(Detection::ClassActivation::Sigmoid);
    DetectionV8 anchor_free(Shape{2, 2, 35}, 8);

    const DetectionHeadMetadata anchored_metadata =
        anchored.get_detection_head_metadata();
    EXPECT_EQ(anchored_metadata.kind, DetectionHeadKind::AnchorBased);
    EXPECT_EQ(anchored_metadata.boxes_per_cell, 2);
    EXPECT_EQ(anchored_metadata.classes_number, 3);
    EXPECT_EQ(anchored_metadata.regression_bins, 1);
    EXPECT_EQ(anchored_metadata.class_activation,
              DetectionClassActivation::Sigmoid);

    const DetectionHeadMetadata anchor_free_metadata =
        anchor_free.get_detection_head_metadata();
    EXPECT_EQ(anchor_free_metadata.kind, DetectionHeadKind::AnchorFree);
    EXPECT_EQ(anchor_free_metadata.classes_number, 3);
    EXPECT_EQ(anchor_free_metadata.regression_bins, 8);
    EXPECT_EQ(anchor_free_metadata.class_activation,
              DetectionClassActivation::Sigmoid);
}

TEST(Detection, ForwardPropagateMatchesHandComputedValuesForKnownLogits)
{
    const Index batch_size = 1;
    const Index grid = 1;
    const Index B = 1;
    const Index C = 3;
    const Index values_per_box = 5 + C;
    const Index channels = B * values_per_box;

    const vector<std::array<float, 2>> anchors{{0.5f, 1.0f}};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Detection>(Shape{grid, grid, channels}, anchors, "detection"));
    neural_network.compile();

    Tensor4 input(batch_size, grid, grid, channels);
    const float logits[] = {1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 0.0f, 1.0f, 2.0f};
    for (Index i = 0; i < channels; ++i)
        input.data()[i] = logits[i];

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input.data(), {batch_size, grid, grid, channels}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const float* out = forward_propagation.get_outputs().as<float>();

    EXPECT_NEAR(out[0], sigmoid(1.0f), tol);
    EXPECT_NEAR(out[1], sigmoid(-1.0f), tol);
    EXPECT_NEAR(out[2], std::exp(0.5f) * anchors[0][0], tol);
    EXPECT_NEAR(out[3], std::exp(-0.5f) * anchors[0][1], tol);
    EXPECT_NEAR(out[4], sigmoid(2.0f), tol);

    const float e0 = std::exp(0.0f - 2.0f);
    const float e1 = std::exp(1.0f - 2.0f);
    const float e2 = std::exp(2.0f - 2.0f);
    const float s = e0 + e1 + e2;
    EXPECT_NEAR(out[5], e0 / s, tol);
    EXPECT_NEAR(out[6], e1 / s, tol);
    EXPECT_NEAR(out[7], e2 / s, tol);

    float class_sum = out[5] + out[6] + out[7];
    EXPECT_NEAR(class_sum, 1.0f, tol);
}

TEST(Detection, ForwardPropagateSigmoidClassActivation)
{
    const Index batch_size = 1;
    const Index grid = 1;
    const Index B = 1;
    const Index C = 3;
    const Index values_per_box = 5 + C;
    const Index channels = B * values_per_box;

    const vector<std::array<float, 2>> anchors{{0.5f, 1.0f}};

    auto detection = make_unique<Detection>(Shape{grid, grid, channels}, anchors, "detection");
    detection->set_class_activation(Detection::ClassActivation::Sigmoid);

    NeuralNetwork neural_network;
    neural_network.add_layer(std::move(detection));
    neural_network.compile();

    Tensor4 input(batch_size, grid, grid, channels);
    const float logits[] = {1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 0.0f, 1.0f, 2.0f};
    for (Index i = 0; i < channels; ++i)
        input.data()[i] = logits[i];

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input.data(), {batch_size, grid, grid, channels}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const float* out = forward_propagation.get_outputs().as<float>();

    EXPECT_NEAR(out[0], sigmoid(1.0f), tol);
    EXPECT_NEAR(out[1], sigmoid(-1.0f), tol);
    EXPECT_NEAR(out[2], std::exp(0.5f) * anchors[0][0], tol);
    EXPECT_NEAR(out[3], std::exp(-0.5f) * anchors[0][1], tol);
    EXPECT_NEAR(out[4], sigmoid(2.0f), tol);

    EXPECT_NEAR(out[5], sigmoid(0.0f), tol);
    EXPECT_NEAR(out[6], sigmoid(1.0f), tol);
    EXPECT_NEAR(out[7], sigmoid(2.0f), tol);

    const float class_sum = out[5] + out[6] + out[7];
    EXPECT_GT(class_sum, 1.0f + tol);
}

TEST(Detection, GpuAnchorsRefreshAfterSameSizeReconfiguration)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    constexpr Index batch_size = 1;
    constexpr Index grid = 1;
    constexpr Index channels = 6;
    const Shape input_shape{grid, grid, channels};

    Configuration::instance().set(Device::CUDA, Type::FP32);

    auto detection = make_unique<Detection>(
        input_shape,
        vector<std::array<float, 2>>{{0.25f, 0.5f}},
        "detection");
    Detection* const detection_ptr = detection.get();

    NeuralNetwork neural_network;
    neural_network.add_layer(std::move(detection));
    neural_network.compile(Device::CUDA);

    vector<float> input(size_t(channels), 0.0f);
    Buffer device_input(Device::CUDA);
    const Index input_bytes = channels * Index(sizeof(float));
    device_input.resize_bytes(input_bytes, Device::CUDA);
    device::copy_async(device_input.data(), input.data(), input_bytes,
                       device::CopyKind::HostToDevice);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    const vector<TensorView> inputs{
        TensorView(device_input.data(),
                   {batch_size, grid, grid, channels},
                   Type::FP32,
                   Device::CUDA)};

    const auto forward_and_read = [&]
    {
        neural_network.forward_propagate(inputs, forward_propagation, false);

        vector<float> output(static_cast<size_t>(channels), 0.0f);
        device::copy_async(output.data(),
                           forward_propagation.get_outputs().get_data(),
                           input_bytes,
                           device::CopyKind::DeviceToHost);
        device::synchronize(device::get_compute_stream());
        return output;
    };

    const vector<float> first = forward_and_read();

    detection_ptr->set(input_shape,
                       vector<std::array<float, 2>>{{0.75f, 1.25f}},
                       "detection");
    const vector<float> second = forward_and_read();

    EXPECT_NEAR(first[2], 0.25f, tol);
    EXPECT_NEAR(first[3], 0.5f, tol);
    EXPECT_NEAR(second[2], 0.75f, tol);
    EXPECT_NEAR(second[3], 1.25f, tol);
}

TEST(Detection, SigmoidClassBackwardGradientMatchesNumerical)
{
    const Index samples_number = 4;
    const Index grid = 2;
    const Index B = 1;
    const Index C = 2;
    const Index channels = B * (5 + C);
    const Index targets_number = 2;

    const Shape input_shape{grid, grid, 3};

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    const vector<std::array<float, 2>> anchors{{0.5f, 0.5f}};

    NeuralNetwork neural_network;

    const Index conv_index = neural_network.add_layer(make_unique<Convolutional>(input_shape,
                                                                                 Shape{1, 1, 3, channels},
                                                                                 "Identity",
                                                                                 Shape{1, 1},
                                                                                 "Same",
                                                                                 false,
                                                                                 "logits"),
                                                      {-1});

    auto detection = make_unique<Detection>(neural_network.get_layer(conv_index)->get_output_shape(), anchors, "detection");
    detection->set_class_activation(Detection::ClassActivation::Sigmoid);
    const Index detection_index = neural_network.add_layer(std::move(detection), {conv_index});

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(detection_index)->get_output_shape()),
                             {detection_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();

    neural_network.get_parameters_map().setConstant(0.1f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
