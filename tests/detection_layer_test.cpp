#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/detection_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"
#include "opennn/tensor_types.h"

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

    neural_network.add_layer(make_unique<Convolutional>(input_shape,
                                                        Shape{1, 1, 3, channels},
                                                        "Identity",
                                                        Shape{1, 1},
                                                        "Same",
                                                        false,
                                                        "logits"),
                             {-1});
    const Index conv_index = neural_network.get_layers_number() - 1;

    auto detection = make_unique<Detection>(neural_network.get_layer(conv_index)->get_output_shape(), anchors, "detection");
    detection->set_class_activation(Detection::ClassActivation::Sigmoid);
    neural_network.add_layer(std::move(detection), {conv_index});
    const Index detection_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(detection_index)->get_output_shape()),
                             {detection_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();

    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(0.1f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
