#include "pch.h"

#include "../opennn/detection_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/tensor_utilities.h"

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

TEST(Detection, ForwardPropagateAppliesSigmoidExpAndSoftmax)
{
    const Index batch_size = 1;
    const Index grid = 2;
    const Index B = 2;
    const Index C = 2;
    const Index values_per_box = 5 + C;
    const Index channels = B * values_per_box;

    const vector<std::array<float, 2>> anchors{{0.3f, 0.4f}, {0.6f, 0.8f}};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Detection>(Shape{grid, grid, channels}, anchors, "detection"));
    neural_network.compile();

    Tensor4 input(batch_size, grid, grid, channels);
    input.setZero();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input.data(), {batch_size, grid, grid, channels}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], grid);
    EXPECT_EQ(output_view.shape[2], grid);
    EXPECT_EQ(output_view.shape[3], channels);

    const float* out = output_view.as<float>();

    for (Index row = 0; row < grid; ++row)
        for (Index col = 0; col < grid; ++col)
        {
            const Index cell = ((0 * grid + row) * grid + col) * channels;

            for (Index box = 0; box < B; ++box)
            {
                const Index base = cell + box * values_per_box;

                // Sigmoid(0) = 0.5 for x, y, objectness.
                EXPECT_NEAR(out[base + 0], 0.5f, tol);
                EXPECT_NEAR(out[base + 1], 0.5f, tol);
                EXPECT_NEAR(out[base + 4], 0.5f, tol);

                // exp(0) * anchor[w/h] = anchor[w/h].
                EXPECT_NEAR(out[base + 2], anchors[size_t(box)][0], tol);
                EXPECT_NEAR(out[base + 3], anchors[size_t(box)][1], tol);

                // Softmax over uniform logits => 1/C.
                float class_sum = 0.0f;
                for (Index c = 0; c < C; ++c)
                {
                    EXPECT_NEAR(out[base + 5 + c], 1.0f / float(C), tol);
                    class_sum += out[base + 5 + c];
                }
                EXPECT_NEAR(class_sum, 1.0f, tol);
            }
        }
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

    // Softmax over [0, 1, 2].
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
