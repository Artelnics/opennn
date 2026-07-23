#include "pch.h"

#include "opennn/non_max_suppression_layer.h"
#include "opennn/neural_network.h"
#include "opennn/tensor_types.h"

using namespace opennn;

namespace {

constexpr float tol = 1e-5f;

void write_box(float* cell, Index box, Index values_per_box,
               float x_cell_rel, float y_cell_rel, float w_abs, float h_abs,
               float obj, initializer_list<float> class_probs)
{
    float* base = cell + box * values_per_box;
    base[0] = x_cell_rel;
    base[1] = y_cell_rel;
    base[2] = w_abs;
    base[3] = h_abs;
    base[4] = obj;
    Index c = 0;
    for (float p : class_probs) base[5 + c++] = p;
}

}

TEST(NonMaxSuppression, DefaultConstructorAndGeneralConstructor)
{
    NonMaxSuppression default_nms;
    EXPECT_EQ(default_nms.get_name(), "NonMaxSuppression");
    EXPECT_EQ(default_nms.get_output_shape(), Shape{});

    constexpr Index grid = 7;
    constexpr Index B = 2;
    constexpr Index C = 3;
    constexpr Index channels = B * (5 + C);
    const Shape input_shape{grid, grid, channels};

    NonMaxSuppression nms(input_shape, B, 0.6f, 0.3f, "detector");
    EXPECT_EQ(nms.get_name(), "NonMaxSuppression");
    EXPECT_EQ(nms.get_label(), "detector");
    EXPECT_EQ(nms.get_input_shape(), input_shape);
    EXPECT_EQ(nms.get_output_shape(), (Shape{grid * grid * B, 6}));
}

TEST(NonMaxSuppression, DecodesBoxCoordinatesToImageRelative)
{
    constexpr Index batch_size = 1;
    constexpr Index grid = 2;
    constexpr Index B = 1;
    constexpr Index C = 1;
    constexpr Index values_per_box = 5 + C;
    constexpr Index channels = B * values_per_box;
    constexpr Index max_boxes = grid * grid * B;
    constexpr float confidence_threshold = 0.5f;
    constexpr float iou_threshold = 0.4f;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<NonMaxSuppression>(
        Shape{grid, grid, channels},
        B,
        confidence_threshold,
        iou_threshold,
        "nms"));
    neural_network.compile();

    Tensor4 input(batch_size, grid, grid, channels);
    input.setZero();

    auto cell_ptr = [&](Index row, Index col) {
        return input.data() + ((0 * grid + row) * grid + col) * channels;
    };

    constexpr float x_cell_rel = 0.25f;
    constexpr float y_cell_rel = 0.75f;
    constexpr float w_abs = 0.30f;
    constexpr float h_abs = 0.40f;
    constexpr Index col = 1;
    constexpr Index row = 1;

    write_box(cell_ptr(row, col), 0, values_per_box,
              x_cell_rel, y_cell_rel, w_abs, h_abs,        0.9f, {1.0f});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input.data(), {batch_size, grid, grid, channels}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const float* out = forward_propagation.get_outputs().as<float>();

    EXPECT_NEAR(out[0 * 6 + 0], (float(col) + x_cell_rel) / float(grid), tol);
    EXPECT_NEAR(out[0 * 6 + 1], (float(row) + y_cell_rel) / float(grid), tol);
    EXPECT_NEAR(out[0 * 6 + 2], w_abs, tol);
    EXPECT_NEAR(out[0 * 6 + 3], h_abs, tol);
    EXPECT_NEAR(out[0 * 6 + 4], 0.9f, tol);
    EXPECT_EQ(Index(out[0 * 6 + 5]), 0);

    for (Index k = 1; k < max_boxes; ++k)
        for (Index v = 0; v < 6; ++v)
            EXPECT_FLOAT_EQ(out[k * 6 + v], 0.0f);
}

TEST(NonMaxSuppression, SuppressesSameClassOverlapKeepsDifferentClassOverlap)
{
    constexpr Index batch_size = 1;
    constexpr Index grid = 2;
    constexpr Index B = 2;
    constexpr Index C = 2;
    constexpr Index values_per_box = 5 + C;
    constexpr Index channels = B * values_per_box;
    constexpr Index max_boxes = grid * grid * B;
    constexpr float confidence_threshold = 0.5f;
    constexpr float iou_threshold = 0.4f;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<NonMaxSuppression>(
        Shape{grid, grid, channels},
        B,
        confidence_threshold,
        iou_threshold,
        "nms"));
    neural_network.compile();

    Tensor4 input(batch_size, grid, grid, channels);
    input.setZero();

    auto cell_ptr = [&](Index row, Index col) {
        return input.data() + ((0 * grid + row) * grid + col) * channels;
    };

    write_box(cell_ptr(0, 0), 0, values_per_box,
              0.1f, 0.1f, 0.4f, 0.4f,        0.9f, {0.8f, 0.2f});
    write_box(cell_ptr(0, 0), 1, values_per_box,
              0.1f, 0.1f, 0.4f, 0.4f,        0.85f, {0.75f, 0.25f});

    write_box(cell_ptr(1, 1), 0, values_per_box,
              0.5f, 0.5f, 0.4f, 0.4f,        0.95f, {0.1f, 0.9f});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input.data(), {batch_size, grid, grid, channels}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const float* out = output_view.as<float>();

    EXPECT_NEAR(out[0 * 6 + 4], 0.95f * 0.9f, tol);
    EXPECT_EQ(Index(out[0 * 6 + 5]), 1);

    EXPECT_NEAR(out[1 * 6 + 4], 0.9f * 0.8f, tol);
    EXPECT_EQ(Index(out[1 * 6 + 5]), 0);

    for (Index k = 2; k < max_boxes; ++k)
        for (Index v = 0; v < 6; ++v)
            EXPECT_FLOAT_EQ(out[k * 6 + v], 0.0f);
}

TEST(NonMaxSuppression, DropsBoxesBelowConfidenceThreshold)
{
    constexpr Index batch_size = 1;
    constexpr Index grid = 2;
    constexpr Index B = 1;
    constexpr Index C = 1;
    constexpr Index values_per_box = 5 + C;
    constexpr Index channels = B * values_per_box;
    constexpr Index max_boxes = grid * grid * B;
    constexpr float confidence_threshold = 0.5f;
    constexpr float iou_threshold = 0.4f;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<NonMaxSuppression>(
        Shape{grid, grid, channels},
        B,
        confidence_threshold,
        iou_threshold,
        "nms"));
    neural_network.compile();

    Tensor4 input(batch_size, grid, grid, channels);
    input.setZero();

    auto cell_ptr = [&](Index row, Index col) {
        return input.data() + ((0 * grid + row) * grid + col) * channels;
    };

    write_box(cell_ptr(0, 0), 0, values_per_box,
              0.5f, 0.5f, 0.3f, 0.3f,        0.4f, {0.9f});

    write_box(cell_ptr(1, 1), 0, values_per_box,
              0.5f, 0.5f, 0.3f, 0.3f,        0.6f, {1.0f});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input.data(), {batch_size, grid, grid, channels}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const float* out = forward_propagation.get_outputs().as<float>();

    EXPECT_NEAR(out[0 * 6 + 4], 0.6f, tol);
    EXPECT_EQ(Index(out[0 * 6 + 5]), 0);

    for (Index k = 1; k < max_boxes; ++k)
        EXPECT_FLOAT_EQ(out[k * 6 + 4], 0.0f);
}
