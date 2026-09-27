#include "tests/pch.h"

#ifndef OPENNN_NO_VISION

#include "opennn/training/loss.h"

#include <cmath>
#include <vector>

using namespace opennn;

TEST(YoloLossCheck, GradientMatchesFiniteDifferences)
{
    // A 2x2 grid with two anchor boxes of 8 values each: x, y, w, h, objectness, three classes.
    constexpr Index boxes_per_cell = 2;
    constexpr Index classes_number = 3;
    const Shape head_shape{1, 2, 2, 16};

    vector<float> prediction(64, 0.5f);
    vector<float> target(64, 0.0f);
    vector<float> gradient(64, 0.0f);

    // Slot 0 of cell (0, 0) holds the only foreground box (class 0); the other seven are background.
    const vector<float> foreground_prediction{0.52f, 0.53f, 0.22f, 0.17f, 0.70f, 0.80f, 0.15f, 0.05f};
    const vector<float> foreground_target{0.5f, 0.5f, 0.2f, 0.15f, 1.0f, 1.0f};
    std::copy(foreground_prediction.begin(), foreground_prediction.end(), prediction.begin());
    std::copy(foreground_target.begin(), foreground_target.end(), target.begin());

    const TensorView prediction_view(prediction.data(), head_shape, Type::FP32);
    const TensorView target_view(target.data(), head_shape, Type::FP32);
    const TensorView gradient_view(gradient.data(), head_shape, Type::FP32);

    const YoloLambdas weights{.giou = 5.0f,
                              .dfl = 0.5f,
                              .noobj = 2.0f,
                              .cls = 0.0f,
                              .focal_gamma = 0.0f,
                              .obj_focal_gamma = 0.0f};

    yolo_gradient_kernel(prediction_view, target_view, gradient_view,
                         boxes_per_cell, classes_number, true, 1.0f, weights);

    const auto error_at = [&]()
    {
        return yolo_error_kernel(prediction_view, target_view,
                                 boxes_per_cell, classes_number, true, weights);
    };

    constexpr float step = 1e-4f;
    float largest_relative_error = 0.0f;
    size_t largest_index = 0;

    for (size_t k = 0; k < prediction.size(); ++k)
    {
        const float original = prediction[k];

        prediction[k] = original + step;
        const float error_forward = error_at();

        prediction[k] = original - step;
        const float error_backward = error_at();

        prediction[k] = original;

        const float numerical = (error_forward - error_backward) / (2.0f * step);
        const float analytical = gradient[k];
        const float relative_error = std::abs(numerical - analytical)
                                   / (std::max(std::abs(numerical), std::abs(analytical)) + 1e-8f);

        if (relative_error > largest_relative_error)
        {
            largest_relative_error = relative_error;
            largest_index = k;
        }
    }

    EXPECT_LT(largest_relative_error, 1e-2f)
        << "Analytic YOLO gradient departs from the central difference at flat index " << largest_index;
}

TEST(YoloLossCheck, ForwardMatchesExpectedValues)
{

    // A 1x1 grid with one box of 7 values: x, y, w, h, objectness, two classes.
    // On a 1x1 grid the normalized box coordinates are the raw x and y values.
    constexpr Index boxes_per_cell = 1;
    constexpr Index classes_number = 2;
    const Shape box_shape{1, 1, 1, 7};
    constexpr float epsilon = 1e-7f;

    const YoloLambdas weights{.giou = 5.0f,
                              .dfl = 0.5f,
                              .noobj = 2.0f,
                              .cls = 0.0f,
                              .focal_gamma = 0.0f,
                              .obj_focal_gamma = 0.0f};

    const auto binary_cross_entropy = [&](float p, float t)
    {
        return -(t * std::log(p + epsilon) + (1.0f - t) * std::log(1.0f - p + epsilon));
    };

    const auto kernel_error = [&](vector<float> prediction, vector<float> target)
    {
        const TensorView prediction_view(prediction.data(), box_shape, Type::FP32);
        const TensorView target_view(target.data(), box_shape, Type::FP32);
        return yolo_error_kernel(prediction_view, target_view,
                                 boxes_per_cell, classes_number, true, weights);
    };

    // Objectness gradient mapped from the post-sigmoid confidence c to logit space.
    const auto objectness_logit_gradient = [&](vector<float> prediction, vector<float> target)
    {
        vector<float> gradient(prediction.size(), 0.0f);
        const TensorView prediction_view(prediction.data(), box_shape, Type::FP32);
        const TensorView target_view(target.data(), box_shape, Type::FP32);
        const TensorView gradient_view(gradient.data(), box_shape, Type::FP32);
        yolo_gradient_kernel(prediction_view, target_view, gradient_view,
                             boxes_per_cell, classes_number, true, 1.0f, weights);
        const float confidence = prediction[4];
        return gradient[4] * confidence * (1.0f - confidence);
    };

    // Case A: identical boxes, so the coordinate term vanishes.
    const vector<float> identical_prediction{0.5f, 0.5f, 0.4f, 0.3f, 0.7f, 0.8f, 0.2f};
    const vector<float> identical_target{0.5f, 0.5f, 0.4f, 0.3f, 1.0f, 1.0f, 0.0f};

    const float identical_expected = binary_cross_entropy(0.7f, 1.0f)
                                   + weights.cls * (binary_cross_entropy(0.8f, 1.0f)
                                                    + binary_cross_entropy(0.2f, 0.0f));
    const float identical_error = kernel_error(identical_prediction, identical_target);

    EXPECT_NEAR(identical_error, identical_expected, 1e-4f)
        << "Case A (identical boxes): the YOLO error should reduce to the confidence and class terms.";

    // Case B: disjoint boxes of equal size, 0.8 apart along x.
    const vector<float> disjoint_prediction{0.1f, 0.5f, 0.1f, 0.1f, 0.5f, 0.6f, 0.4f};
    const vector<float> disjoint_target{0.9f, 0.5f, 0.1f, 0.1f, 1.0f, 1.0f, 0.0f};

    const float iou = 0.0f;
    const float union_area = 0.02f;
    const float enclosing_area = 0.90f * 0.10f;
    const float giou = iou - (enclosing_area - union_area) / enclosing_area;
    const float centre_distance_squared = 0.8f * 0.8f;
    const float enclosing_diagonal_squared = 0.90f * 0.90f + 0.10f * 0.10f + epsilon;
    const float ciou = giou - centre_distance_squared / enclosing_diagonal_squared;   // v = 0: equal aspect ratios

    const float disjoint_expected = weights.giou * (1.0f - ciou)
                                  + binary_cross_entropy(0.5f, 1.0f)
                                  + weights.cls * (binary_cross_entropy(0.6f, 1.0f)
                                                   + binary_cross_entropy(0.4f, 0.0f));
    const float disjoint_error = kernel_error(disjoint_prediction, disjoint_target);

    EXPECT_NEAR(disjoint_error, disjoint_expected, 1e-4f)
        << "Case B (disjoint boxes): the CIoU coordinate term does not match its closed form.";

    // Case C: a low foreground confidence must be pushed up.
    vector<float> low_confidence_prediction = identical_prediction;
    low_confidence_prediction[4] = 0.1f;

    const float foreground_gradient = objectness_logit_gradient(low_confidence_prediction, identical_target);

    EXPECT_LT(foreground_gradient, 0.0f)
        << "Case C (foreground objectness): the logit gradient should be negative, got " << foreground_gradient;

    // Case D: a high background confidence must be pushed down.
    vector<float> background_prediction(7, 0.0f);
    background_prediction[4] = 0.9f;
    const vector<float> background_target(7, 0.0f);

    const float background_gradient = objectness_logit_gradient(background_prediction, background_target);

    EXPECT_GT(background_gradient, 0.0f)
        << "Case D (background objectness): the logit gradient should be positive, got " << background_gradient;

    const float foreground_sign_failure = foreground_gradient < 0.0f ? 0.0f : 1.0f;
    const float background_sign_failure = background_gradient > 0.0f ? 0.0f : 1.0f;

    const float worst_deviation = std::max({std::abs(identical_error - identical_expected),
                                            std::abs(disjoint_error - disjoint_expected),
                                            foreground_sign_failure,
                                            background_sign_failure});

    EXPECT_LT(worst_deviation, 1e-4f)
        << "Summary of cases A to D: at least one closed-form value or gradient sign is wrong.";
}

#endif
