#include "pch.h"

#include "opennn/yolo_dataset.h"

using namespace opennn;

namespace {

constexpr float tol = 1e-3f;

}

TEST(YoloInference, DecodeDetectionsRoundTripsThroughLetterbox)
{

    const Index original_height = 200;
    const Index original_width = 100;
    const Index network_height = 416;
    const Index network_width = 416;

    const float nms_output[12] = {
        0.5f, 0.5f, 0.2f, 0.4f, 0.85f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f,
    };

    const vector<YoloDetection> detections = decode_yolo_detections(
        nms_output,               2,
        original_height, original_width,
        network_height, network_width);

    ASSERT_EQ(detections.size(), 1u);

    EXPECT_NEAR(detections[0].center_x, 50.0f,  tol);
    EXPECT_NEAR(detections[0].center_y, 100.0f, tol);
    EXPECT_NEAR(detections[0].width,    40.0f,  tol);
    EXPECT_NEAR(detections[0].height,   80.0f,  tol);
    EXPECT_FLOAT_EQ(detections[0].score, 0.85f);
    EXPECT_EQ(detections[0].class_id, 1);
}

TEST(YoloInference, DecodeDetectionsStopsAtFirstZeroScore)
{

    const float nms_output[18] = {
        0.5f, 0.5f, 0.3f, 0.3f, 0.9f, 0.0f,
        0.2f, 0.2f, 0.1f, 0.1f, 0.7f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    const vector<YoloDetection> detections = decode_yolo_detections(
        nms_output,               3,
                  100,           100,          100,          100);

    ASSERT_EQ(detections.size(), 2u);
    EXPECT_FLOAT_EQ(detections[0].score, 0.9f);
    EXPECT_FLOAT_EQ(detections[1].score, 0.7f);
}

TEST(YoloInference, DecodeDetectionsSquareImageNoPadding)
{

    const float nms_output[6] = { 0.5f, 0.25f, 0.4f, 0.2f, 1.0f, 0.0f };

    const vector<YoloDetection> detections = decode_yolo_detections(
        nms_output,               1,
                  256,           256,          256,          256);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].center_x, 128.0f, tol);
    EXPECT_NEAR(detections[0].center_y,  64.0f, tol);
    EXPECT_NEAR(detections[0].width,    102.4f, tol);
    EXPECT_NEAR(detections[0].height,   51.2f,  tol);
}

namespace {

void plant_fpn_box(vector<float>& buffer,
                   Index grid, Index boxes_per_cell, Index classes,
                   Index box_slot,
                   float cx, float cy, float w, float h,
                   float objectness, Index class_id)
{
    const Index values_per_box = 5 + classes;
    const Index channels = boxes_per_cell * values_per_box;
    const Index row = Index(cy * float(grid));
    const Index col = Index(cx * float(grid));
    const Index base = (row * grid + col) * channels + box_slot * values_per_box;

    buffer[base + 0] = cx * float(grid) - float(col);
    buffer[base + 1] = cy * float(grid) - float(row);
    buffer[base + 2] = w;
    buffer[base + 3] = h;
    buffer[base + 4] = objectness;
    for (Index c = 0; c < classes; ++c)
        buffer[base + 5 + c] = (c == class_id) ? 1.0f : 0.0f;
}

}

TEST(YoloInference, DecodeFpnSingleHeadRoundTripsThroughLetterbox)
{

    const Index grid = 13;
    const Index boxes_per_cell = 1;
    const Index classes = 2;

    vector<float> head_buffer(size_t(grid * grid * boxes_per_cell * (5 + classes)), 0.0f);
    plant_fpn_box(head_buffer, grid, boxes_per_cell, classes,
                           0,        0.5f,        0.5f,       0.2f,       0.4f,
                          0.85f,           1);

    YoloFpnHead head{ head_buffer.data(), grid, boxes_per_cell, classes };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        { head },
                   200,            100,
                  416,            416);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].center_x, 50.0f,  tol);
    EXPECT_NEAR(detections[0].center_y, 100.0f, tol);
    EXPECT_NEAR(detections[0].width,    40.0f,  tol);
    EXPECT_NEAR(detections[0].height,   80.0f,  tol);
    EXPECT_NEAR(detections[0].score,    0.85f,  tol);
    EXPECT_EQ(detections[0].class_id, 1);
}

TEST(YoloInference, DecodeFpnSuppressesOverlappingSameClassAcrossScales)
{

    const Index classes = 2;

    const Index grid_a = 4;
    vector<float> head_a(size_t(grid_a * grid_a * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_a, grid_a, 1, classes, 0,
                         0.5f,        0.5f,       0.2f,       0.2f,
                          0.9f,           0);

    const Index grid_b = 8;
    vector<float> head_b(size_t(grid_b * grid_b * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_b, grid_b, 1, classes, 0,
                         0.51f,        0.49f,       0.2f,       0.2f,
                          0.7f,           0);

    const vector<YoloFpnHead> heads = {
        { head_a.data(), grid_a, 1, classes },
        { head_b.data(), grid_b, 1, classes },
    };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        heads,            100,            100,           100,           100);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].score, 0.9f, tol);
    EXPECT_EQ(detections[0].class_id, 0);
}

TEST(YoloInference, DecodeFpnKeepsOverlappingDifferentClassesAcrossScales)
{

    const Index classes = 2;

    const Index grid_a = 4;
    vector<float> head_a(size_t(grid_a * grid_a * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_a, grid_a, 1, classes, 0,
                         0.5f,        0.5f,       0.2f,       0.2f,
                          0.9f,           0);

    const Index grid_b = 8;
    vector<float> head_b(size_t(grid_b * grid_b * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_b, grid_b, 1, classes, 0,
                         0.51f,        0.49f,       0.2f,       0.2f,
                          0.7f,           1);

    const vector<YoloFpnHead> heads = {
        { head_a.data(), grid_a, 1, classes },
        { head_b.data(), grid_b, 1, classes },
    };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        heads,            100,            100,           100,           100);

    ASSERT_EQ(detections.size(), 2u);
    EXPECT_NEAR(detections[0].score, 0.9f, tol);
    EXPECT_EQ(detections[0].class_id, 0);
    EXPECT_NEAR(detections[1].score, 0.7f, tol);
    EXPECT_EQ(detections[1].class_id, 1);
}

TEST(YoloInference, DecodeFpnConfidenceThresholdFiltersLowScores)
{

    const Index grid = 4;
    const Index classes = 2;
    vector<float> head(size_t(grid * grid * 1 * (5 + classes)), 0.0f);

    plant_fpn_box(head, grid, 1, classes, 0,
                         0.125f,        0.125f,       0.2f,       0.2f,
                          0.9f,           0);
    plant_fpn_box(head, grid, 1, classes, 0,
                         0.875f,        0.875f,       0.2f,       0.2f,
                          0.3f,           1);

    YoloFpnHead h{ head.data(), grid, 1, classes };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        { h },            100,            100,           100,           100,
                                 0.5f);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].score, 0.9f, tol);
}
