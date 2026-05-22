#include "pch.h"

#include "../opennn/yolo_dataset.h"

using namespace opennn;

namespace {

constexpr float tol = 1e-3f;

}

TEST(YoloInference, DecodeDetectionsRoundTripsThroughLetterbox)
{
    // Original image 100x200; network input 416x416.
    // Forward letterbox: scale = min(416/100, 416/200) = 2.08, fits height exactly,
    // pads width with offset_x = 104.
    //
    // Take a box centered at (50, 100) px in original space, size 40x80 px.
    //   forward-letterboxed center in network px: (50*2.08 + 104, 100*2.08 + 0) = (208, 208)
    //   normalized: (0.5, 0.5); size normalized: (40*2.08/416, 80*2.08/416) = (0.2, 0.4).

    const Index original_height = 200;
    const Index original_width = 100;
    const Index network_height = 416;
    const Index network_width = 416;

    // Two slots: one valid detection, one empty (terminating zero score).
    const float nms_output[12] = {
        0.5f, 0.5f, 0.2f, 0.4f, 0.85f, 1.0f,   // x, y, w, h, score, class_id
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f,   // empty slot
    };

    const vector<YoloDetection> detections = decode_yolo_detections(
        nms_output, /*max_boxes=*/2,
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
    // NMS writes kept boxes contiguously from slot 0 and pads the rest with zeros.
    // The decoder should stop on the first zero-score row.

    const float nms_output[18] = {
        0.5f, 0.5f, 0.3f, 0.3f, 0.9f, 0.0f,
        0.2f, 0.2f, 0.1f, 0.1f, 0.7f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    const vector<YoloDetection> detections = decode_yolo_detections(
        nms_output, /*max_boxes=*/3,
        /*orig_h*/100, /*orig_w*/100, /*net_h*/100, /*net_w*/100);

    ASSERT_EQ(detections.size(), 2u);
    EXPECT_FLOAT_EQ(detections[0].score, 0.9f);
    EXPECT_FLOAT_EQ(detections[1].score, 0.7f);
}

TEST(YoloInference, DecodeDetectionsSquareImageNoPadding)
{
    // Square image equal to network input — letterbox is a no-op (scale=1, offsets=0).
    // Normalized coords map directly to pixel coords.

    const float nms_output[6] = { 0.5f, 0.25f, 0.4f, 0.2f, 1.0f, 0.0f };

    const vector<YoloDetection> detections = decode_yolo_detections(
        nms_output, /*max_boxes=*/1,
        /*orig_h*/256, /*orig_w*/256, /*net_h*/256, /*net_w*/256);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].center_x, 128.0f, tol);
    EXPECT_NEAR(detections[0].center_y,  64.0f, tol);
    EXPECT_NEAR(detections[0].width,    102.4f, tol);
    EXPECT_NEAR(detections[0].height,   51.2f,  tol);
}
