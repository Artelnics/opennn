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

namespace {

// Writes a post-DetectionOperator box into `buffer` (shape [grid, grid, boxes_per_cell*(5+classes)]).
// cx,cy,w,h are normalized image coords; class_probs is a 0/1 one-hot over `classes`.
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

    buffer[base + 0] = cx * float(grid) - float(col);  // cell-relative x in [0,1]
    buffer[base + 1] = cy * float(grid) - float(row);
    buffer[base + 2] = w;
    buffer[base + 3] = h;
    buffer[base + 4] = objectness;
    for (Index c = 0; c < classes; ++c)
        buffer[base + 5 + c] = (c == class_id) ? 1.0f : 0.0f;
}

}  // namespace

TEST(YoloInference, DecodeFpnSingleHeadRoundTripsThroughLetterbox)
{
    // Same scenario as DecodeDetectionsRoundTripsThroughLetterbox, but routed
    // through the cross-scale path with a single head. Confirms the FPN decoder
    // matches the single-head decoder when only one head is supplied.

    const Index grid = 13;
    const Index boxes_per_cell = 1;
    const Index classes = 2;

    vector<float> head_buffer(size_t(grid * grid * boxes_per_cell * (5 + classes)), 0.0f);
    plant_fpn_box(head_buffer, grid, boxes_per_cell, classes,
                  /*slot=*/0, /*cx=*/0.5f, /*cy=*/0.5f, /*w=*/0.2f, /*h=*/0.4f,
                  /*obj=*/0.85f, /*class=*/1);

    YoloFpnHead head{ head_buffer.data(), grid, boxes_per_cell, classes };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        { head },
        /*orig_h=*/200, /*orig_w=*/100,
        /*net_h=*/416,  /*net_w=*/416);

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
    // Two heads at different scales both fire on nearly-identical boxes of the
    // same class. Cross-scale NMS must collapse them — only the higher-score
    // detection survives.

    const Index classes = 2;

    const Index grid_a = 4;
    vector<float> head_a(size_t(grid_a * grid_a * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_a, grid_a, 1, classes, 0,
                  /*cx=*/0.5f, /*cy=*/0.5f, /*w=*/0.2f, /*h=*/0.2f,
                  /*obj=*/0.9f, /*class=*/0);

    const Index grid_b = 8;
    vector<float> head_b(size_t(grid_b * grid_b * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_b, grid_b, 1, classes, 0,
                  /*cx=*/0.51f, /*cy=*/0.49f, /*w=*/0.2f, /*h=*/0.2f,
                  /*obj=*/0.7f, /*class=*/0);

    const vector<YoloFpnHead> heads = {
        { head_a.data(), grid_a, 1, classes },
        { head_b.data(), grid_b, 1, classes },
    };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        heads, /*orig_h=*/100, /*orig_w=*/100, /*net_h=*/100, /*net_w=*/100);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].score, 0.9f, tol);  // higher-confidence head wins
    EXPECT_EQ(detections[0].class_id, 0);
}

TEST(YoloInference, DecodeFpnKeepsOverlappingDifferentClassesAcrossScales)
{
    // Same geometry as the suppression test, but the two heads predict
    // different classes. Class-aware NMS must keep both.

    const Index classes = 2;

    const Index grid_a = 4;
    vector<float> head_a(size_t(grid_a * grid_a * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_a, grid_a, 1, classes, 0,
                  /*cx=*/0.5f, /*cy=*/0.5f, /*w=*/0.2f, /*h=*/0.2f,
                  /*obj=*/0.9f, /*class=*/0);

    const Index grid_b = 8;
    vector<float> head_b(size_t(grid_b * grid_b * 1 * (5 + classes)), 0.0f);
    plant_fpn_box(head_b, grid_b, 1, classes, 0,
                  /*cx=*/0.51f, /*cy=*/0.49f, /*w=*/0.2f, /*h=*/0.2f,
                  /*obj=*/0.7f, /*class=*/1);

    const vector<YoloFpnHead> heads = {
        { head_a.data(), grid_a, 1, classes },
        { head_b.data(), grid_b, 1, classes },
    };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        heads, /*orig_h=*/100, /*orig_w=*/100, /*net_h=*/100, /*net_w=*/100);

    ASSERT_EQ(detections.size(), 2u);
    // Sorted by score in the decoder output.
    EXPECT_NEAR(detections[0].score, 0.9f, tol);
    EXPECT_EQ(detections[0].class_id, 0);
    EXPECT_NEAR(detections[1].score, 0.7f, tol);
    EXPECT_EQ(detections[1].class_id, 1);
}

TEST(YoloInference, DecodeFpnConfidenceThresholdFiltersLowScores)
{
    // Two boxes in one head, scores 0.9 and 0.3, threshold 0.5 — only the high
    // score survives. Also verifies that the threshold is applied to
    // objectness*class_prob, not raw objectness.

    const Index grid = 4;
    const Index classes = 2;
    vector<float> head(size_t(grid * grid * 1 * (5 + classes)), 0.0f);

    plant_fpn_box(head, grid, 1, classes, 0,
                  /*cx=*/0.125f, /*cy=*/0.125f, /*w=*/0.2f, /*h=*/0.2f,
                  /*obj=*/0.9f, /*class=*/0);
    plant_fpn_box(head, grid, 1, classes, 0,
                  /*cx=*/0.875f, /*cy=*/0.875f, /*w=*/0.2f, /*h=*/0.2f,
                  /*obj=*/0.3f, /*class=*/1);

    YoloFpnHead h{ head.data(), grid, 1, classes };

    const vector<YoloDetection> detections = decode_yolo_fpn_detections(
        { h }, /*orig_h=*/100, /*orig_w=*/100, /*net_h=*/100, /*net_w=*/100,
        /*confidence_threshold=*/0.5f);

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_NEAR(detections[0].score, 0.9f, tol);
}
