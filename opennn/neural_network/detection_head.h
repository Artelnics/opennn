//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   H E A D   C O N T R A C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>

#include "opennn/core/opennn_types.h"

namespace opennn
{

enum class DetectionHeadKind { AnchorBased, AnchorFree };
enum class DetectionClassActivation { Softmax, Sigmoid };

struct DetectionHeadMetadata
{
    DetectionHeadKind kind = DetectionHeadKind::AnchorBased;
    Index boxes_per_cell = 1;
    Index classes_number = 0;
    Index regression_bins = 1;
    DetectionClassActivation class_activation = DetectionClassActivation::Softmax;

    bool is_anchor_free() const noexcept { return kind == DetectionHeadKind::AnchorFree; }
    bool uses_sigmoid_classes() const noexcept { return class_activation == DetectionClassActivation::Sigmoid; }
};

// Two primitives the dataset, the loss and non-max suppression all have to
// agree on, because they describe the same encoding from three sides: the
// dataset builds targets in it, the loss scores against it, and suppression
// reads it back. Each kept its own copy -- identical today, and exactly the
// kind of thing that fails silently if one of them is ever tuned alone.

// Intersection over union of two boxes held as centre-x, centre-y, width and
// height in the first four slots of a candidate.
inline float yolo_box_iou(const std::array<float, 6>& a, const std::array<float, 6>& b)
{
    const float a_left = a[0] - 0.5f * a[2];
    const float a_top = a[1] - 0.5f * a[3];
    const float a_right = a[0] + 0.5f * a[2];
    const float a_bottom = a[1] + 0.5f * a[3];

    const float b_left = b[0] - 0.5f * b[2];
    const float b_top = b[1] - 0.5f * b[3];
    const float b_right = b[0] + 0.5f * b[2];
    const float b_bottom = b[1] + 0.5f * b[3];

    const float inter_w = max(0.0f, min(a_right, b_right) - max(a_left, b_left));
    const float inter_h = max(0.0f, min(a_bottom, b_bottom) - max(a_top, b_top));
    const float inter = inter_w * inter_h;

    const float area = a[2] * a[3] + b[2] * b[3] - inter;

    return area > 0.0f ? inter / area : 0.0f;
}

// Distribution Focal Loss decode: the expected bin of a softmax over reg_max
// logits, which is what turns a v8 head's distribution into a distance.
// Shifted by the maximum before exponentiating, so a confident head cannot
// overflow.
inline float dfl_decode(const float* logits, Index reg_max)
{
    float max_l = *max_element(logits, logits + reg_max);
    float sum = 0.0f;
    for (Index i = 0; i < reg_max; ++i) sum += expf(logits[i] - max_l);
    float d = 0.0f;
    for (Index i = 0; i < reg_max; ++i) d += float(i) * expf(logits[i] - max_l) / sum;
    return d;
}

class DetectionHeadEndpoint
{
public:
    virtual ~DetectionHeadEndpoint() = default;

    virtual DetectionHeadMetadata get_detection_head_metadata() const noexcept = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
