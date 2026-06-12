//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P P R E S S I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "non_max_suppression_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

namespace
{

float yolo_iou_xywh(const array<float, 6>& a, const array<float, 6>& b)
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

}


void NonMaxSuppressionOp::set(const Shape& input_shape,
                              Index new_boxes_per_cell,
                              float new_confidence_threshold,
                              float new_iou_threshold)
{
    throw_if(input_shape.rank != 3,
             "NonMaxSuppressionOp: input shape must be rank 3.");
    throw_if(new_boxes_per_cell <= 0,
             "NonMaxSuppressionOp: boxes_per_cell must be positive.");

    grid_size = input_shape[0];
    grid_width = input_shape[1];
    boxes_per_cell = new_boxes_per_cell;
    confidence_threshold = new_confidence_threshold;
    iou_threshold = new_iou_threshold;

    const Index channels = input_shape[2];
    throw_if(channels % boxes_per_cell != 0,
             "NonMaxSuppressionOp: channels must be divisible by boxes_per_cell.");

    classes_number = channels / boxes_per_cell - 5;
    throw_if(classes_number <= 0,
             "NonMaxSuppressionOp: classes_number must be positive.");
}

void NonMaxSuppressionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output = get_output(fp, layer);

    throw_if(input.is_cuda(),
             "NonMaxSuppressionOp GPU path is not implemented yet.");
    apply(input, output);
}

void NonMaxSuppressionOp::apply(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.shape[0];
    const Index channels = input.shape[3];
    const Index values_per_box = 5 + classes_number;
    const Index max_boxes = grid_size * grid_width * boxes_per_cell;

    const float* src = input.as<float>();
    float* dst = output.as<float>();
    fill_n(dst, output.size(), 0.0f);

    for (Index b = 0; b < batch_size; ++b)
    {
        vector<array<float, 6>> candidates;
        candidates.reserve(size_t(max_boxes));

        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_width; ++col)
            {
                const Index cell = ((b * grid_size + row) * grid_width + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    const float* best = max_element(src + base + 5, src + base + 5 + classes_number);
                    const Index best_class = best - (src + base + 5);
                    const float best_probability = *best;

                    const float score = src[base + 4] * best_probability;
                    if (score < confidence_threshold)
                        continue;

                    candidates.push_back({
                        (float(col) + src[base + 0]) / float(grid_width),
                        (float(row) + src[base + 1]) / float(grid_size),
                        src[base + 2],
                        src[base + 3],
                        score,
                        float(best_class)
                    });
                }
            }

        ranges::sort(candidates, greater<>{}, [](const array<float, 6>& box) { return box[4]; });

        Index kept_count = 0;
        for (const array<float, 6>& candidate : candidates)
        {
            bool suppressed = false;
            for (Index j = 0; j < kept_count; ++j)
            {
                const float* kept = dst + (b * max_boxes + j) * 6;
                const array<float, 6> kept_box{kept[0], kept[1], kept[2], kept[3], kept[4], kept[5]};

                if (Index(kept_box[5]) == Index(candidate[5])
                &&  yolo_iou_xywh(candidate, kept_box) > iou_threshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (suppressed)
                continue;

            float* out = dst + (b * max_boxes + kept_count) * 6;
            std::copy(candidate.begin(), candidate.end(), out);
            if (++kept_count == max_boxes)
                break;
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
