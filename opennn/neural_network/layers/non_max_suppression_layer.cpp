//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P P R E S S I O N   L A Y E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/non_max_suppression_layer.h"
#include "opennn/registry.h"
#include "opennn/core/json.h"

#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"
#include "opennn/neural_network/forward_propagation.h"
#include <algorithm>

namespace opennn
{

static float yolo_iou_xywh(const array<float, 6>& a, const array<float, 6>& b)
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

void NonMaxSuppressionOperator::set(const Shape& input_shape,
                              Index new_boxes_per_cell,
                              float new_confidence_threshold,
                              float new_iou_threshold)
{
    throw_if(input_shape.get_rank() != 3,
             "NonMaxSuppressionOperator: input shape must be rank 3.");
    throw_if(new_boxes_per_cell <= 0,
             "NonMaxSuppressionOperator: boxes_per_cell must be positive.");

    grid_size = input_shape[0];
    grid_width = input_shape[1];
    boxes_per_cell = new_boxes_per_cell;
    confidence_threshold = new_confidence_threshold;
    iou_threshold = new_iou_threshold;

    const Index channels = input_shape[2];
    throw_if(channels % boxes_per_cell != 0,
             "NonMaxSuppressionOperator: channels must be divisible by boxes_per_cell.");

    classes_number = channels / boxes_per_cell - 5;
    throw_if(classes_number <= 0,
             "NonMaxSuppressionOperator: classes_number must be positive.");
}

void NonMaxSuppressionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output = get_output(forward_propagation, layer);

    if (is_training) return;

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        cudaStream_t stream = device::get_compute_stream();

        cpu_input_staging.resize(size_t(input.size()));
        device::copy_async(cpu_input_staging.data(), input.as<float>(),
                           input.size() * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);

        TensorView cpu_in{cpu_input_staging.data(), input.get_shape()};

        cpu_output_staging.resize(size_t(output.size()));
        TensorView cpu_out{cpu_output_staging.data(), output.get_shape()};

        apply(cpu_in, cpu_out);

        return device::copy_async(output.as<float>(), cpu_output_staging.data(),
                                  output.size() * Index(sizeof(float)),
                                  device::CopyKind::HostToDevice, stream);
    }
#endif
    apply(input, output);
}

void NonMaxSuppressionOperator::apply(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.get_shape()[0];
    const Index channels = input.get_shape()[3];
    const Index values_per_box = 5 + classes_number;
    const Index max_boxes = grid_size * grid_width * boxes_per_cell;

    const float* src = input.as<float>();
    float* dst = output.as<float>();
    fill_n(dst, output.size(), 0.0f);

    #pragma omp parallel for
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
                        (float(col) + src[base]) / float(grid_width),
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
            ranges::copy(candidate, out);
            if (++kept_count == max_boxes)
                break;
        }
    }
}

NonMaxSuppression::NonMaxSuppression(const Shape& new_input_shape,
                                     Index new_boxes_per_cell,
                                     float new_confidence_threshold,
                                     float new_iou_threshold,
                                     const string& new_label)
    : Layer(LayerType::NonMaxSuppression, false)
{
    operators = {&nms};

    set(new_input_shape,
        new_boxes_per_cell,
        new_confidence_threshold,
        new_iou_threshold,
        new_label);
}

Shape NonMaxSuppression::get_output_shape() const
{
    if (input_shape.get_rank() != 3) return {};
    return {input_shape[0] * input_shape[1] * nms.boxes_per_cell, 6};
}

void NonMaxSuppression::set(const Shape& new_input_shape,
                            Index new_boxes_per_cell,
                            float new_confidence_threshold,
                            float new_iou_threshold,
                            const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "NonMaxSuppression", "input");

    input_shape = new_input_shape;
    nms.boxes_per_cell = new_boxes_per_cell;
    nms.confidence_threshold = new_confidence_threshold;
    nms.iou_threshold = new_iou_threshold;
    set_label(new_label);
    configure_operator();
}

void NonMaxSuppression::apply_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape, nms.boxes_per_cell, nms.confidence_threshold,
        nms.iou_threshold, label);
}

void NonMaxSuppression::configure_operator()
{
    if (input_shape.empty()) return;
    nms.set(input_shape, nms.boxes_per_cell, nms.confidence_threshold, nms.iou_threshold);
}

void NonMaxSuppression::read_JSON_body(const Json* root)
{
    nms.boxes_per_cell = read_json_index(root, "BoxesPerCell");
    nms.confidence_threshold = read_json_float(root, "ConfidenceThreshold");
    nms.iou_threshold = read_json_float(root, "IouThreshold");
    configure_operator();
}

void NonMaxSuppression::write_JSON_body(JsonWriter& writer) const
{
    write_json(writer, {
        {"BoxesPerCell", nms.boxes_per_cell},
        {"ConfidenceThreshold", nms.confidence_threshold},
        {"IouThreshold", nms.iou_threshold}
    });
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
