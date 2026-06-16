//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "detection_operator.h"
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

float yolo_sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

}


void DetectionOp::set(const Shape& input_shape, const vector<array<float, 2>>& new_anchors)
{
    throw_if(input_shape.rank != 3,
             "DetectionOp: input shape must be rank 3.");
    throw_if(new_anchors.empty(),
             "DetectionOp: anchors are empty.");

    grid_size = input_shape[0];
    grid_width = input_shape[1];
    boxes_per_cell = ssize(new_anchors);
    anchors = new_anchors;

    const Index channels = input_shape[2];
    throw_if(channels % boxes_per_cell != 0,
             "DetectionOp: channels must be divisible by boxes_per_cell.");

    classes_number = channels / boxes_per_cell - 5;
    throw_if(classes_number <= 0,
             "DetectionOp: classes_number must be positive.");
}

void DetectionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output = get_output(fp, layer);

    throw_if(input.is_cuda(),
             "DetectionOp GPU path is not implemented yet.");

    apply(input, output);
}

void DetectionOp::apply(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.shape[0];
    const Index channels = input.shape[3];
    const Index values_per_box = 5 + classes_number;

    const float* src = input.as<float>();
    float* dst = output.as<float>();

    #pragma omp parallel for collapse(3)
    for (Index b = 0; b < batch_size; ++b)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_width; ++col)
            {
                const Index cell = ((b * grid_size + row) * grid_width + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    dst[base + 0] = yolo_sigmoid(src[base + 0]);
                    dst[base + 1] = yolo_sigmoid(src[base + 1]);
                    dst[base + 2] = exp(src[base + 2]) * anchors[size_t(box)][0];
                    dst[base + 3] = exp(src[base + 3]) * anchors[size_t(box)][1];
                    dst[base + 4] = yolo_sigmoid(src[base + 4]);

                    if (class_activation == ClassActivation::Sigmoid)
                    {
                        for (Index c = 0; c < classes_number; ++c)
                            dst[base + 5 + c] = yolo_sigmoid(src[base + 5 + c]);
                    }
                    else
                    {
                        const float max_logit = *max_element(src + base + 5, src + base + 5 + classes_number);

                        float sum = 0.0f;
                        for (Index c = 0; c < classes_number; ++c)
                        {
                            const float exp_value = exp(src[base + 5 + c] - max_logit);
                            dst[base + 5 + c] = exp_value;
                            sum += exp_value;
                        }

                        const float inv_sum = 1.0f / (sum + EPSILON);
                        for (Index c = 0; c < classes_number; ++c)
                            dst[base + 5 + c] *= inv_sum;
                    }
                }
            }
}

void DetectionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    const TensorView& output = get_output(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta = get_input_delta(bp, layer);

    if (input_delta.empty()) return;

    throw_if(output_delta.is_cuda(),
             "DetectionOp GPU path is not implemented yet.");

    apply_delta(output, output_delta, input_delta);
}

void DetectionOp::apply_delta(const TensorView& output,
                              const TensorView& output_delta,
                              TensorView& input_delta) const
{
    const Index batch_size = output.shape[0];
    const Index channels = output.shape[3];
    const Index values_per_box = 5 + classes_number;

    const float* out = output.as<float>();
    const float* delta = output_delta.as<float>();
    float* in_delta = input_delta.as<float>();

    #pragma omp parallel for collapse(3)
    for (Index b = 0; b < batch_size; ++b)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_width; ++col)
            {
                const Index cell = ((b * grid_size + row) * grid_width + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    in_delta[base + 0] = delta[base + 0] * out[base + 0] * (1.0f - out[base + 0]);
                    in_delta[base + 1] = delta[base + 1] * out[base + 1] * (1.0f - out[base + 1]);
                    in_delta[base + 2] = delta[base + 2] * out[base + 2];
                    in_delta[base + 3] = delta[base + 3] * out[base + 3];
                    in_delta[base + 4] = delta[base + 4] * out[base + 4] * (1.0f - out[base + 4]);

                    if (class_activation == ClassActivation::Sigmoid)
                    {
                        for (Index c = 0; c < classes_number; ++c)
                        {
                            const float s = out[base + 5 + c];
                            in_delta[base + 5 + c] = delta[base + 5 + c] * s * (1.0f - s);
                        }
                    }
                    else
                    {
                        float dot = 0.0f;
                        for (Index c = 0; c < classes_number; ++c)
                            dot += delta[base + 5 + c] * out[base + 5 + c];

                        for (Index c = 0; c < classes_number; ++c)
                            in_delta[base + 5 + c] = out[base + 5 + c] * (delta[base + 5 + c] - dot);
                    }
                }
            }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
