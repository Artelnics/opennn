//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/registry.h"
#include "opennn/core/enum_map.h"
#include "opennn/core/json.h"
#include "opennn/core/string_utilities.h"

#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/layers/kernel_detection.cuh"
#include <cuda_runtime.h>
#endif

namespace opennn
{

void DetectionOperator::set(const Shape& input_shape, const vector<array<float, 2>>& new_anchors)
{
    throw_if(input_shape.get_rank() != 3,
             "DetectionOperator: input shape must be rank 3.");
    throw_if(new_anchors.empty(),
             "DetectionOperator: anchors are empty.");

    grid_size = input_shape[0];
    grid_width = input_shape[1];
    boxes_per_cell = ssize(new_anchors);
    anchors = new_anchors;

    throw_if(input_shape[2] % boxes_per_cell != 0,
             "DetectionOperator: channels must be divisible by boxes_per_cell.");

    classes_number = input_shape[2] / boxes_per_cell - 5;
    throw_if(classes_number <= 0,
             "DetectionOperator: classes_number must be positive.");
}

void DetectionOperator::forward_propagate(ForwardPropagation& forward_propagation,
                                          size_t layer,
                                          bool)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output = get_output(forward_propagation, layer);

#ifdef OPENNN_HAS_CUDA

    if(input.is_cuda())
    {
        throw_if(grid_size != grid_width,
                 "DetectionOperator GPU: non-square grids not supported.");

        const Index anchor_bytes = Index(anchors.size() * 2 * sizeof(float));

        if(device_anchors.byte_size() < anchor_bytes)
        {
            device_anchors.resize_bytes(anchor_bytes, Device::CUDA);

            vector<float> flat;
            flat.reserve(anchors.size() * 2);
            ranges::copy(anchors | views::join, back_inserter(flat));

            cudaMemcpyAsync(device_anchors.as<float>(),
                            flat.data(),
                            size_t(anchor_bytes),
                            cudaMemcpyHostToDevice,
                            device::get_compute_stream());
        }

        detection_forward_cuda(input.get_shape()[0],
                               grid_size,
                               boxes_per_cell,
                               classes_number,
                               static_cast<int>(class_activation),
                               device_anchors.as<float>(),
                               input.as<float>(),
                               output.as<float>());

        return;
    }

#endif

    const Index batch_size = input.get_shape()[0];
    const Index channels = input.get_shape()[3];
    const Index values_per_box = 5 + classes_number;

    const float* src = input.as<float>();
    float* dst = output.as<float>();

    const auto sigmoid = [](const float x)
    {
        return 1.0f / (1.0f + expf(-x));
    };

    const Index cells_count = batch_size * grid_size * grid_width;

    #pragma omp parallel for
    for (Index cell_index = 0; cell_index < cells_count; ++cell_index)
    {
        const Index cell = cell_index * channels;

        for (Index box = 0; box < boxes_per_cell; ++box)
        {
            const Index base = cell + box * values_per_box;
            const float* box_src = src + base;
            float* box_dst = dst + base;

            box_dst[0] = sigmoid(box_src[0]);
            box_dst[1] = sigmoid(box_src[1]);
            box_dst[2] = expf(clamp(box_src[2], -4.0f, 4.0f)) * anchors[size_t(box)][0];
            box_dst[3] = expf(clamp(box_src[3], -4.0f, 4.0f)) * anchors[size_t(box)][1];
            box_dst[4] = sigmoid(box_src[4]);

            if (class_activation == ClassActivation::Sigmoid)
            {
                for (Index c = 0; c < classes_number; ++c)
                    box_dst[5 + c] = sigmoid(box_src[5 + c]);

                continue;
            }

            const float* logits = box_src + 5;
            float* probabilities = box_dst + 5;

            const float max_logit = *max_element(logits, logits + classes_number);

            float sum = 0.0f;

            for (Index c = 0; c < classes_number; ++c)
            {
                probabilities[c] = expf(logits[c] - max_logit);
                sum += probabilities[c];
            }

            const float inv_sum = 1.0f / (sum + EPSILON);

            for (Index c = 0; c < classes_number; ++c)
                probabilities[c] *= inv_sum;
        }
    }
}

void DetectionOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output = get_output(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta = get_input_delta(back_propagation, layer);

    if (input_delta.empty()) return;

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
    {
        detection_backward_cuda(output.get_shape()[0], grid_size, boxes_per_cell, classes_number,
                                static_cast<int>(class_activation),
                                output.as<float>(), output_delta.as<float>(), input_delta.as<float>());
        return;
    }
#endif

    const Index batch_size = output.get_shape()[0];
    const Index values_per_box = 5 + classes_number;

    const float* out = output.as<float>();
    const float* delta = output_delta.as<float>();
    float* in_delta = input_delta.as<float>();

    const Index cells_count = batch_size * grid_size * grid_width;

    #pragma omp parallel for
    for (Index cell_index = 0; cell_index < cells_count; ++cell_index)
    {
        const Index cell = cell_index * output.get_shape()[3];

        for (Index box = 0; box < boxes_per_cell; ++box)
        {
            const Index base = cell + box * values_per_box;

            in_delta[base] = delta[base] * out[base] * (1.0f - out[base]);
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
                const float dot =
                    inner_product(delta + base + 5, delta + base + 5 + classes_number, out + base + 5, 0.0f);

                for (Index c = 0; c < classes_number; ++c)
                    in_delta[base + 5 + c] = out[base + 5 + c] * (delta[base + 5 + c] - dot);
            }
        }
    }
}

namespace
{

const EnumMap<DetectionOperator::ClassActivation>& class_activation_map()
{
    using ClassActivation = DetectionOperator::ClassActivation;
    static const vector<EnumMap<ClassActivation>::Entry> entries = {
        {ClassActivation::Softmax, "Softmax"},
        {ClassActivation::Sigmoid, "Sigmoid"}
    };
    static const EnumMap<ClassActivation> instance{entries};
    return instance;
}

string anchors_to_string(const vector<array<float, 2>>& anchors)
{
    ostringstream buffer;
    for (size_t i = 0; i < anchors.size(); ++i)
    {
        if (i != 0) buffer << ' ';
        buffer << anchors[i][0] << ' ' << anchors[i][1];
    }
    return buffer.str();
}

vector<array<float, 2>> string_to_anchors(string_view text)
{
    const vector<float> values = parse_number_list<float>(text, "Detection anchors");
    throw_if(values.size() % 2 != 0, "Detection anchors require width-height pairs.");

    vector<array<float, 2>> anchors(values.size() / 2);
    for (size_t i = 0; i < anchors.size(); ++i)
        anchors[i] = {values[2 * i], values[2 * i + 1]};

    return anchors;
}

}

Detection::Detection(const Shape& new_input_shape,
                     const vector<array<float, 2>>& new_anchors,
                     const string& new_label)
    : Layer(LayerType::Detection)
{
    operators = {&detection};

    set(new_input_shape, new_anchors, new_label);
}

void Detection::set(const Shape& new_input_shape,
                    const vector<array<float, 2>>& new_anchors,
                    const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "Detection", "input");

    input_shape = new_input_shape;
    set_label(new_label);
    detection.anchors = new_anchors;
    configure_operator();
}

void Detection::apply_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape, detection.anchors, label);
}

void Detection::configure_operator()
{
    if (input_shape.empty() || detection.anchors.empty()) return;
    detection.set(input_shape, detection.anchors);
}

void Detection::read_JSON_body(const Json* root)
{
    detection.anchors = string_to_anchors(read_json_string(root, "Anchors"));
    detection.class_activation = class_activation_map().from_string(read_json_string(root, "ClassActivation"));
    configure_operator();
}

void Detection::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "Anchors", anchors_to_string(detection.anchors));
    add_json_field(writer, "ClassActivation", class_activation_map().to_string(detection.class_activation));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
