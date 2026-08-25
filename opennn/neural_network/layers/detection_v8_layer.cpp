//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   V 8   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/registry.h"
#include "opennn/core/json.h"

#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/layers/kernel_detection.cuh"
#include <cuda_runtime.h>
#endif

namespace opennn
{

void DetectionV8Operator::set(const Shape& input_shape, Index new_reg_max)
{
    throw_if(input_shape.get_rank() != 3,
             "DetectionV8Operator: input shape must be rank 3.");
    reg_max = max(Index(1), new_reg_max);
    const Index box_ch = 4 * reg_max;
    throw_if(input_shape[2] <= box_ch,
             "DetectionV8Operator: channels must be > 4*reg_max (need at least 1 class).");

    grid_size      = input_shape[0];
    grid_width     = input_shape[1];
    classes_number = input_shape[2] - box_ch;
}

void DetectionV8Operator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode)
{
    const TensorView& input  = get_input(forward_propagation, layer);
    TensorView&       output = get_output(forward_propagation, layer);

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
        return detection_v8_forward_cuda(input.get_shape()[0], grid_size, grid_width, classes_number,
                                         reg_max, input.as<float>(), output.as<float>());
#endif

    const Index batch_size = input.get_shape()[0];
    const Index box_ch     = 4 * reg_max;
    const Index channels   = box_ch + classes_number;
    const Index sig_start  = reg_max > 1 ? box_ch : 0;

    const float* src = input.as<float>();
    float*       dst = output.as<float>();

    const Index cells_count = batch_size * grid_size * grid_width;

    #pragma omp parallel for
    for (Index cell = 0; cell < cells_count; ++cell)
    {
        const Index base = cell * channels;

        for (Index ch = 0; ch < sig_start; ++ch)
            dst[base + ch] = src[base + ch];
        for (Index ch = sig_start; ch < channels; ++ch)
            dst[base + ch] = 1.0f / (1.0f + expf(-src[base + ch]));
    }
}

void DetectionV8Operator::back_propagate(ForwardPropagation& forward_propagation,
                                          BackPropagation&    back_propagation,
                                          size_t              layer) const
{
    const TensorView& output       = get_output(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView&       input_delta  = get_input_delta(back_propagation, layer);

    if (input_delta.empty()) return;

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
        return detection_v8_backward_cuda(output.get_shape()[0], grid_size, grid_width, classes_number,
                                          reg_max, output.as<float>(), output_delta.as<float>(),
                                          input_delta.as<float>());
#endif

    const Index batch_size = output.get_shape()[0];
    const Index box_ch     = 4 * reg_max;
    const Index channels   = box_ch + classes_number;
    const Index sig_start  = reg_max > 1 ? box_ch : 0;

    const float* out      = output.as<float>();
    const float* delta    = output_delta.as<float>();
    float*       in_delta = input_delta.as<float>();

    const Index cells_count = batch_size * grid_size * grid_width;

    #pragma omp parallel for
    for (Index cell = 0; cell < cells_count; ++cell)
    {
        const Index base = cell * channels;

        for (Index ch = 0; ch < sig_start; ++ch)
            in_delta[base + ch] = delta[base + ch];
        for (Index ch = sig_start; ch < channels; ++ch)
        {
            const float s = out[base + ch];
            in_delta[base + ch] = delta[base + ch] * s * (1.0f - s);
        }
    }
}

DetectionV8::DetectionV8(const Shape& new_input_shape, const string& new_label)
    : Layer(LayerType::DetectionV8)
{
    operators = {&detection};
    set(new_input_shape, new_label);
}

DetectionV8::DetectionV8(const Shape& new_input_shape, Index reg_max, const string& new_label)
    : Layer(LayerType::DetectionV8)
{
    operators = {&detection};
    set(new_input_shape, reg_max, new_label);
}

void DetectionV8::set(const Shape& new_input_shape, const string& new_label)
{
    set(new_input_shape, 1, new_label);
}

void DetectionV8::set(const Shape& new_input_shape, Index reg_max, const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "DetectionV8", "input");

    input_shape = new_input_shape;
    set_label(new_label);
    detection.reg_max = max(Index(1), reg_max);
    configure_operator();
}

void DetectionV8::configure_operator()
{
    if (input_shape.empty()) return;
    detection.set(input_shape, detection.reg_max);
}

void DetectionV8::read_JSON_body(const Json* root)
{
    const Index classes  = read_json_index(root, "ClassesNumber");
    const Index gs       = read_json_index(root, "GridSize");
    const Index gw       = read_json_index(root, "GridWidth");
    const Json* rm_node  = root ? root->find("RegMax") : nullptr;
    const Index rm       = rm_node ? Index(read_json_index(root, "RegMax")) : Index(1);
    detection.reg_max    = max(Index(1), rm);
    const Index box_ch   = 4 * detection.reg_max;
    input_shape          = Shape{gs, gw, box_ch + classes};
    configure_operator();
}

void DetectionV8::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "ClassesNumber", to_string(detection.classes_number));
    add_json_field(writer, "GridSize",      to_string(detection.grid_size));
    add_json_field(writer, "GridWidth",     to_string(detection.grid_width));
    add_json_field(writer, "RegMax",        to_string(detection.reg_max));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
