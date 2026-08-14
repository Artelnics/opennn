//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/upsample_layer.h"
#include "opennn/registry.h"
#include "opennn/core/json.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/layers/kernel_upsample.cuh"
#endif

namespace opennn
{

void UpsampleOperator::set(Index in_h, Index in_w, Index ch, Index scale)
{
    throw_if(scale < 1,
             "Upsample: scale_factor must be >= 1.");
    input_height = in_h;
    input_width = in_w;
    channels = ch;
    scale_factor = scale;
}

void UpsampleOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        upsample_forward_cuda(to_int(input.shape[0]), to_int(input_height), to_int(input_width),
                              to_int(channels), to_int(scale_factor),
                              input.as<float>(), output.as<float>());
        return;
    }
#endif

    const Index batch_size = input.shape[0];
    const Index out_h = input_height * scale_factor;
    const Index out_w = input_width * scale_factor;
    const float* src = input.as<float>();
    float* dst = output.as<float>();

    const size_t ch_bytes = size_t(channels) * sizeof(float);

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index oh = 0; oh < out_h; ++oh)
        {
            const Index ih = oh / scale_factor;
            for (Index ow = 0; ow < out_w; ++ow)
            {
                const Index iw = ow / scale_factor;
                memcpy(dst + (b * out_h + oh) * out_w * channels + ow * channels,
                       src + (b * input_height + ih) * input_width * channels + iw * channels,
                       ch_bytes);
            }
        }
}

void UpsampleOperator::back_propagate(ForwardPropagation&, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta = get_input_delta(back_propagation, layer);
    if (input_delta.empty()) return;

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
    {
        upsample_backward_cuda(to_int(input_delta.shape[0]), to_int(input_height), to_int(input_width),
                               to_int(channels), to_int(scale_factor),
                               output_delta.as<float>(), input_delta.as<float>());
        return;
    }
#endif

    const Index batch_size = input_delta.shape[0];
    const Index out_h = input_height * scale_factor;
    const Index out_w = input_width * scale_factor;
    const float* delta = output_delta.as<float>();
    float* in_delta = input_delta.as<float>();

    // The loop visits every input pixel exactly once, so each one clears its own
    // channels before accumulating into them. Zeroing the whole gradient up front
    // instead would be a serial pass over memory the parallel loop rewrites.
    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index ih = 0; ih < input_height; ++ih)
            for (Index iw = 0; iw < input_width; ++iw)
            {
                float* in_ptr = in_delta + ((b * input_height + ih) * input_width + iw) * channels;
                fill_n(in_ptr, channels, 0.0f);

                for (Index dh = 0; dh < scale_factor; ++dh)
                    for (Index dw = 0; dw < scale_factor; ++dw)
                    {
                        const float* out_ptr = delta + ((b * out_h + ih * scale_factor + dh) * out_w + iw * scale_factor + dw) * channels;
                        for (Index c = 0; c < channels; ++c)
                            in_ptr[c] += out_ptr[c];
                    }
            }
}

Upsample::Upsample(const Shape& new_input_shape,
                   Index new_scale_factor,
                   const string& new_label)
    : Layer(LayerType::Upsample)
{
    operators = {&upsample};
    set(new_input_shape, new_scale_factor, new_label);
}

Shape Upsample::get_output_shape() const
{
    if (input_shape.empty()) return {};
    return { input_shape[0] * upsample.scale_factor,
             input_shape[1] * upsample.scale_factor,
             input_shape[2] };
}

void Upsample::set(const Shape& new_input_shape,
                   Index new_scale_factor,
                   const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "Upsample", "input");

    input_shape = new_input_shape;
    upsample.scale_factor = new_scale_factor;
    set_label(new_label);
    configure_operator();
}

void Upsample::apply_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape, upsample.scale_factor, label);
}

void Upsample::set_scale_factor(Index new_scale_factor)
{
    upsample.scale_factor = new_scale_factor;
    configure_operator();
}

void Upsample::configure_operator()
{
    if (input_shape.empty()) return;
    upsample.set(input_shape[0], input_shape[1], input_shape[2], upsample.scale_factor);
}

void Upsample::read_JSON_body(const Json* root)
{
    set_scale_factor(read_json_index(root, "ScaleFactor"));
}

void Upsample::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "ScaleFactor", upsample.scale_factor);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
