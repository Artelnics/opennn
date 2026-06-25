//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L E   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "upsample_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"
#ifdef OPENNN_HAS_CUDA
#  include "kernel.cuh"
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
    apply(input, output);
}

void UpsampleOperator::apply(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.shape[0];
    const Index out_h = input_height * scale_factor;
    const Index out_w = input_width * scale_factor;

    const float* src = input.as<float>();
    float* dst = output.as<float>();

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index oh = 0; oh < out_h; ++oh)
        {
            const Index ih = oh / scale_factor;
            for (Index ow = 0; ow < out_w; ++ow)
            {
                const Index iw = ow / scale_factor;
                const Index in_idx  = ((b * input_height + ih) * input_width + iw) * channels;
                const Index out_idx = ((b * out_h + oh) * out_w + ow) * channels;
                for (Index c = 0; c < channels; ++c)
                    dst[out_idx + c] = src[in_idx + c];
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
    apply_delta(output_delta, input_delta);
}

void UpsampleOperator::apply_delta(const TensorView& output_delta, TensorView& input_delta) const
{
    const Index batch_size = input_delta.shape[0];
    const Index out_h = input_height * scale_factor;
    const Index out_w = input_width * scale_factor;

    const float* delta = output_delta.as<float>();
    float* in_delta = input_delta.as<float>();

    fill_n(in_delta, input_delta.size(), 0.0f);

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index ih = 0; ih < input_height; ++ih)
            for (Index iw = 0; iw < input_width; ++iw)
            {
                const Index in_idx = ((b * input_height + ih) * input_width + iw) * channels;
                for (Index dh = 0; dh < scale_factor; ++dh)
                    for (Index dw = 0; dw < scale_factor; ++dw)
                    {
                        const Index oh = ih * scale_factor + dh;
                        const Index ow = iw * scale_factor + dw;
                        const Index out_idx = ((b * out_h + oh) * out_w + ow) * channels;
                        for (Index c = 0; c < channels; ++c)
                            in_delta[in_idx + c] += delta[out_idx + c];
                    }
            }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
