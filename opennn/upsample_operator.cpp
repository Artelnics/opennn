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

namespace opennn
{

void UpsampleOp::set(Index in_h, Index in_w, Index ch, Index scale)
{
    throw_if(scale < 1,
             "Upsample: scale_factor must be >= 1.");
    input_height = in_h;
    input_width = in_w;
    channels = ch;
    scale_factor = scale;
}

void UpsampleOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    throw_if(input.is_cuda(),
             "UpsampleOp GPU path is not implemented yet.");

    apply(input, output);
}

void UpsampleOp::apply(const TensorView& input, TensorView& output) const
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

void UpsampleOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta = get_input_delta(bp, layer);
    if (input_delta.empty()) return;

    throw_if(output_delta.is_cuda(),
             "UpsampleOp GPU path is not implemented yet.");

    apply_delta(output_delta, input_delta);
}

void UpsampleOp::apply_delta(const TensorView& output_delta, TensorView& input_delta) const
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
