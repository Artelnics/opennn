//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "concatenation_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

void ConcatenationOp::set(Index h, Index w, const vector<Index>& per_input_channels)
{
    throw_if(per_input_channels.empty(),
             "Concatenation: needs at least 1 input.");
    for (Index c : per_input_channels)
        throw_if(c <= 0, "Concatenation: per-input channels must be positive.");
    height = h;
    width = w;
    input_channels = per_input_channels;
}

void ConcatenationOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    const vector<TensorView>& inputs = get_inputs(fp, layer);
    TensorView& output = get_output(fp, layer);

    throw_if(inputs.size() != input_channels.size(),
             "Concatenation: input count mismatch.");

    throw_if(output.is_cuda(),
             "ConcatenationOp GPU path is not implemented yet.");

    const Index batch_size = output.shape[0];
    const Index total_channels = accumulate(input_channels.begin(), input_channels.end(), Index(0));

    float* dst = output.as<float>();

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index h = 0; h < height; ++h)
            for (Index w = 0; w < width; ++w)
            {
                const Index out_idx = ((b * height + h) * width + w) * total_channels;
                Index ch_offset = 0;
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    const Index in_c = input_channels[i];
                    const float* src = inputs[i].as<float>();
                    const Index in_idx = ((b * height + h) * width + w) * in_c;
                    for (Index c = 0; c < in_c; ++c)
                        dst[out_idx + ch_offset + c] = src[in_idx + c];
                    ch_offset += in_c;
                }
            }
}

void ConcatenationOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(bp, layer);

    const auto& backward_slots = bp.backward_slots[layer];
    const bool needs_input_delta = ranges::any_of(input_delta_slots, [&](size_t slot)
    {
        return slot < backward_slots.size() && !backward_slots[slot].empty();
    });

    if (!needs_input_delta) return;

    throw_if(output_delta.is_cuda(),
             "ConcatenationOp GPU path is not implemented yet.");

    const Index batch_size = output_delta.shape[0];
    const Index total_channels = accumulate(input_channels.begin(), input_channels.end(), Index(0));

    const float* delta = output_delta.as<float>();

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index h = 0; h < height; ++h)
            for (Index w = 0; w < width; ++w)
            {
                const Index out_idx = ((b * height + h) * width + w) * total_channels;
                Index ch_offset = 0;
                for (size_t i = 0; i < input_channels.size(); ++i)
                {
                    const Index in_c = input_channels[i];
                    TensorView& in_delta = get_input_delta(bp, layer, i);
                    if (in_delta.empty())
                    {
                        ch_offset += in_c;
                        continue;
                    }
                    float* dst = in_delta.as<float>();
                    const Index in_idx = ((b * height + h) * width + w) * in_c;
                    for (Index c = 0; c < in_c; ++c)
                        dst[in_idx + c] = delta[out_idx + ch_offset + c];
                    ch_offset += in_c;
                }
            }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
