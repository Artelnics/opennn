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
#ifdef OPENNN_HAS_CUDA
#  include "kernel.cuh"
#endif

namespace opennn
{

void ConcatenationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const vector<TensorView>& inputs = get_inputs(forward_propagation, layer);
    TensorView& output = get_output(forward_propagation, layer);

    throw_if(inputs.size() != input_channels.size(),
             "Concatenation: input count mismatch.");

    const Index total_channels = output.shape[3];

#ifdef OPENNN_HAS_CUDA
    if (output.is_cuda())
    {
        const Index batch_size = output.shape[0];
        const Index height     = inputs[0].shape[1];
        const Index width      = inputs[0].shape[2];
        Index ch_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            concat_forward_slice_cuda(to_int(batch_size), to_int(height), to_int(width),
                                      to_int(input_channels[i]), to_int(total_channels), to_int(ch_offset),
                                      inputs[i].as<float>(), output.as<float>());
            ch_offset += input_channels[i];
        }
        return;
    }
#endif

    float* dst = output.as<float>();
    const Index pixels = output.size() / total_channels;

    #pragma omp parallel for
    for (Index pixel = 0; pixel < pixels; ++pixel)
    {
        float* out_row = dst + pixel * total_channels;
        Index ch_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const Index in_c = inputs[i].shape[3];
            memcpy(out_row + ch_offset, inputs[i].as<float>() + pixel * in_c, in_c * sizeof(float));
            ch_offset += in_c;
        }
    }
}

void ConcatenationOperator::back_propagate(ForwardPropagation&, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    const auto& backward_slots = back_propagation.backward_slots[layer];
    const bool needs_input_delta = ranges::any_of(input_delta_slots, [&](size_t slot)
    {
        return slot < backward_slots.size() && !backward_slots[slot].empty();
    });

    if (!needs_input_delta) return;

    const Index total_channels = output_delta.shape[3];

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
    {
        const Index batch_size = output_delta.shape[0];
        const Index height     = output_delta.shape[1];
        const Index width      = output_delta.shape[2];
        Index ch_offset = 0;
        for (size_t i = 0; i < input_channels.size(); ++i)
        {
            TensorView& in_delta = get_input_delta(back_propagation, layer, i);
            if (!in_delta.empty())
                concat_backward_slice_cuda(to_int(batch_size), to_int(height), to_int(width),
                                           to_int(input_channels[i]), to_int(total_channels), to_int(ch_offset),
                                           output_delta.as<float>(), in_delta.as<float>());
            ch_offset += input_channels[i];
        }
        return;
    }
#endif

    const float* delta = output_delta.as<float>();
    const Index pixels = output_delta.size() / total_channels;

    #pragma omp parallel for
    for (Index pixel = 0; pixel < pixels; ++pixel)
    {
        const float* delta_row = delta + pixel * total_channels;
        Index ch_offset = 0;
        for (size_t i = 0; i < input_channels.size(); ++i)
        {
            const Index in_c = input_channels[i];
            TensorView& in_delta = get_input_delta(back_propagation, layer, i);
            if (!in_delta.empty())
                memcpy(in_delta.as<float>() + pixel * in_c, delta_row + ch_offset, in_c * sizeof(float));
            ch_offset += in_c;
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
