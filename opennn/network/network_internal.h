// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "opennn/network/layers/layer.h"

// Private implementation shared by network setup and persistence. Not installed.
namespace opennn::network_detail
{

static inline void validate_source_indices(const vector<Index>& sources, Index layer_index, Index layers_count)
{
    for (const Index src : sources | views::filter([](Index source) { return source >= 0; }))
        throw_if(src >= layers_count || src >= layer_index,
                 "Network: source index {} is not a previous layer for layer {}.", src, layer_index);
}

static inline void validate_source_arity(const Layer& layer,
                                  const vector<Index>& sources,
                                  Index layer_index)
{
    const Index expected_sources = layer.get_sources_number();

    throw_if(ssize(sources) != expected_sources,
             "Network: {} layer {} expects {} sources, got {}.",
             layer.get_name(), layer_index, expected_sources, sources.size());
}

#ifdef OPENNN_HAS_CUDA
static inline Index quantization_channel(const Index element_index,
                                         const Index row_length,
                                         const Index channels,
                                         const int axis)
{
    return axis == 0 ? element_index / row_length : element_index % channels;
}

static inline void finalize_int8_scales(vector<float>& absolute_maxima)
{
    for (float& scale : absolute_maxima)
        scale = scale > 0.0f ? scale / 127.0f : 1.0f;
}

static inline void quantize_int8_host(const float* values, const Index count,
                                      const Index base_index, const Index row_length,
                                      const Index channels, const int axis,
                                      const float* scales, int8_t* out)
{
    #pragma omp parallel for if(count > 4096)
    for (Index i = 0; i < count; ++i)
    {
        const Index channel =
            quantization_channel(base_index + i, row_length, channels, axis);
        out[i] = int8_t(clamp<long>(lroundf(values[i] / scales[channel]), -127, 127));
    }
}
#endif

}
