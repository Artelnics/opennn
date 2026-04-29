//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"
#include "dataset.h"

namespace opennn
{


struct Batch
{
    Batch(const Index = 0, const Dataset* = nullptr);

    void set(const Index = 0, const Dataset* = nullptr);

    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              bool augment = false);

    // Returns input/target views routed to the active device based on the
    // resolved Configuration. There is no separate `_device` / `_active` /
    // host overload exposed publicly anymore.
    const vector<TensorView>& get_inputs() const
    {
#ifdef OPENNN_WITH_CUDA
        if (Configuration::instance().is_gpu()) return input_views_cache;
#endif
        return input_views_host_cache;
    }

    const TensorView& get_targets() const
    {
#ifdef OPENNN_WITH_CUDA
        if (Configuration::instance().is_gpu()) return target_view_cache;
#endif
        return target_view_host_cache;
    }

    Index get_samples_number() const;

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;

    const Dataset* dataset = nullptr;

    Buffer input;
    Shape input_shape;

    Buffer decoder;
    Shape decoder_shape;

    Buffer target;
    Shape target_shape;

    int input_contiguous = -1;
    int decoder_contiguous = -1;
    int target_contiguous = -1;

    void copy_device_async(const Index, cudaStream_t);

    // Cache of TensorView wrappers around the host pinned / device buffers, populated
    // by set(). Kept public for now so subclasses and helpers can read shapes; the
    // active subset is surfaced through get_inputs()/get_targets() above.
    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;       // GPU views; populated only on CUDA mode
    TensorView target_view_cache;

    Index num_input_features = 0;
    Index num_decoder_features = 0;
    Index num_target_features = 0;

    float* inputs_host = nullptr;
    float* decoder_host = nullptr;
    float* targets_host = nullptr;

    Index inputs_host_allocated_size = 0;
    Index decoder_host_allocated_size = 0;
    Index targets_host_allocated_size = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
