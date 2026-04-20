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

    vector<TensorView> get_inputs() const;

    TensorView get_targets() const;

    // Returns the target view for the currently active device (GPU if enabled, else host).
    TensorView get_targets_active() const
    {
#ifdef OPENNN_WITH_CUDA
        return Device::instance().is_gpu() ? get_targets_device() : get_targets();
#else
        return get_targets();
#endif
    }

    Index get_samples_number() const;

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;

    const Dataset* dataset = nullptr;

    Memory input;
    Shape input_shape;

    Memory decoder;
    Shape decoder_shape;

    Memory target;
    Shape target_shape;

    int input_contiguous = -1;
    int decoder_contiguous = -1;
    int target_contiguous = -1;

    void fill_host(const vector<Index>&,
                   const vector<Index>&,
                   const vector<Index>&,
                   const vector<Index>&);

    void copy_device_async(const Index, cudaStream_t);

    const vector<TensorView>& get_inputs_device() const;
    const TensorView& get_targets_device() const;

    vector<TensorView> input_views_cache;
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
