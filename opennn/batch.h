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
    ~Batch();

    Batch(const Batch&)            = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&)                 = delete;
    Batch& operator=(Batch&&)      = delete;

    void set(const Index = 0, const Dataset* = nullptr);

    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              bool is_training = true,
              bool parallelize_samples = true);

    const vector<TensorView>& get_inputs() const
    {
#ifdef OPENNN_HAS_CUDA
        if (is_gpu()) return input_views_cache;
#endif
        return input_views_host_cache;
    }

    const TensorView& get_targets() const
    {
#ifdef OPENNN_HAS_CUDA
        if (is_gpu()) return target_view_cache;
#endif
        return target_view_host_cache;
    }

    Index get_samples_number() const;

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;
    Index current_sample_count = 0;     // set by fill(); may be < samples_number

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

    void copy_device_async(const Index, cudaStream_t, float* fp32_staging);

    Index get_input_elements() const { return samples_number * num_input_features; }

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

    bool needs_fp32_staging = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
