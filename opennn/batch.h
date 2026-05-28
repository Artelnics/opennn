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

#ifdef OPENNN_HAS_CUDA
    void copy_device_async(const Index, cudaStream_t, float* fp32_staging);

    // Recorded at the end of copy_device_async() on compute_stream; waited on
    // by wait_h2d_complete() before the worker recycles this Batch. Created
    // lazily on first H→D copy (CPU-only Batches never touch them).
    cudaEvent_t h2d_done_event = nullptr;
    bool        h2d_done_recorded = false;
#endif

    // Block until the most recent copy_device_async() has finished reading
    // inputs_host / decoder_host / targets_host on the device. Workers MUST
    // call this before re-filling these pinned buffers, otherwise the DMA in
    // flight will race against the new fill(). No-op when no copy is pending
    // (first use of the batch, or already drained).
    void wait_h2d_complete();

    Index get_input_elements() const { return samples_number * input_features_number; }

    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;       // GPU views; populated only on CUDA mode
    TensorView target_view_cache;

    Index input_features_number = 0;
    Index decoder_features_number = 0;
    Index target_features_number = 0;

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
