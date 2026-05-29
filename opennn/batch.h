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
    Batch(Index,
          const Dataset*,
          const Configuration::Resolved&);
    ~Batch();

    Batch(const Batch&)            = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&)                 = delete;
    Batch& operator=(Batch&&)      = delete;

    void set(Index,
             const Dataset*,
             const Configuration::Resolved&);

    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              bool is_training = true,
              bool parallelize_samples = true);

    const vector<TensorView>& get_inputs() const
    {
        if (uses_cuda()) return input_views_cache;
        return input_views_host_cache;
    }

    const TensorView& get_targets() const
    {
        if (uses_cuda()) return target_view_cache;
        return target_view_host_cache;
    }

    bool uses_cuda() const
    {
#ifdef OPENNN_HAS_CUDA
        return config.device == Device::CUDA;
#else
        return false;
#endif
    }

    Index get_samples_number() const;

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;
    Index current_sample_count = 0;     // May be < samples_number for the last batch.

    const Dataset* dataset = nullptr;
    Configuration::Resolved config;

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
    void copy_device_async(cudaStream_t);

    void gather_device_async(const Index current_batch_size,
                             const Index* device_row_indices,
                             const float* resident_input, Index resident_input_features,
                             const float* resident_target, Index resident_target_features);

    Buffer fp32_staging{Device::CUDA};

    cudaEvent_t h2d_done_event = nullptr;
    bool        h2d_done_recorded = false;
#endif

    void wait_h2d_complete();

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
