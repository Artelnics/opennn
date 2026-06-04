//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "configuration.h"
#include "tensor_utilities.h"
#include "dataset.h"

namespace opennn
{

struct SlotBuffers
{
    Buffer buffer;
    Shape  shape;
    Index  features_number = 0;
    float* host = nullptr;
    Index  host_allocated_size = 0;
};

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
              bool is_training = true);

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
        return config.device == Device::CUDA && device::is_cuda_build();
    }

    Index get_samples_number() const;

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;
    Index current_sample_count = 0;     // May be < samples_number for the last batch.
    bool needs_device_copy = true;
    bool use_device_data_buffer = false;

    const Dataset* dataset = nullptr;
    Configuration::Resolved config;

    SlotBuffers input;
    SlotBuffers decoder;
    SlotBuffers target;

    int input_contiguous = -1;
    int decoder_contiguous = -1;
    int target_contiguous = -1;

    void copy_device_async(cudaStream_t);

    void gather_device_async(const vector<Index>& row_indices,
                             const float* source_data,
                             Index source_features,
                             const vector<Index>& input_feature_indices,
                             const vector<Index>& target_feature_indices);

    void record_h2d_done(cudaStream_t);

    Buffer fp32_staging{Device::CUDA};
    Buffer device_data_row_indices{Device::CUDA};
    Buffer device_data_input_feature_indices{Device::CUDA};
    Buffer device_data_target_feature_indices{Device::CUDA};
    vector<Index> device_data_row_indices_host;

    CudaEvent h2d_done_event;
    bool        h2d_done_recorded = false;

    void wait_h2d_complete();

    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;       // GPU views; populated only on CUDA mode
    TensorView target_view_cache;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
