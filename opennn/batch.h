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

    const Dataset* dataset = nullptr;
    Configuration::Resolved config;

    SlotBuffers input;
    SlotBuffers decoder;
    SlotBuffers target;

    int input_contiguous = -1;
    int decoder_contiguous = -1;
    int target_contiguous = -1;

    void copy_device_async(cudaStream_t);
    void upload_to_device_batch_async(Batch&, cudaStream_t);

    void record_h2d_done(cudaStream_t);

    Buffer fp32_staging{Device::CUDA};

    CudaEvent h2d_done_event;
    bool        h2d_done_recorded = false;

    void wait_h2d_complete();
    void wait_h2d_on_compute_stream();

    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;       // GPU views; populated only on CUDA mode
    TensorView target_view_cache;

    // GPU-resident gather (prototype): when the dataset is device-resident, the
    // batch's row indices are stashed here (host int32) so the H2D step can
    // gather rows on-device instead of copying a host-staged batch. The feature
    // blocks are contiguous, so only their starting column is needed.
    bool        device_gather = false;
    vector<int> gather_row_indices;
    Buffer      gather_indices_device{Device::CUDA};
    Index       input_col_offset = 0;
    Index       target_col_offset = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
