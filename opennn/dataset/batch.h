//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/thread_safe_queue.h"

#include <atomic>
#include <mutex>
#include <thread>

namespace opennn
{

class Dataset;

enum class FillMode { Training, Validation, Inference };

struct BatchSlot
{
    Buffer buffer;
    Shape  shape;
    Type   type = Type::FP32;
    
    optional<bool> contiguous;

    float* host = nullptr;
    Index  host_allocated_size = 0;
};

struct DeviceGather
{
    vector<int> row_indices;
    Index input_col_offset = 0;
    Index target_col_offset = 0;

    Index window_past = 0;
    Index window_future = 0;
    Index window_features = 0;
    Index window_target_cols = 0;
    Index window_matrix_rows = 0;
    bool window_multi_target = false;
};

bool bf16_host_input_cast_enabled() noexcept;

struct Batch
{
    Batch(Index,
          const Dataset*,
          const Configuration::EffectiveConfig&,
          bool prefetch_only = false);
    ~Batch();

    Batch(const Batch&)            = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&)                 = delete;
    Batch& operator=(Batch&&)      = delete;

    void set(Index,
             const Dataset*,
             const Configuration::EffectiveConfig&,
             bool prefetch_only = false);

    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              FillMode mode = FillMode::Training);

    const vector<TensorView>& get_inputs() const
    {
        return uses_cuda() ? input_views_cache : input_views_host_cache;
    }

    const TensorView& get_targets() const
    {
        return uses_cuda() ? target_view_cache : target_view_host_cache;
    }

    bool uses_cuda() const
    {
        return input.buffer.get_device() == Device::CUDA && device::is_cuda_build();
    }

    Index get_batch_size() const { return batch_size; }

    bool is_empty() const;

    Index batch_size = 0;
    const Dataset* dataset = nullptr;

    BatchSlot input;
    BatchSlot decoder;
    BatchSlot target;

    void upload_to_device_batch_async(Batch&, cudaStream_t);

    void record_h2d_done(cudaStream_t);

    uint16_t* input_host_bf16 = nullptr;
    Index input_host_bf16_allocated_size = 0;
    Buffer fp32_staging{Device::CUDA};

    CudaEvent h2d_done_event;
    bool h2d_done_recorded = false;

    void wait_h2d_complete();
    void wait_h2d_on_compute_stream();

    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;
    TensorView target_view_cache;

    optional<DeviceGather> device_gather;
    // Pinned, like every other host staging buffer here: the per-step index
    // upload behind the device gather is a cudaMemcpyAsync, and from pageable
    // memory that goes through a driver bounce buffer and blocks the host.
    int* gather_indices_host = nullptr;
    Index gather_indices_host_allocated_bytes = 0;
    Buffer gather_indices_device{Device::CUDA};
};

struct BatchPools
{
    ThreadSafeQueue<Batch*> training_empty_queue;
    ThreadSafeQueue<Batch*> validation_empty_queue;

    vector<unique_ptr<Batch>> training_pool;
    vector<unique_ptr<Batch>> validation_pool;

    ThreadSafeQueue<Batch*>& validation_queue();
};

struct BatchPrefetchSession
{
    BatchPrefetchSession(ThreadSafeQueue<Batch*>&, Index batches_number);
    ~BatchPrefetchSession();

    BatchPrefetchSession(const BatchPrefetchSession&) = delete;
    BatchPrefetchSession& operator=(const BatchPrefetchSession&) = delete;

    Batch* wait(Index iteration);

    void capture_current_exception();

    void rethrow_if_error();

    ThreadSafeQueue<Batch*>& empty_queue;
    vector<atomic<Batch*>> ready_batches;
    atomic<Index> next_iteration{0};
    mutex error_mutex;
    exception_ptr worker_error;
    atomic<bool> error_pending{false};
    vector<jthread> threads;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
