//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "configuration.h"
#include "tensor_types.h"
#include "thread_safe_queue.h"

#include <atomic>
#include <mutex>
#include <thread>

namespace opennn
{

class Dataset;

struct BatchSlot
{
    Buffer buffer;
    Shape  shape;
    Index  features_number = 0;
    
    float* host = nullptr;
    Index  host_allocated_size = 0;
    uint16_t* host_bf16 = nullptr;
    Index     host_bf16_allocated_size = 0;
};

bool bf16_host_input_cast_enabled() noexcept;

struct Batch
{
    Batch(Index,
          const Dataset*,
          const Configuration::Resolved&,
          bool prefetch_only = false);
    ~Batch();

    Batch(const Batch&)            = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&)                 = delete;
    Batch& operator=(Batch&&)      = delete;

    void set(Index,
             const Dataset*,
             const Configuration::Resolved&,
             bool prefetch_only = false);

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
    Index current_sample_count = 0;
    bool needs_device_copy = true;
    bool input_is_bf16 = false;
    // A prefetch-only batch stages data into a separate fixed compute batch and is
    // never computed on itself, so on GPU it omits its own device input/target
    // buffers (it keeps only the pinned-host buffers and gather indices used to
    // stage). Saves one device batch copy per pool slot.
    bool prefetch_only = false;

    const Dataset* dataset = nullptr;
    Configuration::Resolved config;

    BatchSlot input;
    BatchSlot decoder;
    BatchSlot target;

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

    vector<TensorView> input_views_cache;
    TensorView target_view_cache;

    bool        device_gather = false;
    vector<int> gather_row_indices;
    Buffer      gather_indices_host{Device::CPU};
    Buffer      gather_indices_device{Device::CUDA};
    Index       input_col_offset = 0;
    Index       target_col_offset = 0;

    Index window_past = 0;
    Index window_future = 0;
    Index window_features = 0;      // input columns per timestep
    Index window_target_cols = 0;   // raw target columns
    Index window_matrix_rows = 0;
    bool  window_multi_target = false;

};

struct BatchPools
{
    ThreadSafeQueue<Batch*> training_empty_queue;
    ThreadSafeQueue<Batch*> validation_empty_queue;

    vector<unique_ptr<Batch>> training_pool;
    vector<unique_ptr<Batch>> validation_pool;
    unique_ptr<Batch> fixed_training_batch;
    vector<unique_ptr<Batch>> graph_slot_pool;

    bool validation_uses_training_pool = false;

    ThreadSafeQueue<Batch*>& validation_queue();
};

struct BatchPrefetchSession
{
    explicit BatchPrefetchSession(ThreadSafeQueue<Batch*>& queue, Index batches_number)
        : empty_queue(queue),
          ready_batches(size_t(batches_number))
    {
        for (atomic<Batch*>& batch : ready_batches)
            batch.store(nullptr, memory_order_relaxed);
    }

    ~BatchPrefetchSession()
    {
        for (jthread& thread : threads)
            thread.request_stop();

        empty_queue.close();

        threads.clear();

        empty_queue.reopen();
    }

    BatchPrefetchSession(const BatchPrefetchSession&) = delete;
    BatchPrefetchSession& operator=(const BatchPrefetchSession&) = delete;

    Batch* wait(Index iteration)
    {
        Batch* batch = nullptr;
        while (!(batch = ready_batches[size_t(iteration)].load(memory_order_acquire)))
        {
            rethrow_if_error();
            this_thread::yield();
        }

        return batch;
    }

    void capture_current_exception()
    {
        lock_guard<mutex> elock(error_mutex);
        if (!worker_error)
            worker_error = current_exception();
        error_pending.store(true, memory_order_release);
    }

    void rethrow_if_error()
    {
        if (!error_pending.load(memory_order_acquire)) return;

        exception_ptr e;
        {
            lock_guard<mutex> elock(error_mutex);
            swap(e, worker_error);
            error_pending.store(false, memory_order_release);
        }
        if (e) rethrow_exception(e);
    }

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
