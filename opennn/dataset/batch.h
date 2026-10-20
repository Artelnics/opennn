// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/thread_safe_queue.h"

#include <atomic>
#include <array>
#include <mutex>
#include <thread>
#include <utility>

namespace opennn
{

class Dataset;

enum class FillMode { Training, Validation, Inference };

// Constant for a whole run, where the samples a batch draws change every call.
struct FeatureSelection
{
    vector<Index> inputs;
    vector<Index> decoder;
    vector<Index> targets;
};

struct BatchSlot
{
    bool has_data() const noexcept { return !shape.empty() && buffer.data(); }

    Buffer buffer;
    Shape  shape;
    Type   type = Type::FP32;
    
    ColumnContiguity column_contiguity = ColumnContiguity::Unknown;

    device::PinnedBuffer host;
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

struct PackedImageInput
{
    int channels = 0;
    array<float, 4> scale{};
    array<float, 4> offset{};
};

// Where a CUDA batch keeps its samples. Prefetch batches are filled on the host
// without device storage; eager training slots only receive uploads, so they
// need no pinned host staging. CPU batches ignore the setting.
enum class BatchStorage { HostAndDevice, HostOnly, DeviceOnly };

struct Batch
{
    Batch(Index,
          const Dataset*,
          const EffectiveConfig&,
          BatchStorage);

    Batch(Index new_batch_size,
          const Dataset* new_dataset,
          const EffectiveConfig& new_config,
          bool prefetch_only = false)
        : Batch(new_batch_size, new_dataset, new_config,
                prefetch_only ? BatchStorage::HostOnly : BatchStorage::HostAndDevice) {}

    ~Batch();

    Batch(const Batch&)            = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&)                 = delete;
    Batch& operator=(Batch&&)      = delete;

    void set(Index,
             const Dataset*,
             const EffectiveConfig&,
             BatchStorage);

    void set(Index new_batch_size,
             const Dataset* new_dataset,
             const EffectiveConfig& new_config,
             bool prefetch_only = false)
    {
        set(new_batch_size, new_dataset, new_config,
            prefetch_only ? BatchStorage::HostOnly : BatchStorage::HostAndDevice);
    }

    void fill(const vector<Index>& sample_indices,
              const FeatureSelection&,
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

    bool is_empty() const { return input.buffer.empty() && decoder.buffer.empty() && target.buffer.empty(); }

    Index batch_size = 0;
    const Dataset* dataset = nullptr;

    BatchSlot input;
    BatchSlot decoder;
    BatchSlot target;

    void upload_to_device_batch_async(Batch&, DeviceStream);

    void record_h2d_done(DeviceStream);

    device::PinnedBuffer input_host_bf16;
    Buffer fp32_staging{Device::CUDA};

    device::CudaEvent h2d_done_event;
    bool h2d_done_recorded = false;

    void wait_h2d_complete();
    void wait_h2d_on_compute_stream();

    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;
    TensorView target_view_cache;

    optional<DeviceGather> device_gather;
    // Only the eager training staging pool opts in; ordinary batches retain
    // their FP32 host representation, including CUDA graph staging.
    bool pack_image_input = false;
    optional<PackedImageInput> packed_image_input;
    Buffer packed_input_staging{Device::CUDA};
    device::PinnedBuffer gather_indices_host;
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

    bool publish(Index iteration, Batch* batch);

    bool acquire(Batch*& batch) { return empty_queue.wait_pop(batch); }
    void release(Batch* batch) { empty_queue.push(batch); }
    Index claim_iteration() { return next_iteration.fetch_add(1); }

    template <typename Worker>
    void add_worker(Worker&& worker)
    {
        threads.emplace_back(std::forward<Worker>(worker));
    }

    void capture_current_exception();

    void rethrow_if_error();

private:

    enum class SlotState : uint8_t { Pending, Ready, Aborted };

    ThreadSafeQueue<Batch*>& empty_queue;
    vector<Batch*> ready_batches;
    vector<atomic<SlotState>> slot_states;
    atomic<Index> next_iteration{0};
    mutex error_mutex;
    exception_ptr worker_error;
    atomic<bool> error_pending{false};
    vector<jthread> threads;
};

}
