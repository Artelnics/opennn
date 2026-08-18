//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/batch.h"
#include "opennn/core/string_utilities.h"
#include "opennn/dataset/dataset.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/dataset/kernel_gather.cuh"

namespace opennn
{

bool bf16_host_input_cast_enabled() noexcept
{
    static const bool enabled = env_flag_enabled("OPENNN_BF16_HOST_INPUT_CAST", true);

    return enabled;
}

Batch::Batch(const Index new_batch_size,
             const Dataset* new_dataset,
             const Configuration::EffectiveConfig& new_config,
             const bool new_prefetch_only)
{
    set(new_batch_size, new_dataset, new_config, new_prefetch_only);
}

void Batch::set(const Index new_batch_size,
                const Dataset* new_dataset,
                const Configuration::EffectiveConfig& new_config,
                const bool new_prefetch_only)
{
    throw_if(!new_dataset, "dataset is not set.");

    wait_h2d_complete();

    batch_size = new_batch_size;
    dataset = new_dataset;

    input.shape.clear();
    decoder.shape.clear();
    target.shape.clear();

    input.contiguous.reset();
    decoder.contiguous.reset();
    target.contiguous.reset();

    device_gather.reset();

    input_views_host_cache.clear();
    target_view_host_cache = {};

    input_views_cache.clear();
    target_view_cache = {};

    const bool on_gpu = new_config.device == Device::CUDA
                     && device::is_cuda_build();

    const Device batch_device = on_gpu ? Device::CUDA : Device::CPU;

    const Type input_type =
        on_gpu
        && activation_dtype(new_config.training_type) == Type::BF16
        && dataset->supports_bf16_inputs()
            ? Type::BF16
            : Type::FP32;

    const auto setup_buffer = [&](const string& role, BatchSlot& slot, Type type)
    {
        slot.type = type;

        const Shape& dataset_shape = dataset->get_shape(role);

        if (dataset_shape.empty())
        {
            slot.buffer.resize_bytes(0, batch_device);
            return;
        }

        slot.shape = Shape({batch_size}).append(dataset_shape);

        const Index element_bytes = on_gpu
            ? type_bytes(type)
            : Index(sizeof(float));

        const Index device_bytes = slot.shape.size() * element_bytes;

        const Index allocated_device_bytes =
            on_gpu && new_prefetch_only ? Index(0) : device_bytes;

        slot.buffer.resize_bytes(allocated_device_bytes, batch_device);

        memory_debug::record(
            "batch.device",
            format("Batch::{}.buffer", role),
            allocated_device_bytes,
            format("samples={}", batch_size));

        if (!on_gpu) return;

        const Index host_values = slot.shape.size();

        if (host_values > slot.host_allocated_size)
        {
            device::deallocate_pinned_host(slot.host);

            slot.host = static_cast<float*>(
                device::allocate_pinned_host(
                    host_values * Index(sizeof(float))));

            slot.host_allocated_size = host_values;

            memory_debug::record(
                "batch.pinned_host",
                format("Batch::{}.host", role),
                host_values * Index(sizeof(float)),
                format("samples={}", batch_size));
        }
    };

    setup_buffer("Input",   input,   input_type);
    setup_buffer("Target",  target,  Type::FP32);
    setup_buffer("Decoder", decoder, Type::FP32);

    if (decoder.has_data())
        input_views_host_cache.emplace_back(
            decoder.buffer.as<float>(),
            decoder.shape,
            Type::FP32,
            Device::CPU);

    if (input.has_data())
        input_views_host_cache.emplace_back(
            input.buffer.as<float>(),
            input.shape,
            Type::FP32,
            Device::CPU);

    if (target.has_data())
        target_view_host_cache = TensorView(
            target.buffer.as<float>(),
            target.shape,
            Type::FP32,
            Device::CPU);

    if (!on_gpu)
    {
        if (input_host_bf16)
        {
            device::deallocate_pinned_host(input_host_bf16);
            input_host_bf16 = nullptr;
            input_host_bf16_allocated_size = 0;
        }

        fp32_staging.resize_bytes(0, Device::CUDA);
        gather_indices_device.resize_bytes(0, Device::CUDA);

        return;
    }

    const bool host_bf16_input_cast =
        input.type == Type::BF16
        && bf16_host_input_cast_enabled();

    const Index input_host_values = input.shape.size();

    if (host_bf16_input_cast
        && input_host_values > input_host_bf16_allocated_size)
    {
        device::deallocate_pinned_host(input_host_bf16);

        input_host_bf16 = static_cast<uint16_t*>(
            device::allocate_pinned_host(
                input_host_values * Index(sizeof(uint16_t))));

        input_host_bf16_allocated_size = input_host_values;

        memory_debug::record(
            "batch.pinned_host",
            "Batch::input_host_bf16",
            input_host_values * Index(sizeof(uint16_t)),
            format("samples={}", batch_size));
    }
    else if (!host_bf16_input_cast && input_host_bf16)
    {
        device::deallocate_pinned_host(input_host_bf16);
        input_host_bf16 = nullptr;
        input_host_bf16_allocated_size = 0;
    }

    const bool needs_fp32_staging =
        input.type == Type::BF16
        && !host_bf16_input_cast
        && !new_prefetch_only
        && !dataset->uses_device_residency();

    const Index fp32_staging_bytes = needs_fp32_staging
        ? input.shape.size() * Index(sizeof(float))
        : Index(0);

    fp32_staging.resize_bytes(fp32_staging_bytes, Device::CUDA);

    memory_debug::record(
        "batch.device",
        "Batch::fp32_staging",
        fp32_staging_bytes,
        format("samples={}", batch_size));

    const bool may_use_device_gather =
        dataset->uses_device_residency();

    const Index gather_indices_bytes = may_use_device_gather
        ? batch_size * Index(sizeof(int))
        : Index(0);

    if (gather_indices_bytes > gather_indices_host_allocated_bytes)
    {
        device::deallocate_pinned_host(gather_indices_host);

        gather_indices_host = static_cast<int*>(
            device::allocate_pinned_host(gather_indices_bytes));

        gather_indices_host_allocated_bytes = gather_indices_bytes;
    }

    gather_indices_device.resize_bytes(
        gather_indices_bytes,
        Device::CUDA);

    if (may_use_device_gather)
    {
        memory_debug::record(
            "batch.device",
            "Batch::gather_indices_device",
            gather_indices_bytes,
            format("samples={}", batch_size));
    }

    if (input.has_data())
    {
        if (decoder.has_data())
        {
            input_views_cache.emplace_back(
                decoder.buffer.data(),
                decoder.shape,
                decoder.type,
                Device::CUDA);
        }

        input_views_cache.emplace_back(
            input.buffer.data(),
            input.shape,
            input.type,
            Device::CUDA);
    }

    if (target.has_data())
    {
        target_view_cache = TensorView(
            target.buffer.data(),
            target.shape,
            target.type,
            Device::CUDA);
    }

    if (!h2d_done_event)
        h2d_done_event.create();
}

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 FillMode mode)
{
    dataset->fill_batch(*this,
                        sample_indices,
                        input_indices,
                        decoder_indices,
                        target_indices,
                        mode);
}

bool Batch::is_empty() const
{
    return input.buffer.empty() && decoder.buffer.empty() && target.buffer.empty();
}

Batch::~Batch()
{
    wait_h2d_complete();
    device::deallocate_pinned_host(input.host);
    device::deallocate_pinned_host(input_host_bf16);
    device::deallocate_pinned_host(decoder.host);
    device::deallocate_pinned_host(target.host);
    device::deallocate_pinned_host(gather_indices_host);
}

#ifdef OPENNN_HAS_CUDA

void Batch::upload_to_device_batch_async(Batch& destination, cudaStream_t stream)
{
    const Index current_batch_size = batch_size;

    throw_if(!uses_cuda() || !destination.uses_cuda(),
             "Batch::upload_to_device_batch_async requires CUDA batches.");
    throw_if(current_batch_size > destination.batch_size,
             "Batch::upload_to_device_batch_async destination batch is too small.");

    const Index input_values_count = input.shape.size();
    const Index target_values_count = target.shape.size();
    const Index input_values_per_sample = current_batch_size > 0
        ? input_values_count / current_batch_size
        : 0;
    const Index target_values_per_sample = current_batch_size > 0
        ? target_values_count / current_batch_size
        : 0;

    if (device_gather && dataset && dataset->is_device_resident())
    {
        const DeviceGather& gather = *device_gather;
        const float* matrix = dataset->get_device_data();
        const Index matrix_cols = dataset->get_device_data_columns();

        const Index index_bytes = current_batch_size * Index(sizeof(int));
        memcpy(gather_indices_host, gather.row_indices.data(), size_t(index_bytes));
        device::copy_async(gather_indices_device.data(), gather_indices_host,
                           index_bytes,
                           device::CopyKind::HostToDevice, stream);

        const int* idx = gather_indices_device.as<int>();

        if (gather.window_past > 0)
        {
            const WindowLayout window{current_batch_size, gather.window_past,
                                      matrix_cols, gather.window_matrix_rows};

            gather_window_inputs_cuda(matrix, idx, destination.input.buffer.as<float>(), window,
                                      gather.window_features, gather.input_col_offset, stream);
            gather_window_targets_cuda(matrix, idx, destination.target.buffer.as<float>(), window,
                                       gather.window_future, gather.window_target_cols,
                                       gather.window_multi_target, gather.target_col_offset, stream);
            return record_h2d_done(stream);
        }

        gather_rows_cuda(matrix, idx, destination.input.buffer.data(),
                         destination.input.type == Type::BF16,
                         current_batch_size, input_values_per_sample,
                         matrix_cols, gather.input_col_offset, stream);

        gather_rows_cuda(matrix, idx, destination.target.buffer.data(), false,
                         current_batch_size, target_values_per_sample,
                         matrix_cols, gather.target_col_offset, stream);

        return record_h2d_done(stream);
    }

    if (destination.input.type == Type::BF16)
    {
        if (input_host_bf16)
        {
            float_2_bfloat16_host(input_values_count, input.host, input_host_bf16);

            device::copy_async(destination.input.buffer.as<bfloat16>(),
                               input_host_bf16,
                               input_values_count * Index(sizeof(uint16_t)),
                               device::CopyKind::HostToDevice, stream);
        }
        else
        {
            const Index staging_bytes = input_values_count * Index(sizeof(float));
            if (destination.fp32_staging.byte_size() < staging_bytes)
            {
                const Index before = destination.fp32_staging.byte_size();
                destination.fp32_staging.resize_bytes(staging_bytes, Device::CUDA);
                memory_debug::record("batch.device", "Batch::fp32_staging",
                                     destination.fp32_staging.byte_size() - before,
                                     format("samples={}", current_batch_size));
            }
            
            device::copy_async(destination.fp32_staging.as<float>(), input.host, input_values_count * sizeof(float),
                               device::CopyKind::HostToDevice, stream);

            cast_fp32_to_bf16(input_values_count,
                              destination.fp32_staging.as<float>(),
                              destination.input.buffer.as<bfloat16>(),
                              stream);
        }
    }
    else
    {
        device::copy_async(destination.input.buffer.as<float>(), input.host, input_values_count * sizeof(float),
                           device::CopyKind::HostToDevice, stream);
    }

    if (!decoder.shape.empty())
    {
        const Index decoder_values_count = decoder.shape.size();
        device::copy_async(destination.decoder.buffer.as<float>(), decoder.host, decoder_values_count * sizeof(float),
                           device::CopyKind::HostToDevice, stream);
    }

    device::copy_async(destination.target.buffer.as<float>(), target.host, target_values_count * sizeof(float),
                       device::CopyKind::HostToDevice, stream);

    record_h2d_done(stream);
}

#else

void Batch::upload_to_device_batch_async(Batch&, cudaStream_t) OPENNN_CUDA_STUB_BODY(Batch::upload_to_device_batch_async)

#endif

void Batch::record_h2d_done(cudaStream_t stream)
{
    if (!h2d_done_event)
        h2d_done_event.create();

    device::record_event(h2d_done_event, stream);
    h2d_done_recorded = true;
}

void Batch::wait_h2d_complete()
{
    if (h2d_done_recorded)
    {
        device::synchronize_event(h2d_done_event);
        h2d_done_recorded = false;
    }
}

void Batch::wait_h2d_on_compute_stream()
{
    if (h2d_done_recorded)
        device::stream_wait_event(device::get_compute_stream(), h2d_done_event);
}

ThreadSafeQueue<Batch*>& BatchPools::validation_queue()
{

    return validation_pool.empty()
        ? training_empty_queue
        : validation_empty_queue;
}

// Non-null marker published into still-idle ready slots when a worker fails. A consumer
// parked in atomic::wait() only unblocks on a value change, so the failure has to change
// the value it is waiting on; without this it would sleep through the error.
static Batch* aborted_slot()
{
    static int marker = 0;
    return reinterpret_cast<Batch*>(&marker);
}

BatchPrefetchSession::BatchPrefetchSession(ThreadSafeQueue<Batch*>& queue, const Index batches_number)
    : empty_queue(queue),
      ready_batches(size_t(batches_number))
{
    for (atomic<Batch*>& batch : ready_batches)
        batch.store(nullptr, memory_order_relaxed);
}

BatchPrefetchSession::~BatchPrefetchSession()
{
    for (jthread& thread : threads)
        thread.request_stop();

    empty_queue.close();

    threads.clear();

    empty_queue.reopen();
}

Batch* BatchPrefetchSession::wait(const Index iteration)
{
    atomic<Batch*>& ready = ready_batches[size_t(iteration)];

    while (true)
    {
        Batch* const batch = ready.load(memory_order_acquire);

        rethrow_if_error();

        throw_if(batch == aborted_slot(),
                 "BatchPrefetchSession: prefetch worker aborted without an exception.");

        if (batch) return batch;

        ready.wait(nullptr, memory_order_acquire);
    }
}

void BatchPrefetchSession::capture_current_exception()
{
    {
        lock_guard<mutex> elock(error_mutex);
        if (!worker_error)
            worker_error = current_exception();
        error_pending.store(true, memory_order_release);
    }

    // Publish the abort marker before waking, so a consumer that parks between its error
    // check and its wait() still sees a changed value and cannot miss the notification.
    for (atomic<Batch*>& ready : ready_batches)
    {
        Batch* idle = nullptr;
        ready.compare_exchange_strong(idle, aborted_slot(), memory_order_release, memory_order_relaxed);
        ready.notify_all();
    }
}

void BatchPrefetchSession::rethrow_if_error()
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

}
