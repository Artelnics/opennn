//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"
#include "dataset.h"
#include "device_backend.h"
#include "memory_debug.h"

namespace opennn
{

bool bf16_host_input_cast_enabled() noexcept
{
    static const bool enabled = []
    {
        const char* flag = getenv("OPENNN_BF16_HOST_INPUT_CAST");
        return !flag || string(flag) != "0";
    }();

    return enabled;
}

Batch::Batch(const Index new_samples_number,
             const Dataset* new_dataset,
             const Configuration::Resolved& new_config,
             const bool new_prefetch_only)
{
    set(new_samples_number, new_dataset, new_config, new_prefetch_only);
}

void Batch::set(const Index new_samples_number,
                const Dataset* new_dataset,
                const Configuration::Resolved& new_config,
                const bool new_prefetch_only)
{
    throw_if(!new_dataset, "dataset is not set.");

    wait_h2d_complete();

    samples_number = new_samples_number;
    needs_device_copy = true;
    prefetch_only = new_prefetch_only;

    dataset = new_dataset;
    config = new_config;

    input.shape.clear();
    decoder.shape.clear();
    target.shape.clear();
    input.features_number = 0;
    decoder.features_number = 0;
    target.features_number = 0;
    input_views_host_cache.clear();
    target_view_host_cache = {};

    input_views_cache.clear();
    target_view_cache = {};

    const bool on_gpu = uses_cuda();
    const Device batch_device = on_gpu ? Device::CUDA : Device::CPU;
    input_is_bf16 = on_gpu
                 && config.training_type == Type::BF16
                 && dataset->supports_bf16_inputs();
    const Index input_device_bytes = input_is_bf16 ? Index(sizeof(bfloat16)) : Index(sizeof(float));

    const bool host_bf16_input_cast = input_is_bf16 && bf16_host_input_cast_enabled();

    auto setup_buffer = [&](const string& role, BatchSlot& slot, Index device_elem_bytes)
    {
        const Shape& dataset_shape = dataset->get_shape(role);

        if (dataset_shape.empty())
        {
            slot.buffer.resize_bytes(0, batch_device);
            return;
        }

        slot.shape = Shape({samples_number}).append(dataset_shape);
        slot.features_number = dataset_shape.size();

        const Index element_bytes = on_gpu ? device_elem_bytes : Index(sizeof(float));
        const Index device_bytes = slot.shape.size() * element_bytes;
        const Index allocated_device_bytes = (on_gpu && prefetch_only) ? Index(0) : device_bytes;
        slot.buffer.resize_bytes(allocated_device_bytes, batch_device);
        memory_debug::record("batch.device",
                             format("Batch::{}.buffer", role),
                             allocated_device_bytes,
                             format("samples={}", samples_number));

        if (!on_gpu) return;

        const Index host_values = samples_number * slot.features_number;
        if (host_values > slot.host_allocated_size)
        {
            device::deallocate_pinned_host(slot.host);
            slot.host = nullptr;
            slot.host_allocated_size = 0;
            slot.host = static_cast<float*>(
                device::allocate_pinned_host(host_values * Index(sizeof(float))));
            slot.host_allocated_size = host_values;
            memory_debug::record("batch.pinned_host",
                                  format("Batch::{}.host", role),
                                  host_values * Index(sizeof(float)),
                                  format("samples={}", samples_number));
        }

        const bool wants_bf16_host = role == "Input" && host_bf16_input_cast;
        if (wants_bf16_host && host_values > slot.host_bf16_allocated_size)
        {
            device::deallocate_pinned_host(slot.host_bf16);
            slot.host_bf16 = nullptr;
            slot.host_bf16_allocated_size = 0;
            slot.host_bf16 = static_cast<uint16_t*>(
                device::allocate_pinned_host(host_values * Index(sizeof(uint16_t))));
            slot.host_bf16_allocated_size = host_values;
            memory_debug::record("batch.pinned_host",
                                 format("Batch::{}.host_bf16", role),
                                 host_values * Index(sizeof(uint16_t)),
                                 format("samples={}", samples_number));
        }
        else if (!wants_bf16_host && slot.host_bf16)
        {
            device::deallocate_pinned_host(slot.host_bf16);
            slot.host_bf16 = nullptr;
            slot.host_bf16_allocated_size = 0;
        }
    };

    setup_buffer("Input",   input,   input_device_bytes);
    setup_buffer("Target",  target,  Index(sizeof(float)));
    setup_buffer("Decoder", decoder, Index(sizeof(float)));

    if (!decoder.shape.empty() && decoder.buffer.data)
        input_views_host_cache.emplace_back(decoder.buffer.as<float>(), decoder.shape, Type::FP32, Device::CPU);

    if (!input.shape.empty() && input.buffer.data)
        input_views_host_cache.emplace_back(input.buffer.as<float>(), input.shape, Type::FP32, Device::CPU);

    if (!target.shape.empty() && target.buffer.data)
        target_view_host_cache = TensorView(target.buffer.as<float>(), target.shape, Type::FP32, Device::CPU);

    const bool needs_fp32_staging = input_is_bf16
        && !host_bf16_input_cast
        && !prefetch_only
        && !(dataset && dataset->is_device_resident());
    const Index fp32_staging_bytes = needs_fp32_staging
        ? samples_number * input.features_number * Index(sizeof(float))
        : Index(0);
    fp32_staging.resize_bytes(fp32_staging_bytes, Device::CUDA);
    memory_debug::record("batch.device", "Batch::fp32_staging", fp32_staging_bytes,
                         format("samples={}", samples_number));

    const bool may_use_device_gather = on_gpu
        && dataset
        && (dataset->is_device_resident()
            || dataset->get_storage_mode() == Dataset::StorageMode::GPUPersistantData);
    if (may_use_device_gather)
    {
        gather_indices_host.resize_bytes(samples_number * Index(sizeof(int)), Device::CPU);
        gather_indices_device.resize_bytes(samples_number * Index(sizeof(int)), Device::CUDA);
        memory_debug::record("batch.device", "Batch::gather_indices_device",
                              samples_number * Index(sizeof(int)),
                              format("samples={}", samples_number));
    }
    else
    {
        gather_indices_host.resize_bytes(0, Device::CPU);
        gather_indices_device.resize_bytes(0, Device::CUDA);
    }

    if (on_gpu && !input.shape.empty() && input.buffer.data)
    {
        if (!decoder.shape.empty() && decoder.buffer.data)
            input_views_cache.emplace_back(decoder.buffer.data, decoder.shape, Type::FP32, Device::CUDA);

        input_views_cache.emplace_back(input.buffer.data, input.shape, input_is_bf16 ? Type::BF16 : Type::FP32, Device::CUDA);
    }

    if (on_gpu && !target.shape.empty() && target.buffer.data)
        target_view_cache = TensorView(target.buffer.data, target.shape, Type::FP32, Device::CUDA);

    if (on_gpu && !h2d_done_event)
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

Index Batch::get_samples_number() const
{
    return samples_number;
}

void Batch::print() const
{
    cout << "Batch\n"
         << "Inputs:\n"
         << "Input shape:" << input.shape << "\n";

    if (input.buffer.data)
    {
        if (uses_cuda())
            cout << "<CUDA input data not printed>";
        else if (input.shape.rank == 4)
            cout << TensorMap4(const_cast<float*>(input.buffer.as<float>()),
                               input.shape[0],
                               input.shape[1],
                               input.shape[2],
                               input.shape[3]);
        else if (input.shape.rank == 3)
            cout << TensorMap3(const_cast<float*>(input.buffer.as<float>()),
                               input.shape[0],
                               input.shape[1],
                               input.shape[2]);
        else if (input.shape.rank == 2)
            cout << MatrixMap(const_cast<float*>(input.buffer.as<float>()),
                              input.shape[0],
                              input.shape[1]);
    }

    cout << "\n";

    if (!decoder.shape.empty())
        cout << "Decoder:\n"
             << "Decoder shape:" << decoder.shape << "\n";

    cout << "Targets:\n"
         << "Target shape:" << target.shape << "\n";

    if (target.buffer.data && target.shape.rank == 2)
    {
        if (uses_cuda())
            cout << "<CUDA target data not printed>\n";
        else
            cout << MatrixMap(const_cast<float*>(target.buffer.as<float>()),
                              target.shape[0],
                              target.shape[1]) << "\n";
    }
}

bool Batch::is_empty() const
{
    return input.buffer.empty() && decoder.buffer.empty() && target.buffer.empty();
}

Batch::~Batch()
{
    wait_h2d_complete();
    device::deallocate_pinned_host(input.host);
    device::deallocate_pinned_host(input.host_bf16);
    device::deallocate_pinned_host(decoder.host);
    device::deallocate_pinned_host(decoder.host_bf16);
    device::deallocate_pinned_host(target.host);
    device::deallocate_pinned_host(target.host_bf16);
}

#ifdef OPENNN_HAS_CUDA

void Batch::copy_device_async(cudaStream_t stream)
{
    upload_to_device_batch_async(*this, stream);
}

void Batch::upload_to_device_batch_async(Batch& destination, cudaStream_t stream)
{
    const Index current_batch_size = samples_number;
    throw_if(!uses_cuda() || !destination.uses_cuda(),
             "Batch::upload_to_device_batch_async requires CUDA batches.");
    throw_if(current_batch_size > destination.samples_number,
             "Batch::upload_to_device_batch_async destination batch is too small.");

    needs_device_copy = false;
    destination.needs_device_copy = false;

    const Index input_values_count  = current_batch_size * input.features_number;
    const Index target_values_count = current_batch_size * target.features_number;

    if (device_gather && dataset && dataset->is_device_resident())
    {
        const float* matrix = dataset->get_device_data();
        const Index matrix_cols = dataset->get_device_data_columns();

        const Index index_bytes = current_batch_size * Index(sizeof(int));
        memcpy(gather_indices_host.data, gather_row_indices.data(), size_t(index_bytes));
        device::copy_async(gather_indices_device.data, gather_indices_host.data,
                           index_bytes,
                           device::CopyKind::HostToDevice, stream);

        const int* idx = gather_indices_device.as<int>();

        if (window_past > 0)
        {
            gather_window_rows_cuda(matrix, idx, destination.input.buffer.as<float>(),
                                    current_batch_size, window_past, window_features,
                                    matrix_cols, window_matrix_rows, input_col_offset, stream);
            gather_window_targets_cuda(matrix, idx, destination.target.buffer.as<float>(),
                                       current_batch_size, window_past, window_future,
                                       window_target_cols, window_multi_target,
                                       matrix_cols, window_matrix_rows, target_col_offset, stream);
            record_h2d_done(stream);
            return;
        }

        if (destination.input_is_bf16)
        {
            gather_rows_bf16_cuda(matrix, idx, destination.input.buffer.as<bfloat16>(),
                                  current_batch_size, input.features_number, matrix_cols, input_col_offset, stream);
        }
        else
        {
            gather_rows_cuda(matrix, idx, destination.input.buffer.as<float>(),
                             current_batch_size, input.features_number, matrix_cols, input_col_offset, stream);
        }

        gather_rows_cuda(matrix, idx, destination.target.buffer.as<float>(),
                         current_batch_size, target.features_number, matrix_cols, target_col_offset, stream);

        record_h2d_done(stream);
        return;
    }

    auto copy_to_device_async = [&](void* destination, const void* source, Index bytes) {
        device::copy_async(destination, source, bytes, device::CopyKind::HostToDevice, stream);
    };

    if (destination.input_is_bf16)
    {
        if (input.host_bf16)
        {
            const float* src = input.host;
            uint16_t* dst = input.host_bf16;
            #pragma omp parallel for if(input_values_count > 4096)
            for (Index i = 0; i < input_values_count; ++i)
                dst[i] = static_cast<uint16_t>(bit_cast<uint32_t>(src[i]) >> 16);

            copy_to_device_async(destination.input.buffer.as<bfloat16>(),
                                 input.host_bf16,
                                 input_values_count * Index(sizeof(uint16_t)));
        }
        else
        {
            const Index staging_bytes = input_values_count * Index(sizeof(float));
            if (destination.fp32_staging.bytes < staging_bytes)
            {
                const Index before = destination.fp32_staging.bytes;
                destination.fp32_staging.resize_bytes(staging_bytes, Device::CUDA);
                memory_debug::record("batch.device", "Batch::fp32_staging",
                                     destination.fp32_staging.bytes - before,
                                     format("samples={}", current_batch_size));
            }
            copy_to_device_async(destination.fp32_staging.as<float>(), input.host, input_values_count * sizeof(float));
            cast_fp32_to_bf16(input_values_count,
                              destination.fp32_staging.as<float>(),
                              destination.input.buffer.as<bfloat16>(),
                              stream);
        }
    }
    else
    {
        copy_to_device_async(destination.input.buffer.as<float>(), input.host, input_values_count * sizeof(float));
    }

    if (!decoder.shape.empty())
    {
        const Index decoder_values_count = current_batch_size * decoder.features_number;
        copy_to_device_async(destination.decoder.buffer.as<float>(), decoder.host, decoder_values_count * sizeof(float));
    }

    copy_to_device_async(destination.target.buffer.as<float>(), target.host, target_values_count * sizeof(float));

    record_h2d_done(stream);
}

#else

void Batch::copy_device_async(cudaStream_t)
{
    throw runtime_error("Batch::copy_device_async requires CUDA support.");
}

void Batch::upload_to_device_batch_async(Batch&, cudaStream_t)
{
    throw runtime_error("Batch::upload_to_device_batch_async requires CUDA support.");
}

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
    return validation_uses_training_pool
        ? training_empty_queue
        : validation_empty_queue;
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
    Batch* batch = nullptr;
    while (!(batch = ready_batches[size_t(iteration)].load(memory_order_acquire)))
    {
        rethrow_if_error();
        this_thread::yield();
    }

    return batch;
}

void BatchPrefetchSession::capture_current_exception()
{
    lock_guard<mutex> elock(error_mutex);
    if (!worker_error)
        worker_error = current_exception();
    error_pending.store(true, memory_order_release);
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
