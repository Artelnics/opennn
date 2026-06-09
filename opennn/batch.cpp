//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"
#include "device_backend.h"

namespace opennn
{

Batch::Batch(const Index new_samples_number,
             const Dataset* new_dataset,
             const Configuration::Resolved& new_config)
{
    set(new_samples_number, new_dataset, new_config);
}

void Batch::set(const Index new_samples_number,
                const Dataset* new_dataset,
                const Configuration::Resolved& new_config)
{
    throw_if(!new_dataset, "dataset is not set.");

    wait_h2d_complete();

    samples_number = new_samples_number;
    current_sample_count = new_samples_number;
    needs_device_copy = true;

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
    const bool bf16_input = on_gpu
                         && config.training_type == Type::BF16
                         && dataset->supports_bf16_inputs();
    const Index input_device_bytes = bf16_input ? Index(sizeof(bfloat16)) : Index(sizeof(float));

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
        slot.buffer.resize_bytes(slot.shape.size() * element_bytes, batch_device);

        if (!on_gpu) return;

        if (const Index size = samples_number * slot.features_number; size > slot.host_allocated_size)
        {
            device::deallocate_pinned_host(slot.host);
            slot.host = nullptr;
            slot.host_allocated_size = 0;
            slot.host = static_cast<float*>(
                device::allocate_pinned_host(size * Index(sizeof(float))));
            slot.host_allocated_size = size;
        }
    };

    setup_buffer("Input",   input,   input_device_bytes);
    setup_buffer("Target",  target,  Index(sizeof(float)));
    setup_buffer("Decoder", decoder, Index(sizeof(float)));

    if (!decoder.shape.empty())
        input_views_host_cache.emplace_back(decoder.buffer.as<float>(), decoder.shape, Type::FP32, Device::CPU);

    if (!input.shape.empty())
        input_views_host_cache.emplace_back(input.buffer.as<float>(), input.shape, Type::FP32, Device::CPU);

    if (!target.shape.empty())
        target_view_host_cache = TensorView(target.buffer.as<float>(), target.shape, Type::FP32, Device::CPU);

    fp32_staging.resize_bytes(bf16_input
        ? samples_number * input.features_number * Index(sizeof(float))
        : Index(0),
        Device::CUDA);

    // Pre-allocate the GPU-resident gather index buffer at warmup time so the
    // per-batch upload never resizes inside the (allocation-frozen) train loop.
    if (on_gpu)
        gather_indices_device.resize_bytes(samples_number * Index(sizeof(int)), Device::CUDA);

    if (on_gpu && !input.shape.empty() && input.buffer.data)
    {
        if (!decoder.shape.empty() && decoder.buffer.data)
            input_views_cache.emplace_back(decoder.buffer.data, decoder.shape, Type::FP32, Device::CUDA);

        input_views_cache.emplace_back(input.buffer.data, input.shape, bf16_input ? Type::BF16 : Type::FP32, Device::CUDA);
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
                 bool is_training)
{
    dataset->fill_batch(*this,
                        sample_indices,
                        input_indices,
                        decoder_indices,
                        target_indices,
                        is_training);
}

Index Batch::get_samples_number() const
{
    return samples_number;
}

void Batch::print() const
{
    cout << "Batch" << "\n"
         << "Inputs:" << "\n"
         << "Input shape:" << input.shape << "\n";

    if (input.buffer.data)
    {
        if (input.shape.rank == 4)
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
        cout << "Decoder:" << "\n"
             << "Decoder shape:" << decoder.shape << "\n";

    cout << "Targets:" << "\n"
         << "Target shape:" << target.shape << "\n";

    if (target.buffer.data && target.shape.rank == 2)
        cout << MatrixMap(const_cast<float*>(target.buffer.as<float>()),
                            target.shape[0],
                            target.shape[1]) << "\n";
}

bool Batch::is_empty() const
{
    return input.buffer.empty() && decoder.buffer.empty() && target.buffer.empty();
}

Batch::~Batch()
{
    wait_h2d_complete();
    device::deallocate_pinned_host(input.host);
    device::deallocate_pinned_host(decoder.host);
    device::deallocate_pinned_host(target.host);
}

#ifdef OPENNN_HAS_CUDA

void Batch::copy_device_async(cudaStream_t stream)
{
    upload_to_device_batch_async(*this, stream);
}

void Batch::upload_to_device_batch_async(Batch& destination, cudaStream_t stream)
{
    const Index current_batch_size = current_sample_count;
    throw_if(!uses_cuda() || !destination.uses_cuda(),
             "Batch::upload_to_device_batch_async requires CUDA batches.");
    throw_if(current_batch_size > destination.samples_number,
             "Batch::upload_to_device_batch_async destination batch is too small.");

    destination.current_sample_count = current_batch_size;
    needs_device_copy = false;
    destination.needs_device_copy = false;

    const Index input_values_count  = current_batch_size * input.features_number;
    const Index target_values_count = current_batch_size * target.features_number;

    // GPU-resident gather path (prototype): rows are pulled directly from the
    // device-resident dataset, so there is no host staging buffer to copy.
    if (device_gather && dataset && dataset->is_device_resident())
    {
        const float* matrix = dataset->get_device_data();
        const Index matrix_cols = dataset->get_data_columns();

        // Upload this batch's row indices (small: current_batch_size ints).
        // The buffer is pre-sized at warmup; never resize inside the loop.
        device::copy_async(gather_indices_device.data, gather_row_indices.data(),
                           current_batch_size * Index(sizeof(int)),
                           device::CopyKind::HostToDevice, stream);

        const int* idx = gather_indices_device.as<int>();

        if (!destination.fp32_staging.empty())
        {
            gather_rows_cuda(matrix, idx, destination.fp32_staging.as<float>(),
                             current_batch_size, input.features_number, matrix_cols, input_col_offset, stream);
            cast_fp32_to_bf16_cuda(input_values_count,
                                   destination.fp32_staging.as<float>(),
                                   destination.input.buffer.as<bfloat16>(),
                                   stream);
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

    if (!destination.fp32_staging.empty())
    {
        assert(destination.fp32_staging.bytes >= input_values_count * Index(sizeof(float)));
        copy_to_device_async(destination.fp32_staging.as<float>(), input.host, input_values_count * sizeof(float));
        cast_fp32_to_bf16_cuda(input_values_count,
                               destination.fp32_staging.as<float>(),
                               destination.input.buffer.as<bfloat16>(),
                               stream);
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

// Device-side wait: make the compute stream wait for this batch's H2D upload
// (issued on the transfer stream) before any kernel consumes the data. Does not
// touch h2d_done_recorded, which the fill worker uses for host-buffer reuse.
void Batch::wait_h2d_on_compute_stream()
{
    if (h2d_done_recorded)
        device::stream_wait_event(Backend::get_compute_stream(), h2d_done_event);
}

}
