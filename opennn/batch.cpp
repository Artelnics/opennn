//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"
#include "language_dataset.h"

#ifdef OPENNN_HAS_CUDA
#include "kernel.cuh"
#endif

namespace opennn
{

Batch::Batch(const Index new_samples_number,
             const Dataset* new_dataset,
             const Configuration::Resolved& new_config)
{
    set(new_samples_number, new_dataset, new_config);
}

#ifndef OPENNN_HAS_CUDA
Batch::~Batch() = default;
#endif

void Batch::set(const Index new_samples_number,
                const Dataset* new_dataset,
                const Configuration::Resolved& new_config)
{
    if (!new_dataset)
        throw runtime_error("dataset is not set.");

    samples_number = new_samples_number;

    dataset = new_dataset;
    config = new_config;

#ifdef OPENNN_HAS_CUDA
    const bool on_gpu = uses_cuda();
    const bool bf16_input = on_gpu
                         && config.training_type == Type::BF16
                         && dynamic_cast<const LanguageDataset*>(dataset) == nullptr;
    const Index input_device_bytes = bf16_input ? Index(sizeof(bfloat16)) : Index(sizeof(float));
#else
    const Index input_device_bytes = Index(sizeof(float));
#endif

    auto setup_buffer = [&](const string& role,
                            Shape& shape, Buffer& buffer,
                            [[maybe_unused]] Index& num_features,
                            [[maybe_unused]] float*& host_buf,
                            [[maybe_unused]] Index& host_alloc,
                            [[maybe_unused]] Index device_elem_bytes)
    {
        const Shape& dataset_shape = dataset->get_shape(role);
        if (dataset_shape.empty()) return;

        shape = Shape({samples_number}).append(dataset_shape);

#ifdef OPENNN_HAS_CUDA
        if (on_gpu)
        {
            buffer.resize_bytes(shape.size() * device_elem_bytes, Device::CUDA);
            num_features = dataset_shape.size();

            if (const Index size = samples_number * num_features; size > host_alloc)
            {
                if (host_buf) cudaFreeHost(host_buf);
                CHECK_CUDA(cudaMallocHost(&host_buf, size * sizeof(float)));
                host_alloc = size;
            }

            return;
        }
#endif
        buffer.resize_bytes(shape.size() * Index(sizeof(float)), Device::CPU);
    };

    setup_buffer("Input",   input_shape,   input,
                 input_features_number,   inputs_host,  inputs_host_allocated_size,
                 input_device_bytes);
    setup_buffer("Target",  target_shape,  target,
                 target_features_number,  targets_host, targets_host_allocated_size,
                 Index(sizeof(float)));
    setup_buffer("Decoder", decoder_shape, decoder,
                 decoder_features_number, decoder_host, decoder_host_allocated_size,
                 Index(sizeof(float)));

    input_views_host_cache.clear();

    if (!decoder_shape.empty())
        input_views_host_cache.emplace_back(decoder.as<float>(), decoder_shape, Type::FP32, Device::CPU);

    if (!input_shape.empty())
        input_views_host_cache.emplace_back(input.as<float>(), input_shape, Type::FP32, Device::CPU);

    if (!target_shape.empty())
        target_view_host_cache = TensorView(target.as<float>(), target_shape, Type::FP32, Device::CPU);

#ifdef OPENNN_HAS_CUDA
    needs_fp32_staging = bf16_input;

    if (!input_shape.empty() && input.data)
    {
        input_views_cache.clear();

        if (!decoder_shape.empty() && decoder.data)
            input_views_cache.emplace_back(decoder.data, decoder_shape, Type::FP32, Device::CUDA);

        input_views_cache.emplace_back(input.data, input_shape, bf16_input ? Type::BF16 : Type::FP32, Device::CUDA);
    }

    if (!target_shape.empty() && target.data)
        target_view_cache = TensorView(target.data, target_shape, Type::FP32, Device::CUDA);
#endif
}

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 bool is_training,
                 bool parallelize_samples)
{
    current_sample_count = ssize(sample_indices);

    const bool on_gpu = uses_cuda();

    float* const input_buffer   = on_gpu ? inputs_host  : input.as<float>();
    float* const decoder_buffer = on_gpu ? decoder_host : decoder.as<float>();
    float* const target_buffer  = on_gpu ? targets_host : target.as<float>();

    if (input_contiguous < 0 && !input_indices.empty())
        input_contiguous = is_contiguous(input_indices) ? 1 : 0;
    if (decoder_contiguous < 0 && !decoder_indices.empty())
        decoder_contiguous = is_contiguous(decoder_indices) ? 1 : 0;
    if (target_contiguous < 0 && !target_indices.empty())
        target_contiguous = is_contiguous(target_indices) ? 1 : 0;

    const bool parallelize = parallelize_samples && !on_gpu;

    dataset->fill_inputs(sample_indices, input_indices, input_buffer,
                         is_training, parallelize, input_contiguous);

    if (!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder_buffer,
                              is_training, parallelize, decoder_contiguous);

    dataset->fill_targets(sample_indices, target_indices, target_buffer,
                          is_training, parallelize, target_contiguous);
}

Index Batch::get_samples_number() const
{
    return samples_number;
}

void Batch::print() const
{
    cout << "Batch" << "\n"
         << "Inputs:" << "\n"
         << "Input shape:" << input_shape << "\n";

    if (input_shape.rank == 4)
        cout << TensorMap4(const_cast<float*>(input.as<float>()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2],
                           input_shape[3]);
    else if (input_shape.rank == 3)
        cout << TensorMap3(const_cast<float*>(input.as<float>()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);
    else if (input_shape.rank == 2)
        cout << MatrixMap(const_cast<float*>(input.as<float>()),
                          input_shape[0],
                          input_shape[1]);

    cout << "\n";

    if (!decoder_shape.empty())
    {
        cout << "Decoder:" << "\n"
             << "Decoder shape:" << decoder_shape << "\n";
    }

    cout << "Targets:" << "\n"
         << "Target shape:" << target_shape << "\n";

    cout << MatrixMap(const_cast<float*>(target.as<float>()),
                      target_shape[0],
                      target_shape[1]) << "\n";
}

bool Batch::is_empty() const
{
    return input.empty();
}

#ifdef OPENNN_HAS_CUDA

Batch::~Batch()
{
    if (h2d_done_event) cudaEventDestroy(h2d_done_event);
    if (inputs_host)  cudaFreeHost(inputs_host);
    if (decoder_host) cudaFreeHost(decoder_host);
    if (targets_host) cudaFreeHost(targets_host);
}

void Batch::copy_device_async(const Index current_batch_size, cudaStream_t stream, float* fp32_staging)
{
    const Index input_size  = current_batch_size * input_features_number;
    const Index target_size = current_batch_size * target_features_number;

    auto copy_to_device = [&](void* destination, const void* source, Index bytes) {
        CHECK_CUDA(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
    };

    if (needs_fp32_staging)
    {
        assert(fp32_staging != nullptr);
        copy_to_device(fp32_staging, inputs_host, input_size * sizeof(float));
        cast_fp32_to_bf16_cuda(input_size, fp32_staging, input.as<bfloat16>(), stream);
    }
    else
    {
        copy_to_device(input.as<float>(), inputs_host, input_size * sizeof(float));
    }

    if (!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * decoder_features_number;
        copy_to_device(decoder.as<float>(), decoder_host, decoder_size * sizeof(float));
    }

    copy_to_device(target.as<float>(), targets_host, target_size * sizeof(float));

    if (!h2d_done_event)
        CHECK_CUDA(cudaEventCreateWithFlags(&h2d_done_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(h2d_done_event, stream));
    h2d_done_recorded = true;
}

void Batch::gather_device_async(const Index current_batch_size,
                                const Index* device_row_indices,
                                const float* resident_input, Index resident_input_features,
                                const float* resident_target, Index resident_target_features)
{
    current_sample_count = current_batch_size;

    if (needs_fp32_staging)
        gather_rows_cuda<float, bfloat16>(current_batch_size, resident_input_features,
                                          device_row_indices, resident_input, input.as<bfloat16>());
    else
        gather_rows_cuda<float, float>(current_batch_size, resident_input_features,
                                       device_row_indices, resident_input, input.as<float>());

    gather_rows_cuda<float, float>(current_batch_size, resident_target_features,
                                   device_row_indices, resident_target, target.as<float>());
}

#endif

void Batch::wait_h2d_complete()
{
#ifdef OPENNN_HAS_CUDA
    if (h2d_done_recorded)
    {
        CHECK_CUDA(cudaEventSynchronize(h2d_done_event));
        h2d_done_recorded = false;
    }
#endif
}

}
