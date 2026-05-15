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

Batch::Batch(const Index new_samples_number, const Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}

Batch::~Batch()
{
#ifdef OPENNN_HAS_CUDA
    if (inputs_host)  cudaFreeHost(inputs_host);
    if (decoder_host) cudaFreeHost(decoder_host);
    if (targets_host) cudaFreeHost(targets_host);
#endif
}

void Batch::set(const Index new_samples_number, const Dataset* new_dataset)
{
    if (!new_dataset)
        throw runtime_error("dataset is not set.");

    samples_number = new_samples_number;

    dataset = new_dataset;

#ifdef OPENNN_HAS_CUDA
    [[maybe_unused]] const bool on_gpu = is_gpu();
    const bool bf16_input = on_gpu
                         && is_bf16_training()
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
            num_features = dataset->get_features_number(role);

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
                 num_input_features,   inputs_host,  inputs_host_allocated_size,
                 input_device_bytes);
    setup_buffer("Target",  target_shape,  target,
                 num_target_features,  targets_host, targets_host_allocated_size,
                 Index(sizeof(float)));
    setup_buffer("Decoder", decoder_shape, decoder,
                 num_decoder_features, decoder_host, decoder_host_allocated_size,
                 Index(sizeof(float)));

    input_views_host_cache.clear();
    input_views_host_cache.reserve(decoder_shape.empty() ? 1 : 2);

    if (!decoder_shape.empty())
        input_views_host_cache.emplace_back(decoder.as<float>(), decoder_shape);

    if (!input_shape.empty())
        input_views_host_cache.emplace_back(input.as<float>(), input_shape);

    if (!target_shape.empty())
        target_view_host_cache = TensorView(target.as<float>(), target_shape);

#ifdef OPENNN_HAS_CUDA
    needs_fp32_staging = bf16_input;

    if (!input_shape.empty() && input.data)
    {
        input_views_cache.clear();

        if (!decoder_shape.empty() && decoder.data)
            input_views_cache.emplace_back(decoder.data, decoder_shape, Type::FP32);
        
            input_views_cache.emplace_back(input.data, input_shape, bf16_input ? Type::BF16 : Type::FP32);
    }

    if (!target_shape.empty() && target.data)
        target_view_cache = TensorView(target.data, target_shape, Type::FP32);
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

    const bool on_gpu = is_gpu();

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

    if (is_training)
        dataset->augment_inputs(input_buffer, sample_indices.size());

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

void Batch::copy_device_async(const Index current_batch_size, cudaStream_t stream, float* fp32_staging)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    if (needs_fp32_staging)
    {
        assert(fp32_staging != nullptr);
        CHECK_CUDA(cudaMemcpyAsync(fp32_staging, inputs_host,
                                   input_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        cast_fp32_to_bf16_cuda(input_size,
                               fp32_staging,
                               input.as<bfloat16>(),
                               stream);
    }
    else
    {
        CHECK_CUDA(cudaMemcpyAsync(input.as<float>(), inputs_host,
                                   input_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }

    if (!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * num_decoder_features;
        CHECK_CUDA(cudaMemcpyAsync(decoder.as<float>(), 
                                   decoder_host, 
                                   decoder_size * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(target.as<float>(), 
                               targets_host, 
                               target_size * sizeof(float), 
                               cudaMemcpyHostToDevice, 
                               stream));
}

#endif

}
