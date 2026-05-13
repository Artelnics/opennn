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
    if (targets_host) cudaFreeHost(targets_host);
    if (decoder_host) cudaFreeHost(decoder_host);
#endif
}

void Batch::set(const Index new_samples_number, const Dataset* new_dataset)
{
    if (!new_dataset)
        throw runtime_error("dataset is not set.");

    const bool on_gpu = is_gpu();

    samples_number = new_samples_number;

    dataset = new_dataset;

    const bool bf16_input = on_gpu
                         && is_bf16_training()
                         && dynamic_cast<const LanguageDataset*>(dataset) == nullptr;

    // Input

    const Shape& dataset_input_shape = dataset->get_shape("Input");

    if (!dataset_input_shape.empty())
    {
        input_shape = Shape({samples_number}).append(dataset_input_shape);

#ifdef OPENNN_HAS_CUDA
        if (on_gpu)
        {
            const Index elem_bytes = bf16_input ? Index(sizeof(__nv_bfloat16)) : Index(sizeof(float));
            input.resize_bytes(input_shape.size() * elem_bytes, Device::CUDA);

            num_input_features = dataset->get_features_number("Input");
            const Index input_size = samples_number * num_input_features;

            if (input_size > inputs_host_allocated_size)
            {
                if (inputs_host) cudaFreeHost(inputs_host);
                CHECK_CUDA(cudaMallocHost(&inputs_host, input_size * sizeof(float)));
                inputs_host_allocated_size = input_size;
            }

            needs_fp32_staging = bf16_input;
        }
        else
#endif
        {
            input.resize_bytes(input_shape.size() * Index(sizeof(float)), Device::CPU);
        }
    }

    // Target

    const Shape& dataset_target_shape = dataset->get_shape("Target");

    if (!dataset_target_shape.empty())
    {
        target_shape = Shape({samples_number}).append(dataset_target_shape);

#ifdef OPENNN_HAS_CUDA
        if (on_gpu)
        {
            target.resize_bytes(target_shape.size() * Index(sizeof(float)), Device::CUDA);

            num_target_features = dataset->get_features_number("Target");
            const Index target_size = samples_number * num_target_features;

            if (target_size > targets_host_allocated_size)
            {
                if (targets_host) cudaFreeHost(targets_host);
                CHECK_CUDA(cudaMallocHost(&targets_host, target_size * sizeof(float)));
                targets_host_allocated_size = target_size;
            }
        }
        else
#endif
        {
            target.resize_bytes(target_shape.size() * Index(sizeof(float)), Device::CPU);
        }
    }

    // Decoder

    const Shape& dataset_decoder_shape = dataset->get_shape("Decoder");

    if (!dataset_decoder_shape.empty())
    {
        decoder_shape = Shape({samples_number}).append(dataset_decoder_shape);

#ifdef OPENNN_HAS_CUDA
        if (on_gpu)
        {
            decoder.resize_bytes(decoder_shape.size() * Index(sizeof(float)), Device::CUDA);

            num_decoder_features = dataset->get_features_number("Decoder");
            const Index decoder_size = samples_number * num_decoder_features;

            if (decoder_size > decoder_host_allocated_size)
            {
                if (decoder_host) cudaFreeHost(decoder_host);
                CHECK_CUDA(cudaMallocHost(&decoder_host, decoder_size * sizeof(float)));
                decoder_host_allocated_size = decoder_size;
            }
        }
        else
#endif
        {
            decoder.resize_bytes(decoder_shape.size() * Index(sizeof(float)), Device::CPU);
        }
    }

    input_views_host_cache.clear();
    input_views_host_cache.reserve(decoder_shape.empty() ? 1 : 2);

    if (!decoder_shape.empty())
        input_views_host_cache.emplace_back(const_cast<float*>(decoder.as<float>()), decoder_shape);

    if (!input_shape.empty())
        input_views_host_cache.emplace_back(const_cast<float*>(input.as<float>()), input_shape);

    if (!target_shape.empty())
        target_view_host_cache = TensorView(const_cast<float*>(target.as<float>()), target_shape);

#ifdef OPENNN_HAS_CUDA
    if (!input_shape.empty() && input.data)
    {
        const Type input_dtype = bf16_input ? Type::BF16 : Type::FP32;
        TensorView in_view(input.data, input_shape, input_dtype);

        if (!decoder_shape.empty() && decoder.data)
        {
            TensorView dec_view(decoder.data, decoder_shape, Type::FP32);
            input_views_cache = { dec_view, in_view };
        }
        else
        {
            input_views_cache = { in_view };
        }
    }

    if (!target_shape.empty() && target.data)
        target_view_cache = TensorView(target.data, target_shape, Type::FP32);
#endif
}

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 bool augment)
{
    const bool on_gpu = is_gpu();

    float* input_buffer   = on_gpu ? inputs_host   : input.as<float>();
    float* decoder_buffer = on_gpu ? decoder_host  : decoder.as<float>();
    float* target_buffer  = on_gpu ? targets_host  : target.as<float>();

    const bool parallelize = !on_gpu;

    if (input_contiguous < 0 && !input_indices.empty())
        input_contiguous = is_contiguous(input_indices) ? 1 : 0;
    if (decoder_contiguous < 0 && !decoder_indices.empty())
        decoder_contiguous = is_contiguous(decoder_indices) ? 1 : 0;
    if (target_contiguous < 0 && !target_indices.empty())
        target_contiguous = is_contiguous(target_indices) ? 1 : 0;

    dataset->fill_inputs(sample_indices, input_indices, input_buffer, parallelize, input_contiguous);

    if (augment)
        dataset->augment_inputs(input_buffer, sample_indices.size());

    if (!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder_buffer, parallelize, decoder_contiguous);

    dataset->fill_targets(sample_indices, target_indices, target_buffer, parallelize, target_contiguous);
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
                               input.as<__nv_bfloat16>(),
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
