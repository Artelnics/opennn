//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"

namespace opennn
{

Batch::Batch(const Index new_samples_number, const Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}

void Batch::set(const Index new_samples_number, const Dataset* new_dataset)
{
    if(!new_dataset)
        throw runtime_error("Batch error: dataset is not set.");

    samples_number = new_samples_number;

    dataset = new_dataset;

    // Input

    const Shape& dataset_input_shape = dataset->get_shape("Input");

    if(!dataset_input_shape.empty())
    {
        input_shape = Shape({samples_number}).append(dataset_input_shape);
        input.resize_bytes(input_shape.size() * Index(sizeof(type)), DeviceType::Cpu);

#ifdef OPENNN_WITH_CUDA
        input.resize_bytes(input_shape.size() * Index(sizeof(float)), DeviceType::Gpu);

        num_input_features = dataset->get_features_number("Input");
        const Index input_size = samples_number * num_input_features;

        if(input_size > inputs_host_allocated_size)
        {
            if(inputs_host) cudaFreeHost(inputs_host);
            CHECK_CUDA(cudaMallocHost(&inputs_host, input_size * sizeof(float)));
            inputs_host_allocated_size = input_size;
        }
#endif
    }

    // Target

    const Shape& dataset_target_shape = dataset->get_shape("Target");

    if(!dataset_target_shape.empty())
    {
        target_shape = Shape({samples_number}).append(dataset_target_shape);
        target.resize_bytes(target_shape.size() * Index(sizeof(type)), DeviceType::Cpu);

#ifdef OPENNN_WITH_CUDA
        target.resize_bytes(target_shape.size() * Index(sizeof(float)), DeviceType::Gpu);

        num_target_features = dataset->get_features_number("Target");
        const Index target_size = samples_number * num_target_features;

        if(target_size > targets_host_allocated_size)
        {
            if(targets_host) cudaFreeHost(targets_host);
            CHECK_CUDA(cudaMallocHost(&targets_host, target_size * sizeof(float)));
            targets_host_allocated_size = target_size;
        }
#endif
    }

    // Decoder

    const Shape& dataset_decoder_shape = dataset->get_shape("Decoder");

    if(!dataset_decoder_shape.empty())
    {
        decoder_shape = Shape({samples_number}).append(dataset_decoder_shape);
        decoder.resize_bytes(decoder_shape.size() * Index(sizeof(type)), DeviceType::Cpu);

#ifdef OPENNN_WITH_CUDA
        decoder.resize_bytes(decoder_shape.size() * Index(sizeof(float)), DeviceType::Gpu);

        num_decoder_features = dataset->get_features_number("Decoder");
        const Index decoder_size = samples_number * num_decoder_features;

        if(decoder_size > decoder_host_allocated_size)
        {
            if(decoder_host) cudaFreeHost(decoder_host);
            CHECK_CUDA(cudaMallocHost(&decoder_host, decoder_size * sizeof(float)));
            decoder_host_allocated_size = decoder_size;
        }
#endif
    }

    // Host view caches (populated once per batch size change — zero allocations per forward pass)
    input_views_host_cache.clear();
    input_views_host_cache.reserve(decoder_shape.empty() ? 1 : 2);

    if(!decoder_shape.empty())
        input_views_host_cache.push_back(TensorView(const_cast<type*>(decoder.as<type>()), decoder_shape));

    if(!input_shape.empty())
        input_views_host_cache.push_back(TensorView(const_cast<type*>(input.as<type>()), input_shape));

    if(!target_shape.empty())
        target_view_host_cache = TensorView(const_cast<type*>(target.as<type>()), target_shape);

#ifdef OPENNN_WITH_CUDA
    if(!input_shape.empty() && input.as<type>())
    {
        TensorView in_view(input.as<type>(), input_shape, CUDNN_DATA_FLOAT);

        if(!decoder_shape.empty() && decoder.as<type>())
        {
            TensorView dec_view(decoder.as<type>(), decoder_shape, CUDNN_DATA_FLOAT);
            input_views_cache = { dec_view, in_view };
        }
        else
        {
            input_views_cache = { in_view };
        }
    }

    if(!target_shape.empty() && target.as<type>())
        target_view_cache = TensorView(target.as<type>(), target_shape, CUDNN_DATA_FLOAT);
#endif
}

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 bool augment)
{
    const bool is_gpu = Device::instance().is_gpu();

    // GPU path writes into pinned host buffers (pre-allocated in set()) so that
    // copy_device_async can DMA them straight to device memory. CPU path writes
    // into Buffer's storage.
    type* input_dst   = is_gpu ? inputs_host   : input.as<type>();
    type* decoder_dst = is_gpu ? decoder_host  : decoder.as<type>();
    type* target_dst  = is_gpu ? targets_host  : target.as<type>();

    // Serial fill on the GPU worker thread (leaves CPU cores for the GPU driver);
    // OMP parallel fill on CPU-only runs.
    const bool parallelize = !is_gpu;

    if(input_contiguous < 0 && !input_indices.empty())
        input_contiguous = is_contiguous(input_indices) ? 1 : 0;
    if(decoder_contiguous < 0 && !decoder_indices.empty())
        decoder_contiguous = is_contiguous(decoder_indices) ? 1 : 0;
    if(target_contiguous < 0 && !target_indices.empty())
        target_contiguous = is_contiguous(target_indices) ? 1 : 0;

    dataset->fill_inputs(sample_indices, input_indices, input_dst, parallelize, input_contiguous);

    if(augment)
        dataset->augment_inputs(input_dst, sample_indices.size());

    if(!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder_dst, parallelize, decoder_contiguous);

    dataset->fill_targets(sample_indices, target_indices, target_dst, parallelize, target_contiguous);
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
        cout << TensorMap4(const_cast<type*>(input.as<type>()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2],
                           input_shape[3]);
    else if (input_shape.rank == 3)
        cout << TensorMap3(const_cast<type*>(input.as<type>()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);
    else if (input_shape.rank == 2)
        cout << MatrixMap(const_cast<type*>(input.as<type>()),
                          input_shape[0],
                          input_shape[1]);

    cout << "\n";

    if(!decoder_shape.empty())
    {
        cout << "Decoder:" << "\n"
             << "Decoder shape:" << decoder_shape << "\n";
    }

    cout << "Targets:" << "\n"
         << "Target shape:" << target_shape << "\n";

    cout << MatrixMap(const_cast<type*>(target.as<type>()),
                      target_shape[0],
                      target_shape[1]) << "\n";
}

bool Batch::is_empty() const
{
    return input.empty();
}

#ifdef OPENNN_WITH_CUDA

void Batch::copy_device_async(const Index current_batch_size, cudaStream_t stream)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpyAsync(input.as<type>(), inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    if(!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * num_decoder_features;
        CHECK_CUDA(cudaMemcpyAsync(decoder.as<type>(), decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(target.as<type>(), targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice, stream));
}

const vector<TensorView>& Batch::get_inputs_device() const
{
    return input_views_cache;
}

const TensorView& Batch::get_targets_device() const
{
    return target_view_cache;
}

#endif

}
