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
    if(!new_dataset) return;

    samples_number = new_samples_number;

    dataset = new_dataset;

    // Input

    const Shape& dataset_input_shape = dataset->get_shape("Input");

    if(!dataset_input_shape.empty())
    {
        input_shape = Shape({samples_number}).append(dataset_input_shape);
        input.resize(input_shape.size());

#ifdef CUDA
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
        target.resize(target_shape.size());

#ifdef CUDA
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
        decoder.resize(decoder_shape.size());

#ifdef CUDA
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
}

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 bool augment)
{
    dataset->fill_inputs(sample_indices, input_indices, input.data());

    if(augment)
        dataset->augment_inputs(input.data(), sample_indices.size());

    if(!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder.data());

    dataset->fill_targets(sample_indices, target_indices, target.data());
}

Index Batch::get_samples_number() const
{
    return samples_number;
}

void Batch::print() const
{
    cout << "Batch" << endl
         << "Inputs:" << endl
         << "Input shape:" << input_shape << endl;

    if (input_shape.rank == 4)
        cout << TensorMap4(const_cast<type*>(input.data()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2],
                           input_shape[3]);
    else if (input_shape.rank == 3)
        cout << TensorMap3(const_cast<type*>(input.data()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);
    else if (input_shape.rank == 2)
        cout << MatrixMap(const_cast<type*>(input.data()),
                          input_shape[0],
                          input_shape[1]);

    cout << endl;

    if(!decoder_shape.empty())
    {
        cout << "Decoder:" << endl
             << "Decoder shape:" << decoder_shape << endl;
    }

    cout << "Targets:" << endl
         << "Target shape:" << target_shape << endl;

    cout << MatrixMap(const_cast<type*>(target.data()),
                      target_shape[0],
                      target_shape[1]) << endl;
}

bool Batch::is_empty() const
{
    return input.size() == 0;
}

vector<TensorView> Batch::get_inputs() const
{
    vector<TensorView> input_views;
    input_views.reserve(decoder_shape.empty() ? 1 : 2);

    if(!decoder_shape.empty())
        input_views.push_back({const_cast<type*>(decoder.data()), decoder_shape});

    input_views.push_back({const_cast<type*>(input.data()), input_shape});

    return input_views;
}

TensorView Batch::get_targets() const
{
    return {const_cast<type*>(target.data()), target_shape};
}

#ifdef CUDA

void Batch::fill_host(const vector<Index>& sample_indices,
                      const vector<Index>& input_indices,
                      const vector<Index>& decoder_indices,
                      const vector<Index>& target_indices)
{
    dataset->fill_inputs(sample_indices, input_indices, inputs_host, false);

    if(!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder_host, false);

    dataset->fill_targets(sample_indices, target_indices, targets_host, false);
}

void Batch::copy_device(const Index current_batch_size)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpy(input.data(), inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice));

    if(!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * num_decoder_features;
        CHECK_CUDA(cudaMemcpy(decoder.data(), decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMemcpy(target.data(), targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice));
}

void Batch::copy_device_async(const Index current_batch_size, cudaStream_t stream)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpyAsync(input.data(), inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    if(!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * num_decoder_features;
        CHECK_CUDA(cudaMemcpyAsync(decoder.data(), decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(target.data(), targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice, stream));
}

vector<TensorView> Batch::get_inputs_device() const
{
    if(!decoder_shape.empty())
        return { {const_cast<type*>(decoder.data()), decoder_shape},
                 {const_cast<type*>(input.data()), input_shape} };

    return { {const_cast<type*>(input.data()), input_shape} };
}

TensorView Batch::get_targets_device() const
{
    return {const_cast<type*>(target.data()), target_shape};
}

#endif

}
