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

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 bool augment)
{
    dataset->fill_inputs(sample_indices, input_indices, input_vector.data());

    if(augment)
        dataset->augment_inputs(input_vector.data(), sample_indices.size());

    if(!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder_vector.data());

    dataset->fill_targets(sample_indices, target_indices, target_vector.data());
}

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
        input_vector.resize(input_shape.size());
    }

    // Target

    const Shape& dataset_target_shape = dataset->get_shape("Target");

    if(!dataset_target_shape.empty())
    {
        target_shape = Shape({samples_number}).append(dataset_target_shape);
        target_vector.resize(target_shape.size());
    }

    // Decoder

    const Shape& dataset_decoder_shape = dataset->get_shape("Decoder");

    if(!dataset_decoder_shape.empty())
    {
        decoder_shape = Shape({samples_number}).append(dataset_decoder_shape);
        decoder_vector.resize(decoder_shape.size());
    }
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
        cout << TensorMap4(const_cast<type*>(input_vector.data()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2],
                           input_shape[3]);
    else if (input_shape.rank == 3)
        cout << TensorMap3(const_cast<type*>(input_vector.data()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);
    else if (input_shape.rank == 2)
        cout << MatrixMap(const_cast<type*>(input_vector.data()),
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

    cout << MatrixMap(const_cast<type*>(target_vector.data()),
                      target_shape[0],
                      target_shape[1]) << endl;
}

bool Batch::is_empty() const
{
    return input_vector.size() == 0;
}

vector<TensorView> Batch::get_inputs() const
{
    vector<TensorView> input_views = {{const_cast<type*>(input_vector.data()), input_shape}};

    if(!decoder_shape.empty())
        input_views.insert(input_views.begin(), {const_cast<type*>(decoder_vector.data()), decoder_shape});

    return input_views;
}

TensorView Batch::get_targets() const
{
    return {const_cast<type*>(target_vector.data()) , target_shape};
}

#ifdef CUDA

void BatchCuda::fill(const vector<Index>& sample_indices,
                     const vector<Index>& input_indices,
                     const vector<Index>& decoder_indices,
                     const vector<Index>& target_indices)
{
    fill_host(sample_indices, input_indices, decoder_indices, target_indices);

    const Index batch_size = sample_indices.size();

    copy_device(batch_size);
}

void BatchCuda::fill_host(const vector<Index>& sample_indices,
                          const vector<Index>& input_indices,
                          const vector<Index>& decoder_indices,
                          const vector<Index>& target_indices)
{
    dataset->fill_inputs(sample_indices, input_indices, inputs_host, false);

    if(!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder_host, false);

    dataset->fill_targets(sample_indices, target_indices, targets_host, false);
}

BatchCuda::BatchCuda(const Index new_samples_number, Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}

BatchCuda::~BatchCuda()
{
    cudaFreeHost(inputs_host);
    inputs_host = nullptr;

    cudaFreeHost(decoder_host);
    decoder_host = nullptr;

    cudaFreeHost(targets_host);
    targets_host = nullptr;
}

void BatchCuda::set(const Index new_samples_number, Dataset* new_dataset)
{
    if(!new_dataset) return;


    dataset = new_dataset;

    const Shape& dataset_input_shape = dataset->get_shape("Input");
    const Shape& dataset_decoder_shape = dataset->get_shape("Decoder");
    const Shape& dataset_target_shape = dataset->get_shape("Target");

    if(!dataset_input_shape.empty())
    {
        num_input_features = dataset->get_features_number("Input");
        const Index input_size = samples_number * num_input_features;

        input_shape = Shape({samples_number}).append(dataset_input_shape);

        if (input_size > inputs_host_allocated_size)
        {
            cudaFreeHost(inputs_host);
            CHECK_CUDA(cudaMallocHost(&inputs_host, input_size * sizeof(float)));
            inputs_host_allocated_size = input_size;
        }

        inputs_device.resize({samples_number, num_input_features});
    }
    if(!dataset_decoder_shape.empty())
    {
        const Index num_decoder_features = dataset->get_features_number("Decoder");

        decoder_shape = Shape({samples_number}).append(dataset_decoder_shape);

        const Index decoder_size = samples_number * num_decoder_features;

        if (decoder_size > decoder_host_allocated_size)
        {
            cudaFreeHost(decoder_host);
            CHECK_CUDA(cudaMallocHost(&decoder_host, decoder_size * sizeof(float)));
            decoder_host_allocated_size = decoder_size;
        }

        decoder_device.resize({samples_number, num_decoder_features});
    }
    if(!dataset_target_shape.empty())
    {
        num_target_features = dataset->get_features_number("Target");
        const Index target_size = samples_number * num_target_features;

        target_shape = Shape({samples_number}).append(dataset_target_shape);

        if (target_size > targets_host_allocated_size)
        {
            cudaFreeHost(targets_host);
            CHECK_CUDA(cudaMallocHost(&targets_host, target_size * sizeof(float)));
            targets_host_allocated_size = target_size;
        }

        targets_device.resize({samples_number, num_target_features});
    }
}

void BatchCuda::copy_device(const Index current_batch_size)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpy(inputs_device.data, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice));

    if(!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * num_decoder_features;
        CHECK_CUDA(cudaMemcpy(decoder_device.data, decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMemcpy(targets_device.data, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice));
}

void BatchCuda::copy_device_async(const Index current_batch_size, cudaStream_t stream)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpyAsync(inputs_device.data, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    if(!decoder_shape.empty())
    {
        const Index decoder_size = current_batch_size * num_decoder_features;
        CHECK_CUDA(cudaMemcpyAsync(decoder_device.data, decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(targets_device.data, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice, stream));
}

MatrixR BatchCuda::get_inputs_from_device() const
{
    const Index inputs_number = dataset->get_variables_number("Input");

    MatrixR inputs = MatrixR::Zero(samples_number, inputs_number);

    CHECK_CUDA(cudaMemcpy(inputs.data(), inputs_device.data, samples_number * inputs_number * sizeof(type), cudaMemcpyDeviceToHost));

    return inputs;
}

MatrixR BatchCuda::get_decoder_from_device() const
{
    const Index decoder_number = dataset->get_variables_number("Decoder");

    MatrixR decoder = MatrixR::Zero(samples_number, decoder_number);

    CHECK_CUDA(cudaMemcpy(decoder.data(), decoder_device.data, samples_number * decoder_number * sizeof(type), cudaMemcpyDeviceToHost));

    return decoder;
}

MatrixR BatchCuda::get_targets_from_device() const
{
    const Index targets_number = target_shape[1];

    MatrixR targets = MatrixR::Zero(samples_number, targets_number);

    CHECK_CUDA(cudaMemcpy(targets.data(), targets_device.data, samples_number * targets_number * sizeof(type), cudaMemcpyDeviceToHost));

    return targets;
}

vector<TensorView> BatchCuda::get_inputs_device() const
{
    if(!decoder_shape.empty())
        return { decoder_device.view(), inputs_device.view() };

    return { inputs_device.view() };
}

TensorView BatchCuda::get_targets_device() const
{
    return { targets_device.view() };
}

Index BatchCuda::get_samples_number() const
{
    return samples_number;
}

void BatchCuda::print() const
{
    // @todo
}

bool BatchCuda::is_empty() const
{
    return input_shape.empty();
}

#endif

}
