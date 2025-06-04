//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"
#include "tensors.h"
#include "image_dataset.h"
#include "language_dataset.h"
#include "images.h"
#include "time_series_dataset.h"

namespace opennn
{

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices)
{
    const Tensor<type, 2>& data = dataset->get_data();

    const Index batch_size = sample_indices.size();
    const Index sequence_length = sample_indices.size() / batch_size;
    const Index input_size = data.dimension(1);

    // fill inputs
    if(is_instance_of<ImageDataset>(dataset))
    {
        ImageDataset* image_Dataset = dynamic_cast<ImageDataset*>(dataset);

        if (image_Dataset->get_augmentation())
        {
            // @todo

            //Tensor<type, 2> augmented_data = perform_augmentation(data);

            //fill_tensor_data(augmented_data, sample_indices, input_indices, input_data);
        }
        else
            fill_tensor_data(data, sample_indices, input_indices, input_tensor.data());
    }
    else if(is_instance_of<TimeSeriesDataset>(dataset)){
        //fill_tensor_data(data, sample_indices, input_indices, input_tensor.data());
        fill_tensor_3D(data, sample_indices, input_indices, input_tensor.data());
        input_dimensions = { batch_size, sequence_length, input_size };
    }
    else
        fill_tensor_data(data, sample_indices, input_indices, input_tensor.data());

    // fill targets
    if(is_instance_of<TimeSeriesDataset>(dataset))
        fill_tensor_3D(data, sample_indices, target_indices, target_tensor.data());
    else if (is_instance_of<LanguageDataset>(dataset))
        fill_tensor_data(data, sample_indices, decoder_indices, decoder_tensor.data());
    else
        fill_tensor_data(data, sample_indices, target_indices, target_tensor.data());

}


Tensor<type, 2> Batch::perform_augmentation(const Tensor<type, 2>& data)
{
    ImageDataset* image_Dataset = static_cast<ImageDataset*>(dataset);

    const dimensions input_dimensions = dataset->get_dimensions(Dataset::VariableUse::Input);

    const Index input_height = input_dimensions[0];
    const Index input_width = input_dimensions[1];
    const Index channels = input_dimensions[2];

    TensorMap<Tensor<type, 4>> inputs(input_tensor.data(),
                                      samples_number,
                                      input_height,
                                      input_width,
                                      channels);
   
    const bool random_reflection_axis_x = image_Dataset->get_random_reflection_axis_x();
    const bool random_reflection_axis_y = image_Dataset->get_random_reflection_axis_y();
    const type random_rotation_minimum = image_Dataset->get_random_rotation_minimum();
    const type random_rotation_maximum = image_Dataset->get_random_rotation_maximum();
    const type random_horizontal_translation_minimum = image_Dataset->get_random_horizontal_translation_minimum();
    const type random_horizontal_translation_maximum = image_Dataset->get_random_horizontal_translation_maximum();
    const type random_vertical_translation_minimum = image_Dataset->get_random_vertical_translation_minimum();
    const type random_vertical_translation_maximum = image_Dataset->get_random_vertical_translation_maximum();

    for(Index batch_index = 0; batch_index < samples_number; batch_index++)
    {
        Tensor<type, 3> image = inputs.chip(batch_index, 0);

        if(random_reflection_axis_x)
            reflect_image_x(thread_pool_device.get(), image);

        if(random_reflection_axis_y)
            reflect_image_y(thread_pool_device.get(), image);

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
        {
            const type angle = random_rotation_minimum < random_rotation_maximum
                             ? random_rotation_minimum + type(rand())
                             : random_rotation_maximum;

            rotate_image(thread_pool_device.get(), image, image, angle);
        }

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
        {
            const type translation = random_horizontal_translation_minimum < random_horizontal_translation_maximum
                                   ? random_horizontal_translation_minimum + type(rand())
                                   : random_horizontal_translation_maximum;

            translate_image_x(thread_pool_device.get(), image, image, translation);
        }

        if(random_vertical_translation_minimum != 0 && random_vertical_translation_maximum != 0)
        {
            const type translation = random_vertical_translation_minimum < random_vertical_translation_maximum
                                         ? random_vertical_translation_minimum + type(rand())
                                         : random_vertical_translation_maximum;

            translate_image_y(thread_pool_device.get(), image, image, translation);
        }
    } 

    return Tensor<type, 2>();
}


Batch::Batch(const Index& new_samples_number, Dataset* new_data_set)
{
    if(thread_pool != nullptr)
        shutdown_threads();

    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    set(new_samples_number, new_data_set);
}


void Batch::shutdown_threads()
{
    thread_pool_device.reset();

    if(thread_pool)
        thread_pool.release();

    thread_pool.reset();
}


void Batch::set(const Index& new_samples_number, Dataset* new_data_set)
{
    if (!new_data_set) return;

    samples_number = new_samples_number;
    dataset = new_data_set;

    const dimensions& data_set_input_dimensions = dataset->get_dimensions(Dataset::VariableUse::Input);
    const dimensions& data_set_target_dimensions = dataset->get_dimensions(Dataset::VariableUse::Target);

    if (!data_set_input_dimensions.empty())
    {

        input_dimensions = {samples_number};
        input_dimensions.insert(input_dimensions.end(), data_set_input_dimensions.begin(), data_set_input_dimensions.end());

        const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
        input_tensor.resize(input_size);
    }

    // @todo
    // const dimensions& data_set_decoder_dimensions = data_set->get_dimensions(Dataset::VariableUse::Decoder);

    // if (!data_set_decoder_dimensions.empty())
    // {
    //     decoder_dimensions = { samples_number };
    //     decoder_dimensions.insert(decoder_dimensions.end(), data_set_decoder_dimensions.begin(), data_set_decoder_dimensions.end());

    //     const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());
    //     decoder_tensor.resize(decoder_size);
    // }

    if (!data_set_target_dimensions.empty())
    {

        target_dimensions = {samples_number};
        target_dimensions.insert(target_dimensions.end(), data_set_target_dimensions.begin(), data_set_target_dimensions.end());

        const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());
        target_tensor.resize(target_size);
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
         << "Input dimensions:" << endl;

    print_vector(input_dimensions);
    
    if(input_dimensions.size() == 4)
    {
        const TensorMap<Tensor<type, 4>> inputs((type*)input_tensor.data(),
                                                input_dimensions[0],
                                                input_dimensions[1],
                                                input_dimensions[2],
                                                input_dimensions[3]);

        cout << inputs << endl;
    }
    else
    {
        const TensorMap<Tensor<type, 2>> inputs((type*)input_tensor.data(),
                                                input_dimensions[0],
                                                input_dimensions[1]);

        cout << inputs << endl;
    }
    
    cout << "Decoder:" << endl
         << "Decoder dimensions:" << endl;

    print_vector(decoder_dimensions);

    cout << "Targets:" << endl
         << "Target dimensions:" << endl;

    print_vector(target_dimensions);

    const TensorMap<Tensor<type, 2>> targets((type*)target_tensor.data(),
                                             target_dimensions[0],
                                             target_dimensions[1]);

    cout << targets << endl;
}


bool Batch::is_empty() const
{
    return input_tensor.size() == 0;
}


vector<pair<type*, dimensions>> Batch::get_input_pairs() const
{
    vector<pair<type*, dimensions>> input_pairs = {{(type*)input_tensor.data(), input_dimensions}};

    if (!decoder_dimensions.empty())
        input_pairs.insert(input_pairs.begin(), {(type*)decoder_tensor.data(), decoder_dimensions});

    return input_pairs;
}


pair<type*, dimensions> Batch::get_target_pair() const
{
    return { (type*)target_tensor.data() , target_dimensions};
}


#ifdef OPENNN_CUDA

void BatchCuda::fill(const vector<Index>& sample_indices,
                     const vector<Index>& input_indices,
                     const vector<Index>& decoder_indices,
                     const vector<Index>& target_indices)
{
    const Tensor<type, 2>& data = Dataset->get_data();

    if (is_instance_of<ImageDataset>(Dataset))
    {
        ImageDataset* image_Dataset = dynamic_cast<ImageDataset*>(Dataset);

        if (image_Dataset->get_augmentation())
        {
            // @todo

            //Tensor<type, 2> augmented_data = perform_augmentation(data);

            //fill_tensor_data_row_major(data, sample_indices, input_indices, inputs_host);
        }
        else
            fill_tensor_data_row_major(data, sample_indices, input_indices, inputs_host);
    }
    else
        fill_tensor_data(data, sample_indices, input_indices, inputs_host);

    if (is_instance_of<LanguageDataset>(Dataset))
        fill_tensor_data(data, sample_indices, decoder_indices, decoder_host);

    fill_tensor_data(data, sample_indices, target_indices, targets_host);

    copy_device();
}


BatchCuda::BatchCuda(const Index& new_samples_number, Dataset* new_data_set)
{
    set(new_samples_number, new_data_set);
}


void BatchCuda::set(const Index& new_samples_number, Dataset* new_data_set)
{
    if (!new_data_set) return;

    samples_number = new_samples_number;
    Dataset = new_data_set;

    const dimensions& data_set_input_dimensions = Dataset->get_dimensions(Dataset::VariableUse::Input);
    const dimensions& data_set_decoder_dimensions = Dataset->get_dimensions(Dataset::VariableUse::Decoder);
    const dimensions& data_set_target_dimensions = Dataset->get_dimensions(Dataset::VariableUse::Target);

    if (!data_set_input_dimensions.empty())
    {
        input_dimensions = { samples_number };
        input_dimensions.insert(input_dimensions.end(), data_set_input_dimensions.begin(), data_set_input_dimensions.end());

        const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());

        CHECK_CUDA(cudaMallocHost(&inputs_host, input_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&inputs_device, input_size * sizeof(float)));
    }

    if (!data_set_decoder_dimensions.empty())
    {
        decoder_dimensions = { samples_number };
        decoder_dimensions.insert(decoder_dimensions.end(), data_set_decoder_dimensions.begin(), data_set_decoder_dimensions.end());

        const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());

        CHECK_CUDA(cudaMallocHost(&decoder_host, decoder_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&decoder_device, decoder_size * sizeof(float)));
    }

    if (!data_set_target_dimensions.empty())
    {
        target_dimensions = { samples_number };
        target_dimensions.insert(target_dimensions.end(), data_set_target_dimensions.begin(), data_set_target_dimensions.end());

        const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());

        CHECK_CUDA(cudaMallocHost(&targets_host, target_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&targets_device, target_size * sizeof(float)));
    }
}


void BatchCuda::copy_device()
{
    const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
    const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());
    const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());

    CHECK_CUDA(cudaMemcpy(inputs_device, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice));

    if (!decoder_dimensions.empty())
        CHECK_CUDA(cudaMemcpy(decoder_device, decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(targets_device, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice));
}


Tensor<type, 2> BatchCuda::get_inputs_device() const
{
    const Index inputs_number = Dataset->get_raw_variables_number(Dataset::VariableUse::Input);

    Tensor<type, 2> inputs(samples_number, inputs_number);

    inputs.setZero();

    CHECK_CUDA(cudaMemcpy(inputs.data(), inputs_device, samples_number * inputs_number * sizeof(type), cudaMemcpyDeviceToHost));

    return inputs;
}


Tensor<type, 2> BatchCuda::get_decoder_device() const
{
    const Index decoder_number = Dataset->get_raw_variables_number(Dataset::VariableUse::Decoder);

    Tensor<type, 2> decoder(samples_number, decoder_number);

    decoder.setZero();

    CHECK_CUDA(cudaMemcpy(decoder.data(), inputs_device, samples_number * decoder_number * sizeof(type), cudaMemcpyDeviceToHost));

    return decoder;
}


Tensor<type, 2> BatchCuda::get_targets_device() const
{
    const Index targets_number = target_dimensions[1];

    Tensor<type, 2> targets(samples_number, targets_number);

    targets.setZero();

    CHECK_CUDA(cudaMemcpy(targets.data(), targets_device, samples_number * targets_number * sizeof(type), cudaMemcpyDeviceToHost));

    return targets;
}


vector<float*> BatchCuda::get_input_device() const
{
    vector<float*> inputs = { inputs_device };

    if (!decoder_dimensions.empty())
        inputs.insert(inputs.begin(), decoder_device );

    return inputs;
}


pair<type*, dimensions> BatchCuda::get_target_pair_device() const
{
    pair<type*, dimensions> target_pair = {targets_device , target_dimensions};

    return target_pair;
}


Index BatchCuda::get_samples_number() const
{
    return samples_number;
}


void BatchCuda::print() const
{
    if (!input_dimensions.empty())
        cout << "get_inputs_device:" << endl << get_inputs_device() << endl;

    if (!decoder_dimensions.empty())
        cout << "get_decoder_device:" << endl << get_decoder_device() << endl;

    if (!target_dimensions.empty())
        cout << "get_targets_device:" << endl << get_targets_device() << endl;
}


bool BatchCuda::is_empty() const
{
    return input_dimensions.empty();
}


void BatchCuda::free()
{
    cudaFreeHost(inputs_host);
    cudaFreeHost(decoder_host);
    cudaFreeHost(targets_host);
    cudaFree(inputs_device);
    cudaFree(decoder_device);
    cudaFree(targets_device);

    inputs_device = nullptr;
    decoder_device = nullptr;
    targets_device = nullptr;
    inputs_host = nullptr;
    decoder_host = nullptr;
    targets_host = nullptr;
}

#endif


} // namespace opennn

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
