//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"
#include "tensors.h"
#include "image_data_set.h"
#include "language_data_set.h"
#include "images.h"
#include "time_series_data_set.h"

namespace opennn
{

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices)
{
    const Tensor<type, 2>& data = data_set->get_data();

    const Index rows_number = data.dimension(0);
    const Index columns_number = data.dimension(1);

    const Index batch_size = static_cast<Index>(ceil(rows_number * 0.1));
    const Index sequence_length = rows_number / batch_size;
    const Index input_size = columns_number;

    // input_tensor.resize(batch_size * sequence_length * input_size);
    // target_tensor.resize(batch_size * sequence_length * target_indices.size());

    if (!decoder_indices.empty())
        decoder_tensor.resize(batch_size * sequence_length * decoder_indices.size());

    if(is_instance_of<ImageDataSet>(data_set))
    {
        ImageDataSet* image_data_set = dynamic_cast<ImageDataSet*>(data_set);

        if (image_data_set->get_augmentation())
        {
            // @todo

            //Tensor<type, 2> augmented_data = perform_augmentation(data);

            //fill_tensor_data(augmented_data, sample_indices, input_indices, input_data);
        }
        else
        {
            fill_tensor_data(data, sample_indices, input_indices, input_tensor.data());
        }
    }
    else if(is_instance_of<TimeSeriesDataSet>(data_set))
        fill_tensor_3D(data, sample_indices, input_indices, input_tensor.data());
    else
        fill_tensor_data(data, sample_indices, input_indices, input_tensor.data());

    if (is_instance_of<LanguageDataSet>(data_set))
            fill_tensor_data(data, sample_indices, decoder_indices, decoder_tensor.data());

    fill_tensor_data(data, sample_indices, target_indices, target_tensor.data());
    
}


Tensor<type, 2> Batch::perform_augmentation(const Tensor<type, 2>& data)
{
    ImageDataSet* image_data_set = static_cast<ImageDataSet*>(data_set);

    const dimensions input_dimensions = data_set->get_dimensions(DataSet::VariableUse::Input);

    const Index input_height = input_dimensions[0];
    const Index input_width = input_dimensions[1];
    const Index channels = input_dimensions[2];

    TensorMap<Tensor<type, 4>> inputs(input_tensor.data(),
                                      samples_number,
                                      input_height,
                                      input_width,
                                      channels);
   
    const bool random_reflection_axis_x = image_data_set->get_random_reflection_axis_x();
    const bool random_reflection_axis_y = image_data_set->get_random_reflection_axis_y();
    const type random_rotation_minimum = image_data_set->get_random_rotation_minimum();
    const type random_rotation_maximum = image_data_set->get_random_rotation_maximum();
    const type random_horizontal_translation_minimum = image_data_set->get_random_horizontal_translation_minimum();
    const type random_horizontal_translation_maximum = image_data_set->get_random_horizontal_translation_maximum();
    const type random_vertical_translation_minimum = image_data_set->get_random_vertical_translation_minimum();
    const type random_vertical_translation_maximum = image_data_set->get_random_vertical_translation_maximum();

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


Batch::Batch(const Index& new_samples_number, DataSet* new_data_set)
{
    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    set(new_samples_number, new_data_set);
}


void Batch::set(const Index& new_samples_number, DataSet* new_data_set)
{
    if (!new_data_set) return;

    samples_number = new_samples_number;
    data_set = new_data_set;

    const dimensions& data_set_input_dimensions = data_set->get_dimensions(DataSet::VariableUse::Input);
    const dimensions& data_set_decoder_dimensions = data_set->get_dimensions(DataSet::VariableUse::Decoder);
    const dimensions& data_set_target_dimensions = data_set->get_dimensions(DataSet::VariableUse::Target);

    if (!data_set_input_dimensions.empty())
    {
        input_dimensions = { samples_number};
        input_dimensions.insert(input_dimensions.end(), data_set_input_dimensions.begin(), data_set_input_dimensions.end());

        const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
        input_tensor.resize(input_size);
    }

    if (!data_set_decoder_dimensions.empty())
    {
        decoder_dimensions = { samples_number };
        decoder_dimensions.insert(decoder_dimensions.end(), data_set_decoder_dimensions.begin(), data_set_decoder_dimensions.end());

        const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());
        decoder_tensor.resize(decoder_size);
    }

    if (!data_set_target_dimensions.empty())
    {
        target_dimensions = { samples_number};
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
    const Tensor<type, 2>& data = data_set->get_data();

    if (is_instance_of<ImageDataSet>(data_set))
    {
        ImageDataSet* image_data_set = dynamic_cast<ImageDataSet*>(data_set);

        if (image_data_set->get_augmentation())
        {
            // @todo

            //Tensor<type, 2> augmented_data = perform_augmentation(data);

            //fill_tensor_data_row_major(data, sample_indices, input_indices, inputs_host);
        }
        else
        {
            fill_tensor_data_row_major(data, sample_indices, input_indices, inputs_host);

            const Index height = image_data_set->get_image_height();
            const Index width = image_data_set->get_image_width();
            const Index channels = image_data_set->get_channels_number();

            //fill_tensor_data_row_major_corrected(data,sample_indices,input_indices,height, width, channels,inputs_host);
        }
    }
    else
    {
        fill_tensor_data(data, sample_indices, input_indices, inputs_host);
    }

    if (is_instance_of<LanguageDataSet>(data_set))
        fill_tensor_data(data, sample_indices, decoder_indices, decoder_host);

    fill_tensor_data(data, sample_indices, target_indices, targets_host);

    copy_device();
}


BatchCuda::BatchCuda(const Index& new_samples_number, DataSet* new_data_set)
{
    set(new_samples_number, new_data_set);
}


void BatchCuda::set(const Index& new_samples_number, DataSet* new_data_set)
{
    if (!new_data_set) return;

    samples_number = new_samples_number;
    data_set = new_data_set;

    const dimensions& data_set_input_dimensions = data_set->get_dimensions(DataSet::VariableUse::Input);
    const dimensions& data_set_decoder_dimensions = data_set->get_dimensions(DataSet::VariableUse::Decoder);
    const dimensions& data_set_target_dimensions = data_set->get_dimensions(DataSet::VariableUse::Target);

    if (!data_set_input_dimensions.empty())
    {
        input_dimensions = { samples_number };
        input_dimensions.insert(input_dimensions.end(), data_set_input_dimensions.begin(), data_set_input_dimensions.end());

        const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());

        if (cudaMallocHost(&inputs_host, input_size * sizeof(float)) != cudaSuccess)
            cout << "Inputs host allocation error" << endl;

        if (cudaMalloc(&inputs_device, input_size * sizeof(float)) != cudaSuccess)
            cout << "Inputs allocation error" << endl;
    }

    if (!data_set_decoder_dimensions.empty())
    {
        decoder_dimensions = { samples_number };
        decoder_dimensions.insert(decoder_dimensions.end(), data_set_decoder_dimensions.begin(), data_set_decoder_dimensions.end());

        const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());

        if (cudaMallocHost(&decoder_host, decoder_size * sizeof(float)) != cudaSuccess)
            cout << "Decoder host allocation error" << endl;

        if (cudaMalloc(&decoder_device, decoder_size * sizeof(float)) != cudaSuccess)
            cout << "Decoder allocation error" << endl;
    }

    if (!data_set_target_dimensions.empty())
    {
        target_dimensions = { samples_number };
        target_dimensions.insert(target_dimensions.end(), data_set_target_dimensions.begin(), data_set_target_dimensions.end());

        const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());

        if (cudaMallocHost(&targets_host, target_size * sizeof(float)) != cudaSuccess)
            cout << "Targets host allocation error" << endl;

        if (cudaMalloc(&targets_device, target_size * sizeof(float)) != cudaSuccess)
            cout << "Targets allocation error" << endl;
    }
}


void BatchCuda::copy_device()
{
    const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
    const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());
    const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());

    if (cudaMemcpy(inputs_device, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Inputs copy error" << endl;

    if (!decoder_dimensions.empty())
        if (cudaMemcpy(decoder_device, decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "Decoder copy error" << endl;

    if (cudaMemcpy(targets_device, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Targets copy error" << endl;
}


Tensor<type, 2> BatchCuda::get_inputs_device() const
{
    const Index inputs_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);

    Tensor<type, 2> inputs(samples_number, inputs_number);

    inputs.setZero();

    if (cudaMemcpy(inputs.data(), inputs_device, samples_number * inputs_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Cuda matrix memcpy error" << endl;

    return inputs;
}


Tensor<type, 2> BatchCuda::get_decoder_device() const
{
    const Index decoder_number = data_set->get_raw_variables_number(DataSet::VariableUse::Decoder);

    Tensor<type, 2> decoder(samples_number, decoder_number);

    decoder.setZero();

    if (cudaMemcpy(decoder.data(), inputs_device, samples_number * decoder_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Cuda matrix memcpy error" << endl;

    return decoder;
}


Tensor<type, 2> BatchCuda::get_targets_device() const
{
    const Index targets_number = target_dimensions[1];

    Tensor<type, 2> targets(samples_number, targets_number);

    targets.setZero();

    if (cudaMemcpy(targets.data(), targets_device, samples_number * targets_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Cuda matrix memcpy error" << endl;

    return targets;
}


vector<pair<type*, dimensions>> BatchCuda::get_input_pairs_device() const
{
    vector<pair<type*, dimensions>> input_pairs = { {inputs_device, input_dimensions} };

    if (!decoder_dimensions.empty())
        input_pairs.insert(input_pairs.begin(), { decoder_device, decoder_dimensions });

    return input_pairs;
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
