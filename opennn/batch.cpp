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
#include "images.h"
#include "language_data_set.h"

namespace opennn
{

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& target_indices,
                 const vector<Index>& context_indices)
{
    const Tensor<type, 2>& data = data_set->get_data();

    ImageDataSet* image_data_set = dynamic_cast<ImageDataSet*>(data_set);

    if(image_data_set && image_data_set->get_augmentation())
    {
/*
        // @TODO
        Tensor<type, 2>& augmented_data = perform_augmentation(data);

        fill_tensor_data(augmented_data, sample_indices, input_indices, input_data);
*/
    }
    else
    {
        fill_tensor_data(data, sample_indices, input_indices, input_tensor.data());
    }

    if (has_context())
        fill_tensor_data(data, sample_indices, context_indices, context_tensor.data());

    fill_tensor_data(data, sample_indices, target_indices, target_tensor.data());
}


Tensor<type, 2> Batch::perform_augmentation(const Tensor<type, 2>& data)
{
    ImageDataSet* image_data_set = static_cast<ImageDataSet*>(data_set);

    const dimensions& input_dimensions = data_set->get_input_dimensions();

    const Index input_height = input_dimensions[0];
    const Index input_width = input_dimensions[1];
    const Index channels = input_dimensions[2];

    TensorMap<Tensor<type, 4>> inputs(input_tensor.data(),
                                      batch_size,
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

    for(Index batch_index = 0; batch_index < batch_size; batch_index++)
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

            translate_image(thread_pool_device.get(), image, image, translation);
        }
    } 

    Tensor<type, 2> todo;
    return todo;
}


Batch::Batch(const Index& new_samples_number, DataSet* new_data_set)
{
    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    set(new_samples_number, new_data_set);
}


void Batch::set(const Index& new_batch_size, DataSet* new_data_set)
{        
    if (!new_data_set) return;

    batch_size = new_batch_size;
    data_set = new_data_set;

    const Index input_variables_number = data_set->get_variables_number(DataSet::VariableUse::Input);
    const Index target_variables_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    const dimensions& data_set_input_dimensions = data_set->get_input_dimensions();
    const dimensions& data_set_target_dimensions = data_set->get_target_dimensions();

    if(data_set_input_dimensions.size() == 1)
    {
        input_dimensions = {{batch_size, input_variables_number}};
        input_tensor.resize(batch_size*input_variables_number);
    }
    else if(data_set_input_dimensions.size() == 2)
    {
        const Index rows_number = data_set_input_dimensions[0];
        const Index columns_number = data_set_input_dimensions[1];

        input_dimensions = {{batch_size, rows_number, columns_number}};
        input_tensor.resize(batch_size*rows_number* columns_number);
    }
    else if(data_set_input_dimensions.size() == 3)
    {
        const Index rows_number = data_set_input_dimensions[0];
        const Index columns_number = data_set_input_dimensions[1];
        const Index channels = data_set_input_dimensions[2];

        input_dimensions = {{batch_size, rows_number, columns_number, channels}};
        input_tensor.resize(batch_size*channels*rows_number*columns_number);
    }

    if(data_set_target_dimensions.size() == 1)
    {
        target_dimensions = {{batch_size, target_variables_number}};
        target_tensor.resize(batch_size*target_variables_number);
    }
    else if(data_set_target_dimensions.size() == 2)
    {
        const Index rows_number = data_set_target_dimensions[0];
        const Index columns_number = data_set_target_dimensions[1];

        target_dimensions = {{batch_size, rows_number, columns_number}};
        target_tensor.resize(batch_size*rows_number*columns_number);
    }
    else if(data_set_target_dimensions.size() == 3)
    {
        const Index rows_number = data_set_target_dimensions[0];
        const Index columns_number = data_set_target_dimensions[1];
        const Index channels = data_set_target_dimensions[2];

        target_dimensions = {{batch_size, rows_number, columns_number, channels}};

        target_tensor.resize(batch_size*channels*rows_number*columns_number);
    }

    // LanguageDataSet

    if(is_instance_of<LanguageDataSet>(data_set))
    {
        LanguageDataSet* language_data_set = static_cast<LanguageDataSet*>(data_set);

        const Index context_variables_number = language_data_set->get_variables_number(DataSet::VariableUse::Context);

        const dimensions data_set_context_dimensions = language_data_set->get_context_dimensions();

        if(data_set_context_dimensions.size() == 1)
        {
            context_dimensions = {{batch_size, context_variables_number}};

            context_tensor.resize(batch_size*context_variables_number);
        }
        else if(data_set_context_dimensions.size() == 2)
        {
            const Index rows_number = context_dimensions[0];
            const Index columns_number = context_dimensions[1];

            context_dimensions = {{batch_size, rows_number, columns_number}};

            context_tensor.resize(batch_size*rows_number*columns_number);
        }
        else if(data_set_context_dimensions.size() == 3)
        {
            const Index channels = context_dimensions[0];
            const Index rows_number = context_dimensions[1];
            const Index columns_number = context_dimensions[2];

            context_dimensions = {{batch_size, channels, rows_number, columns_number}};

            context_tensor.resize(batch_size*channels*rows_number*columns_number);
        }
    }
}


Index Batch::get_batch_samples_number() const
{
    return batch_size;
}


void Batch::print() const
{
    const Index inputs_rank = input_dimensions.size();
    const Index targets_rank = target_dimensions.size();

    cout << "Batch" << endl
         << "Inputs:" << endl
         << "Inputs dimensions:" << endl;

    for(Index i = 0; i < inputs_rank; i++)
        cout << input_dimensions[i] << endl;

    if(inputs_rank == 4)
    {
        const TensorMap<Tensor<type, 4>> inputs((type*)input_tensor.data(),
                                                input_dimensions[0],
                                                input_dimensions[1],
                                                input_dimensions[2],
                                                input_dimensions[3]);

        cout << inputs << endl;
    }

    cout << "Targets:" << endl
         << "Targets dimensions:" << endl;

    for(Index i = 0; i < targets_rank; i++)
        cout << target_dimensions[i] << endl;

    const TensorMap<Tensor<type, 2>> targets((type*)target_tensor.data(),
                                             target_dimensions[0],
                                             target_dimensions[1]);

    cout << targets << endl;
}


bool Batch::is_empty() const
{
    return input_tensor.size() == 0;
}


bool Batch::has_context() const
{
    return context_tensor.size() != 0;
}


vector<pair<type*, dimensions>> Batch::get_input_pairs() const
{
    vector<pair<type*, dimensions>> input_pairs = {{(type*)input_tensor.data(), input_dimensions}};

    if (has_context())
        input_pairs.push_back({(type*)context_tensor.data(), context_dimensions});

    return input_pairs;
}


pair<type*, dimensions> Batch::get_target_pair() const
{
    return { (type*)target_tensor.data() , target_dimensions};
}

} // namespace opennn

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
