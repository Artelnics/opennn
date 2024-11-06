#include "batch.h"
#include "tensors.h"
#include "image_data_set.h"
#include "images.h"
#include "language_data_set.h"

namespace opennn
{

Batch::~Batch()
{
    delete thread_pool;
    delete thread_pool_device;
}


void Batch::fill(const Tensor<Index, 1>& samples_indices,
                 const Tensor<Index, 1>& inputs_indices,
                 const Tensor<Index, 1>& targets_indices,
                 const Tensor<Index, 1>& context_indices)
{
    const Tensor<type, 2>& data = data_set->get_data();

    ImageDataSet* image_data_set = dynamic_cast<ImageDataSet*>(data_set);

    if (image_data_set && image_data_set->get_augmentation())
    {

        ImageDataSet* image_data_set = static_cast<ImageDataSet*>(data_set);
/*
        // @TODO
        Tensor<type, 2>& augmented_data = perform_augmentation(data);

        fill_tensor_data(augmented_data, samples_indices, inputs_indices, input_data);
*/
    }
    else
    {
        fill_tensor_data(data, samples_indices, inputs_indices, input_data);
    }

    if (has_context)
        fill_tensor_data(data, samples_indices, context_indices, context_data);

    fill_tensor_data(data, samples_indices, targets_indices, targets_data);
}


Tensor<type, 2> Batch::perform_augmentation(const Tensor<type, 2>& data)
{
    ImageDataSet* image_data_set = static_cast<ImageDataSet*>(data_set);

    const dimensions& input_dimensions = data_set->get_input_dimensions();

    const Index input_height = input_dimensions[0];
    const Index input_width = input_dimensions[1];
    const Index channels = input_dimensions[2];

    TensorMap<Tensor<type, 4>> inputs(input_data,
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
        const Tensor<type, 3> image = inputs.chip(batch_index, 0);
        cout << image << endl;
        system("pause");

        if(random_reflection_axis_x)
        {
            //reflect_image_x(thread_pool_device, image);
        }

        if(random_reflection_axis_y)
        {
            //reflect_image_y(thread_pool_device, image);
        }

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
        {
            const type angle = random_rotation_minimum < random_rotation_maximum
                             ? random_rotation_minimum + type(rand())
                             : random_rotation_maximum;

            //rotate_image(thread_pool_device, image, image, angle);
        }

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
        {
            const type translation = random_horizontal_translation_minimum < random_horizontal_translation_maximum
                                   ? random_horizontal_translation_minimum + type(rand())
                                   : random_horizontal_translation_maximum;

            //translate_image(thread_pool_device, image, image, translation);
        }
    } 

    Tensor<type, 2> todo;
    return todo;
}


Batch::Batch(const Index& new_samples_number, DataSet* new_data_set)
{
    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    set(new_samples_number, new_data_set);
}


bool Batch::is_empty() const
{
    return get_batch_samples_number() == 0;
}


void Batch::set(const Index& new_batch_size, DataSet* new_data_set)
{        
    batch_size = new_batch_size;
    data_set = new_data_set;

    const Index input_variables_number = data_set->get_variables_number(DataSet::VariableUse::Input);
    const Index target_variables_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    const dimensions& data_set_input_dimensions = data_set->get_input_dimensions();
    const dimensions& data_set_target_dimensions = data_set->get_target_dimensions();

    if(data_set_input_dimensions.size() == 1)
    {
        input_dimensions = {{batch_size, input_variables_number}};
        inputs_tensor.resize(batch_size*input_variables_number);
    }
    else if(data_set_input_dimensions.size() == 2)
    {
        const Index rows_number = data_set_input_dimensions[0];
        const Index columns_number = data_set_input_dimensions[1];

        input_dimensions = {{batch_size, rows_number, columns_number}};
        inputs_tensor.resize(batch_size*rows_number* columns_number);
    }
    else if(data_set_input_dimensions.size() == 3)
    {
        const Index rows_number = data_set_input_dimensions[0];
        const Index columns_number = data_set_input_dimensions[1];
        const Index channels = data_set_input_dimensions[2];

        input_dimensions = {{batch_size, rows_number, columns_number, channels}};
        inputs_tensor.resize(batch_size*channels*rows_number*columns_number);
    }

    input_data = inputs_tensor.data();

    if(data_set_target_dimensions.size() == 1)
    {
        targets_dimensions = {{batch_size, target_variables_number}};

        targets_tensor.resize(batch_size*target_variables_number);
    }
    else if(data_set_target_dimensions.size() == 2)
    {
        const Index rows_number = data_set_target_dimensions[0];
        const Index columns_number = data_set_target_dimensions[1];

        targets_dimensions = {{batch_size, rows_number, columns_number}};
        targets_tensor.resize(batch_size*rows_number*columns_number);
    }
    else if(data_set_target_dimensions.size() == 3)
    {
        const Index rows_number = data_set_target_dimensions[0];
        const Index columns_number = data_set_target_dimensions[1];
        const Index channels = data_set_target_dimensions[2];

        targets_dimensions = {{batch_size, rows_number, columns_number, channels}};

        targets_tensor.resize(batch_size*channels*rows_number*columns_number);
    }

    targets_data = targets_tensor.data();

    // LanguageDataSet

    if(is_instance_of<LanguageDataSet>(data_set))
    {
        has_context = true;

        LanguageDataSet* language_data_set = static_cast<LanguageDataSet*>(data_set);

        const Index context_variables_number = language_data_set->get_variables_number(DataSet::VariableUse::Context);

        const Tensor<Index, 1> data_set_context_dimensions = language_data_set->get_context_variables_dimensions();

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

        context_data = context_tensor.data();
    }
}


Index Batch::get_batch_samples_number() const
{
    return batch_size;
}


void Batch::print() const
{
    const Index inputs_rank = input_dimensions.size();
    const Index targets_rank = targets_dimensions.size();

    cout << "Batch" << endl
         << "Inputs:" << endl
         << "Inputs dimensions:" << endl;

    for(Index i = 0; i < inputs_rank; i++)
        cout << input_dimensions[i] << endl;

    if(inputs_rank == 4)
    {
        const TensorMap<Tensor<type, 4>> inputs(input_data,
                                                input_dimensions[0],
                                                input_dimensions[1],
                                                input_dimensions[2],
                                                input_dimensions[3]);

        cout << inputs << endl;
    }

    cout << "Targets:" << endl
         << "Targets dimensions:" << endl;

    for(Index i = 0; i < targets_rank; i++)
        cout << targets_dimensions[i] << endl;

    const TensorMap<Tensor<type, 2>> targets(targets_data,
                                             targets_dimensions[0],
                                             targets_dimensions[1]);

    cout << targets << endl;
}


vector<pair<type*, dimensions>> Batch::get_input_pairs() const
{
    vector<pair<type*, dimensions>> input_pairs(has_context ? 2 : 1);

    input_pairs[0] = { input_data, input_dimensions };


    if (has_context)
        input_pairs[1] = { context_data, context_dimensions };
    return input_pairs;
}


pair<type*, dimensions> Batch::get_targets_pair() const
{
    return { targets_data , targets_dimensions };
}

}

// namespace opennn

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
