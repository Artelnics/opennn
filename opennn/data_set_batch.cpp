#include "data_set_batch.h"
#include "tensor_utilities.h"
#include "image_data_set.h"
#include "opennn_images.h"

namespace opennn
{

DataSetBatch::~DataSetBatch()
{

}


void DataSetBatch::fill(const Tensor<Index, 1>& samples_indices,
                        const Tensor<Index, 1>& inputs_indices,
                        const Tensor<Index, 1>& targets_indices)
{
    const Tensor<type, 2>& data = data_set->get_data();

    const Tensor<Index, 1>& input_variables_dimensions = data_set->get_input_variables_dimensions();

    if(input_variables_dimensions.size() == 1)
    {
        fill_submatrix(data, samples_indices, inputs_indices, inputs_data);
    }
    else if(input_variables_dimensions.size() == 3)
    {
        const Index rows_number = input_variables_dimensions(0);
        const Index raw_variables_number = input_variables_dimensions(1);
        const Index channels_number = input_variables_dimensions(2);

        TensorMap<Tensor<type, 4>> inputs(inputs_data,
                                          batch_size,
                                          rows_number,
                                          raw_variables_number,
                                          channels_number);

        /// @todo index will not work. Since it is for images, it will contain all rows in the matrix.

        #pragma omp parallel for

        for(Index image = 0; image < batch_size; image++)
        {
            Index index = 0;

            for(Index row = 0; row < rows_number; row++)
            {
                for(Index column = 0; column < raw_variables_number; column++)
                {
                    for(Index channel = 0; channel < channels_number ; channel++)
                    {
                        inputs(image, row, column, channel) = data(image, index);

                        index++;
                    }
                }
            }
        }

        const bool augmentation = data_set->get_augmentation();

        if(augmentation) perform_augmentation();
    }

    /// todo dimensions

    fill_submatrix(data, samples_indices, targets_indices, targets_data);
}


void DataSetBatch::perform_augmentation() const
{
    ImageDataSet* image_data_set
            = static_cast<ImageDataSet*>(data_set);

    const Tensor<Index, 1>& input_variables_dimensions = data_set->get_input_variables_dimensions();

    const Index rows_number = input_variables_dimensions(0);
    const Index raw_variables_number = input_variables_dimensions(1);
    const Index channels_number = input_variables_dimensions(2);
    const Index input_size = rows_number*raw_variables_number*channels_number;

//    TensorMap<Tensor<type, 4>> inputs(inputs_data,
//                                      batch_size,
//                                      rows_number,
//                                      raw_variables_number,
//                                      channels_number);

    const bool random_reflection_axis_x = image_data_set->get_random_reflection_axis_x();
    const bool random_reflection_axis_y = image_data_set->get_random_reflection_axis_y();
    const type random_rotation_minimum = image_data_set->get_random_rotation_minimum();
    const type random_rotation_maximum = image_data_set->get_random_rotation_maximum();
    const type random_rescaling_minimum = type(0);
    const type random_rescaling_maximum = type(0);
    const type random_horizontal_translation_minimum = image_data_set->get_random_horizontal_translation_minimum();
    const type random_horizontal_translation_maximum = image_data_set->get_random_horizontal_translation_maximum();
    // const type random_vertical_translation_minimum = image_data_set->get_random_vertical_translation_minimum();
    // const type random_vertical_translation_maximum = image_data_set->get_random_vertical_translation_maximum();

    for(Index batch = 0; batch < batch_size; batch++)
    {
/*
        TensorMap<Tensor<type, 3>> image(inputs_tensor.data() + batch*input_size,
                                         rows_number,
                                         raw_variables_number,
                                         channels_number);

        if(random_reflection_axis_x)
        {
            reflect_image_x(image, image);
        }

        if(random_reflection_axis_y)
        {
            reflect_image_y(image, image);
        }

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
        {
            const type angle = (random_rotation_minimum < random_rotation_maximum)
                    ? random_rotation_minimum + type(rand())
                    : random_rotation_maximum;

            rotate_image(image, image, angle);
        }

        if(random_rescaling_minimum != 0 && random_rescaling_maximum != 0)
        {
            const type rescaling = (random_rescaling_minimum < random_rescaling_maximum)
                    ? random_rescaling_minimum + type(rand())
                    : random_rescaling_maximum;

            //rescale_image(image, image, rescaling);

        }

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
        {
            const type translation = (random_horizontal_translation_minimum < random_rescaling_maximum)
                    ? random_horizontal_translation_minimum + type(rand())
                    : random_rescaling_maximum;

            translate_image(image, image, translation);
        }
        */
    }
    
}


DataSetBatch::DataSetBatch(const Index& new_samples_number, DataSet* new_data_set)
{
    set(new_samples_number, new_data_set);
}


void DataSetBatch::set(const Index& new_batch_size, DataSet* new_data_set)
{
    batch_size = new_batch_size;

    data_set = new_data_set;

    const Index input_variables_number = data_set->get_input_variables_number();
    const Index target_variables_number = data_set->get_target_variables_number();

    const Tensor<Index, 1> input_variables_dimensions = data_set->get_input_variables_dimensions();
    const Tensor<Index, 1> target_variables_dimensions = data_set->get_target_variables_dimensions();

    if(input_variables_dimensions.size() == 1)
    {
        inputs_dimensions = {{batch_size, input_variables_number}};

        inputs_tensor.resize(1);
        inputs_tensor.resize(batch_size*input_variables_number);
    }
    else if(input_variables_dimensions.size() == 2)
    {
        const Index rows_number = input_variables_dimensions(0);
        const Index raw_variables_number = input_variables_dimensions(1);

        inputs_dimensions = {{batch_size, rows_number, raw_variables_number}};

        inputs_tensor.resize(1);
        inputs_tensor.resize(batch_size*rows_number*raw_variables_number);
    }
    else if(input_variables_dimensions.size() == 3)
    {
        const Index channels_number = input_variables_dimensions(0);
        const Index rows_number = input_variables_dimensions(1);
        const Index raw_variables_number = input_variables_dimensions(2);

        inputs_dimensions = {{batch_size, channels_number, rows_number, raw_variables_number}};

        inputs_tensor.resize(1);
        inputs_tensor.resize(batch_size*channels_number*rows_number*raw_variables_number);
    }

    inputs_data = inputs_tensor.data();

    if(target_variables_dimensions.size() == 1)
    {
        targets_dimensions = {{batch_size, target_variables_number}};

        targets_tensor.resize(1);
        targets_tensor.resize(batch_size*target_variables_number);
    }
    else if(target_variables_dimensions.size() == 2)
    {
        const Index rows_number = target_variables_dimensions(0);
        const Index raw_variables_number = target_variables_dimensions(1);

        targets_dimensions = {{batch_size, rows_number, raw_variables_number}};

        targets_tensor.resize(1);
        targets_tensor.resize(batch_size*rows_number*raw_variables_number);
    }
    else if(target_variables_dimensions.size() == 3)
    {
        const Index channels_number = target_variables_dimensions(0);
        const Index rows_number = target_variables_dimensions(1);
        const Index raw_variables_number = target_variables_dimensions(2);

        targets_dimensions = {{batch_size, channels_number, rows_number, raw_variables_number}};

        targets_tensor.resize(1);
        targets_tensor.resize(batch_size*channels_number*rows_number*raw_variables_number);
    }

    targets_data = targets_tensor.data();
}


Index DataSetBatch::get_batch_samples_number() const
{
    return batch_size;
}


void DataSetBatch::print() const
{
    cout << "Batch" << endl;

    cout << "Inputs dimensions:" << endl;

    const Index inputs_rank = inputs_dimensions.size();

    for(Index i = 0; i < inputs_rank; i++)
    {
//        cout << inputs_dimensions[i] << endl;
    }

    cout << "Inputs:" << endl;
    cout << inputs_tensor << endl;

    cout << "Targets:" << endl;
    cout << targets_tensor << endl;
}


std::pair<type *, dimensions> DataSetBatch::get_inputs_pair() const
{
    pair<type *, dimensions> inputs;

    inputs.first = inputs_data;
    inputs.second = inputs_dimensions;

    return inputs;
}


std::pair<type *, dimensions> DataSetBatch::get_targets_pair() const
{
    pair<type *, dimensions> targets;

    targets.first = targets_data;
    targets.second = targets_dimensions;

    return targets;
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
