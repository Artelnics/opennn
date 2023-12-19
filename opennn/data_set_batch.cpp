#include "data_set_batch.h"

namespace opennn
{


void DataSetBatch::fill(const Tensor<Index, 1>& samples,
                        const Tensor<Index, 1>& inputs,
                        const Tensor<Index, 1>& targets)
{
    const Tensor<type, 2>& data = data_set_pointer->get_data();
    const Tensor<Index, 1>& input_variables_dimensions = data_set_pointer->get_input_variables_dimensions();

    if(input_variables_dimensions.size() == 1)
    {
        fill_submatrix(data, samples, inputs, this->inputs(0).get_data());
    }
    else if(input_variables_dimensions.size() == 3)
    {
        const Index rows_number = input_variables_dimensions(0);
        const Index columns_number = input_variables_dimensions(1);
        const Index channels_number = input_variables_dimensions(2);

        TensorMap<Tensor<type, 4>> inputs = this->inputs(0).to_tensor_map<4>();

        #pragma omp parallel for

        for(Index image = 0; image < batch_size; image++)
        {
            Index index = 0;

            for(Index row = 0; row < rows_number; row++)
            {
                for(Index column = 0; column < columns_number; column++)
                {
                    for(Index channel = 0; channel < channels_number ; channel++)
                    {
                        inputs(image, row, column, channel) = data(image, index);

                        index++;
                    }
                }
            }
        }
/*
        const bool augmentation = data_set_pointer->get_augmentation();

        if(augmentation) perform_augmentation();
*/
    }
    fill_submatrix(data, samples, targets, this->targets.get_data());
}


void DataSetBatch::perform_augmentation()
{
/*
    const Tensor<Index, 1>& input_variables_dimensions = data_set_pointer->get_input_variables_dimensions();

    const Index rows_number = input_variables_dimensions(0);
    const Index columns_number = input_variables_dimensions(1);
    const Index channels_number = input_variables_dimensions(2);
    const Index input_size = rows_number*columns_number*channels_number;

//    TensorMap<Tensor<type, 4>> inputs(inputs_data,
//                                      batch_size,
//                                      rows_number,
//                                      columns_number,
//                                      channels_number);

    const bool random_reflection_axis_x = data_set_pointer->get_random_reflection_axis_x();
    const bool random_reflection_axis_y = data_set_pointer->get_random_reflection_axis_y();
    const type random_rotation_minimum = data_set_pointer->get_random_rotation_minimum();
    const type random_rotation_maximum = data_set_pointer->get_random_rotation_maximum();
    const type random_rescaling_minimum = type(0);
    const type random_rescaling_maximum = type(0);
    const type random_horizontal_translation_minimum = data_set_pointer->get_random_horizontal_translation_minimum();
    const type random_horizontal_translation_maximum = data_set_pointer->get_random_horizontal_translation_maximum();
    const type random_vertical_translation_minimum = data_set_pointer->get_random_vertical_translation_minimum();
    const type random_vertical_translation_maximum = data_set_pointer->get_random_vertical_translation_maximum();

    for(Index batch = 0; batch < batch_size; batch++)
    {

        TensorMap<Tensor<type, 3>> current_image((this->inputs(0).get_data()) + batch*input_size,
                                                 rows_number,
                                                 columns_number,
                                                 channels_number);

        if(random_reflection_axis_x)
        {
            reflect_image_x(current_image, current_image);
        }

        if(random_reflection_axis_y)
        {
            reflect_image_y(current_image, current_image);
        }

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
        {
            const type angle = (random_rotation_minimum < random_rotation_maximum)
                    ? random_rotation_minimum + rand()
                    : random_rotation_maximum;

            rotate_image(current_image, current_image, angle);
        }

        if(random_rescaling_minimum != 0 && random_rescaling_maximum != 0)
        {
            const type rescaling = (random_rescaling_minimum < random_rescaling_maximum)
                    ? random_rescaling_minimum + rand()
                    : random_rescaling_maximum;

//            rescale_image(current_image, current_image, rescaling);
        }

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
        {
            const type translation = (random_horizontal_translation_minimum < random_rescaling_maximum)
                    ? random_horizontal_translation_minimum + rand()
                    : random_rescaling_maximum;

            translate_image(current_image, current_image, translation);
        }
    }
*/
}


DataSetBatch::DataSetBatch(const Index& new_samples_number, DataSet* new_data_set_pointer)
{
    set(new_samples_number, new_data_set_pointer);
}


void DataSetBatch::set(const Index& new_batch_size, DataSet* new_data_set_pointer)
{
    batch_size = new_batch_size;

    data_set_pointer = new_data_set_pointer;

    const Index input_variables_number = data_set_pointer->get_input_numeric_variables_number();
    const Index target_variables_number = data_set_pointer->get_target_numeric_variables_number();

    const Tensor<Index, 1> input_variables_dimensions = data_set_pointer->get_input_variables_dimensions();

    if(input_variables_dimensions.size() == 1)
    {
        inputs.resize(1);

        Tensor<Index, 1> inputs_dimensions(2);
        inputs_dimensions.setValues({batch_size, input_variables_number});

        inputs(0).set_dimensions(inputs_dimensions);
    }
    else if(input_variables_dimensions.size() == 3)
    {
        const Index channels_number = input_variables_dimensions(0);
        const Index rows_number = input_variables_dimensions(1);
        const Index columns_number = input_variables_dimensions(2);

        Tensor<Index, 1> inputs_dimensions(4);
        inputs_dimensions.setValues({batch_size, channels_number, rows_number, columns_number});

        inputs.resize(1);
        inputs(0) = DynamicTensor<type>(inputs_dimensions);
    }

    Tensor<Index, 1> targets_dimensions(2);
    targets_dimensions.setValues({batch_size, target_variables_number});

    targets.set_dimensions(targets_dimensions);
}


Index DataSetBatch::get_batch_samples_number() const
{
    return batch_size;
}


void DataSetBatch::print() const
{
    cout << "Batch" << endl;

    cout << "Inputs dimensions:" << endl;
    cout << inputs(0).get_dimensions() << endl;

    cout << "Dimensions " << endl;
    cout << inputs(0).get_dimensions().dimensions() << endl;

    cout << "Inputs:" << endl;
    if(inputs(0).get_dimensions().size() == 2)
        cout << inputs(0).to_tensor_map<2>() << endl;
    else if(inputs(0).get_dimensions().size() == 4)
        cout << inputs(0).to_tensor_map<4>() << endl;
    cout << "Targets dimensions:" << endl;
    cout << targets.get_dimensions() << endl;

    cout << "Targets:" << endl;
    cout << targets.to_tensor_map<2>() << endl;
}



}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
