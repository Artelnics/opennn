#include "images.h"
#include <stdexcept>
#include <iostream>

namespace opennn
{

Tensor<unsigned char, 3> read_bmp_image(const string& filename)
{
    FILE* file = fopen(filename.data(), "rb");

    if(!file)
        throw runtime_error("Cannot open the file.\n");

    unsigned char info[54];

    fread(info, sizeof(unsigned char), 54, file);

    const Index width_no_padding = abs(*(int*)&info[18]);
    const Index height = abs(*(int*)&info[22]);
    const Index bits_per_pixel = abs(*(int*)&info[28]);

    const int channels = bits_per_pixel == 24
                       ? 3
                       : 1;
    
    // const Index channels = channels;
    
    Index padding = 0;

    const Index width = width_no_padding;

    while((channels*width + padding)% 4 != 0)
        padding++;

    const size_t size = height*(channels*width + padding);

    Tensor<unsigned char, 1> raw_image;

    raw_image.resize(size);

    const int data_offset = *(int*)(&info[0x0A]);
    fseek(file, (long int)(data_offset - 54), SEEK_CUR);

    fread(raw_image.data(), sizeof(unsigned char), size, file);

    fclose(file);

    Tensor<unsigned char, 3> image(height, width, channels);

    const Index xxx = width * channels + padding;

    for(Index i = 0; i < height; i++)
        for(Index j = 0; j < width; ++j)
            for(Index k = 0; k < channels; ++k)
                image(i, j, k) = raw_image[i*xxx  + j*channels + k];
    
    return image;
}


void reflect_image_x(const ThreadPoolDevice* thread_pool_device,
                     TensorMap<Tensor<type, 3>>& image)
{
    const Eigen::array<bool, 3> reflect_horizontal_dimensions = {false, true, false};

    Tensor<type, 3> reversed_image = image.reverse(reflect_horizontal_dimensions);

    image = reversed_image;
}


void reflect_image_y(const ThreadPoolDevice* thread_pool_device,
                     const Tensor<type, 3>& input,
                     Tensor<type, 3>& output)
{
    const Eigen::array<bool, 3> reflect_vertical_dimesions = {true, false, false};

    output.device(*thread_pool_device) = input.reverse(reflect_vertical_dimesions);
}


void rotate_image(const ThreadPoolDevice* thread_pool_device,
                  const Tensor<type, 3>& input,
                  Tensor<type, 3>& output,
                  const type& angle_degree)
{
    const Index width = input.dimension(0);
    const Index height = input.dimension(1);
    const Index channels = input.dimension(2);

    const type rotation_center_x = type(width) / type(2);
    const type rotation_center_y = type(height) / type(2);

    const type angle_rad = -angle_degree * type(3.1415927) / type(180.0);
    const type cos_angle = cos(angle_rad);
    const type sin_angle = sin(angle_rad);

    Tensor<type,2> rotation_matrix(3, 3);

    rotation_matrix.setZero();
    rotation_matrix(0, 0) = cos_angle;
    rotation_matrix(0, 1) = -sin_angle;
    rotation_matrix(1, 0) = sin_angle;
    rotation_matrix(1, 1) = cos_angle;
    rotation_matrix(0, 2) = rotation_center_x - cos_angle * rotation_center_x + sin_angle * rotation_center_y;
    rotation_matrix(1, 2) = rotation_center_y - sin_angle * rotation_center_x - cos_angle * rotation_center_y;
    rotation_matrix(2, 2) = type(1);

    Tensor<type, 1> coordinates(3);
    Tensor<type, 1> transformed_coordinates(3);
    const Eigen::array<IndexPair<Index>, 1> contract_dims = {IndexPair<Index>(1,0)};

    for(Index x = 0; x < width; x++)
    {
        for(Index y = 0; y < height; y++)
        {
            coordinates(0) = type(x);
            coordinates(1) = type(y);
            coordinates(2) = type(1);

            transformed_coordinates = rotation_matrix.contract(coordinates, contract_dims);

            if(transformed_coordinates[0] >= 0 && transformed_coordinates[0] < width
            && transformed_coordinates[1] >= 0 && transformed_coordinates[1] < height)
            {
                for(Index channel = 0; channel < channels; channel++)
                {
                    output(x, y, channel) = input(int(transformed_coordinates[0]),
                                                  int(transformed_coordinates[1]),
                                                  channel);
                }
            }
            else
            {
                for(Index channel = 0; channel < channels; channel++)
                    output(x, y, channel) = type(0);
            }
        }
    }
}


void translate_image(const ThreadPoolDevice* thread_pool_device,
                     const Tensor<type, 3>& input,
                     Tensor<type, 3>& output,
                     const Index& shift)
{
    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    output.setZero();

    const Index height = input.dimension(0);
    const Index width = input.dimension(1);
    const Index channels = input.dimension(2);
    const Index input_size = height*width;

    const Index limit_column = width - shift;

    for(Index i = 0; i < limit_column * channels; i++)
    {
        const Index channel = i % channels;
        const Index raw_variable = i / channels;

        const TensorMap<const Tensor<type, 2>> input_column_map(input.data() + raw_variable*height + channel*input_size,
                                                           height,
                                                          1);

        TensorMap<Tensor<type, 2>> output_column_map(output.data() + (raw_variable + shift)*height + channel*input_size,
                                                     height,
                                                     1);

        output_column_map = input_column_map;
    }
}


Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>& image, 
                                        const int& rows_number,
                                        const int& columns_number,
                                        const int& padding)
{
    Tensor<unsigned char, 1> data_without_padding(image.size() - padding*rows_number);

    unsigned char* image_data = image.data();

    const int channels = 3;

    if(rows_number % 4 == 0)
    {
        copy(image_data,
             image_data + columns_number * channels * rows_number,
             data_without_padding.data());
    }
    else
    {
        for(int i = 0; i < rows_number; i++)
        {
            if(i == 0)
            {
                copy(image_data,
                     image_data + columns_number * channels, data_without_padding.data());
            }
            else
            {
                copy(image_data + channels * columns_number * i + padding * i,
                    image_data + channels * columns_number * (i+1) + padding * i,
                    data_without_padding.data() + channels * columns_number * i);
            }
        }
    }

    return data_without_padding;
}


void rescale_image(const ThreadPoolDevice*, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const type&)
{

}

} // namespace opennn
