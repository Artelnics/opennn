#include "images.h"
#include "tensors.h"

namespace opennn
{
/*
    Tensor<unsigned char, 1> ImageDataSet::read_bmp_image(const string& filename)
    {
        FILE* file = fopen(filename.data(), "rb");

        if (!file)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                << "void read_bmp_image() method.\n"
                << "Couldn't open the file.\n";

            throw runtime_error(buffer.str());
        }

        unsigned char info[54];
        fread(info, sizeof(unsigned char), 54, file);

        const Index width_no_padding = *(int*)&info[18];
        image_height = *(int*)&info[22];
        const Index bits_per_pixel = *(int*)&info[28];
        int channels;

        bits_per_pixel == 24 ? channels = 3 : channels = 1;

        channels_number = channels;

        padding = 0;

        image_width = width_no_padding;

        while ((channels * image_width + padding) % 4 != 0)
            padding++;

        const size_t size = image_height * (channels_number * image_width + padding);

        Tensor<unsigned char, 1> image(size);
        image.setZero();

        int data_offset = *(int*)(&info[0x0A]);
        fseek(file, (long int)(data_offset - 54), SEEK_CUR);

        fread(image.data(), sizeof(unsigned char), size, file);
        fclose(file);

        if (channels_number == 3)
        {
            const int rows_number = static_cast<int>(get_image_height());
            const int raw_variables_number = static_cast<int>(get_image_width());

            Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, raw_variables_number, padding);

            const Eigen::array<Eigen::Index, 3> dims_3D = { channels, rows_number, raw_variables_number };
            const Eigen::array<Eigen::Index, 1> dims_1D = { rows_number * raw_variables_number };

            Tensor<unsigned char, 1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2, 0).reshape(dims_1D); // row_major
            Tensor<unsigned char, 1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1, 0).reshape(dims_1D); // row_major
            Tensor<unsigned char, 1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0, 0).reshape(dims_1D); // row_major

            Tensor<unsigned char, 1> red_channel_flatted_sorted(red_channel_flatted.size());
            Tensor<unsigned char, 1> green_channel_flatted_sorted(green_channel_flatted.size());
            Tensor<unsigned char, 1> blue_channel_flatted_sorted(blue_channel_flatted.size());

            red_channel_flatted_sorted.setZero();
            green_channel_flatted_sorted.setZero();
            blue_channel_flatted_sorted.setZero();

            sort_channel(red_channel_flatted, red_channel_flatted_sorted, raw_variables_number);
            sort_channel(green_channel_flatted, green_channel_flatted_sorted, raw_variables_number);
            sort_channel(blue_channel_flatted, blue_channel_flatted_sorted, raw_variables_number);

            Tensor<unsigned char, 1> red_green_concatenation(red_channel_flatted_sorted.size() + green_channel_flatted_sorted.size());
            red_green_concatenation = red_channel_flatted_sorted.concatenate(green_channel_flatted_sorted, 0); // To allow a double concatenation

            image = red_green_concatenation.concatenate(blue_channel_flatted_sorted, 0);
        }

        return image;
    }
*/


/// @todo ChatGPT gives something easier

void read_bmp_image(const string& filename, Tensor<type, 3>& image)
{/*
    Tensor<Tensor<type, 1>, 1> image;

    FILE* file = fopen(filename.data(), "rb");

    if(!file)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp_image() method.\n"
               << "Couldn't open the file.\n";

        throw runtime_error(buffer.str());
    }

    unsigned char info[54];

    fread(info, sizeof(unsigned char), 54, file);

    const Index width_no_padding = *(int*)&info[18];
    const Index image_height = *(int*)&info[22];
    const Index bits_per_pixel = *(int*)&info[28];
    int channels;

    bits_per_pixel == 24 ? channels = 3 : channels = 1;
    const Index channels_number = channels;
    Index padding = 0;

    const Index image_width = width_no_padding;

    while((channels*image_width + padding)% 4 != 0)
        padding++;

    const size_t size = image_height*(channels_number*image_width + padding);

    Tensor<Tensor<type, 1>, 1> image_data(2); // One tensor is pixel data and the other one is [height, width, channels]

    Tensor<unsigned char, 1> image(size);
    image.setZero();

    Tensor<type, 1> image_dimensions(3);
    image_dimensions(0) = type(image_height);
    image_dimensions(1) = type(image_width);
    image_dimensions(2) = type(channels_number);

    image_data(1) = image_dimensions;

    int data_offset = *(int*)(&info[0x0A]);
    fseek(file, (long int)(data_offset - 54), SEEK_CUR);

    fread(image.data(), sizeof(unsigned char), size, file);
    fclose(file);

    if(channels_number == 3)
    {
        const int rows_number = static_cast<int>(image_height);
        const int raw_variables_number = static_cast<int>(image_width);

        const Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, raw_variables_number, padding);

        const Eigen::array<Eigen::Index, 3> dims_3D = {channels, rows_number, raw_variables_number};
        const Eigen::array<Eigen::Index, 1> dims_1D = {rows_number*raw_variables_number};

        Tensor<unsigned char,1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0,0).reshape(dims_1D); // row_major

        Tensor<unsigned char,1> red_channel_flatted_sorted(red_channel_flatted.size());
        Tensor<unsigned char,1> green_channel_flatted_sorted(green_channel_flatted.size());
        Tensor<unsigned char,1> blue_channel_flatted_sorted(blue_channel_flatted.size());

        red_channel_flatted_sorted.setZero();
        green_channel_flatted_sorted.setZero();
        blue_channel_flatted_sorted.setZero();

        sort_channel(red_channel_flatted, red_channel_flatted_sorted, raw_variables_number);
        sort_channel(green_channel_flatted, green_channel_flatted_sorted, raw_variables_number);
        sort_channel(blue_channel_flatted, blue_channel_flatted_sorted,raw_variables_number);

        Tensor<unsigned char, 1> red_green_concatenation(red_channel_flatted_sorted.size() + green_channel_flatted_sorted.size());
        red_green_concatenation = red_channel_flatted_sorted.concatenate(green_channel_flatted_sorted,0); // To allow a double concatenation

        image = red_green_concatenation.concatenate(blue_channel_flatted_sorted, 0);

        Tensor<type, 1> image_type(image.size());

        for(Index i = 0; i < image_type.size(); i++)
        {
            image_type(i) = (type)image(i);
        }

        image_data(0) = image_type;
    }

    return image_data;
*/
}


/// @todo bad variables names

void sort_channel(Tensor<unsigned char,1>& original, Tensor<unsigned char,1>& sorted, const int& raw_variables_number)
{
    unsigned char* aux_row = nullptr;

    aux_row = (unsigned char*)malloc(size_t(raw_variables_number*sizeof(unsigned char)));

    const int rows_number = static_cast<int>(original.size()/raw_variables_number);

    for(int i = 0; i <rows_number; i++)
    {
        copy(execution::par, 
            original.data() + raw_variables_number * rows_number - (i + 1) * raw_variables_number,
            original.data() + raw_variables_number * rows_number - i * raw_variables_number,
            aux_row);

        // reverse(aux_row, aux_row + raw_variables_number); //uncomment this if the lower right corner px should be in the upper left corner.

        copy(execution::par, 
             aux_row, aux_row + raw_variables_number, 
             sorted.data() + raw_variables_number * i);
    }
}


void reflect_image_x(const ThreadPoolDevice* thread_pool_device, Tensor<type, 3>& input,
                     Tensor<type, 3>& output)
{
    const Eigen::array<bool, 3> reflect_horizontal_dimesions = {false, true, false};

    output.device(*thread_pool_device) = input.reverse(reflect_horizontal_dimesions);
}


void reflect_image_y(const ThreadPoolDevice* thread_pool_device, Tensor<type, 3>& input,
                     Tensor<type, 3>& output)
{
    const Eigen::array<bool, 3> reflect_vertical_dimesions = {true, false, false};

    output.device(*thread_pool_device) = input.reverse(reflect_vertical_dimesions);
}


// @todo Improve performance

void rotate_image(const ThreadPoolDevice* thread_pool_device, Tensor<type, 3>& input,
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
    Eigen::array<IndexPair<Index>, 1> contract_dims = {IndexPair<Index>(1,0)};

    for(Index x = 0; x < width; x++)
    {
        for(Index y = 0; y < height; y++)
        {
            coordinates(0) = type(x);
            coordinates(1) = type(y);
            coordinates(2) = type(1);

            transformed_coordinates = rotation_matrix.contract(coordinates, contract_dims);

            if(transformed_coordinates[0] >= 0 && transformed_coordinates[0] < width &&
               transformed_coordinates[1] >= 0 && transformed_coordinates[1] < height)
            {
                for(Index channel = 0; channel < channels; channel++)
                {
                    output(x, y, channel) = input(static_cast<int>(transformed_coordinates[0]),
                                                  static_cast<int>(transformed_coordinates[1]),
                                                  channel);
                }
            }
            else
            {
                for(Index channel = 0; channel < channels; channel++)
                {
                    output(x, y, channel) = type(0);
                }
            }
        }
    }
}


void translate_image(Tensor<type, 3>& input,
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
        const Index column = i / channels;

        const TensorMap<const Tensor<type, 2>> input_raw_variable_map(input.data() + column*height + channel*input_size,
                                                           height,
                                                          1);

        TensorMap<Tensor<type, 2>> output_raw_variable_map(output.data() + (column + shift)*height + channel*input_size,
                                                     height,
                                                     1);

        output_raw_variable_map = input_raw_variable_map;
    }
}

}
