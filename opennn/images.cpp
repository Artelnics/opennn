#include "images.h"
#include "tensors.h"

namespace opennn
{

    Tensor<unsigned char, 1> read_bmp_image(const string& filename)
    {
        Index image_height = 0;
        Index image_width = 0;
        Index channels_number = 0;
        Index padding = 0;

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
/*
            const int rows_number = int(get_image_height());
            const int columns_number = int(get_image_width());

            Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, columns_number, padding);

            const Eigen::array<Eigen::Index, 3> dims_3D = { channels, rows_number, columns_number };
            const Eigen::array<Eigen::Index, 1> dims_1D = { rows_number * columns_number };

            Tensor<unsigned char, 1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2, 0).reshape(dims_1D); // row_major
            Tensor<unsigned char, 1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1, 0).reshape(dims_1D); // row_major
            Tensor<unsigned char, 1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0, 0).reshape(dims_1D); // row_major

            Tensor<unsigned char, 1> red_channel_flatted_sorted(red_channel_flatted.size());
            Tensor<unsigned char, 1> green_channel_flatted_sorted(green_channel_flatted.size());
            Tensor<unsigned char, 1> blue_channel_flatted_sorted(blue_channel_flatted.size());

            red_channel_flatted_sorted.setZero();
            green_channel_flatted_sorted.setZero();
            blue_channel_flatted_sorted.setZero();

            sort_channel(red_channel_flatted, red_channel_flatted_sorted, columns_number);
            sort_channel(green_channel_flatted, green_channel_flatted_sorted, columns_number);
            sort_channel(blue_channel_flatted, blue_channel_flatted_sorted, columns_number);

            Tensor<unsigned char, 1> red_green_concatenation(red_channel_flatted_sorted.size() + green_channel_flatted_sorted.size());
            red_green_concatenation = red_channel_flatted_sorted.concatenate(green_channel_flatted_sorted, 0); // To allow a double concatenation

            image = red_green_concatenation.concatenate(blue_channel_flatted_sorted, 0);
*/
        }

        return image;
    }



/// @todo ChatGPT gives something easier

void read_bmp_image(const string& filename, Tensor<type, 3>& image)
{
//    Tensor<Tensor<type, 1>, 1> image;

    FILE* file = fopen(filename.data(), "rb");

    if(!file)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp_image() method.\n"
               << "Couldn't open the file.\n";

        throw runtime_error(buffer.str());
    }

    /*
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
        const int rows_number = int(image_height);
        const int columns_number = int(image_width);

        const Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, columns_number, padding);

        const Eigen::array<Eigen::Index, 3> dims_3D = {channels, rows_number, columns_number};
        const Eigen::array<Eigen::Index, 1> dims_1D = {rows_number* columns_number};

        Tensor<unsigned char,1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0,0).reshape(dims_1D); // row_major

        Tensor<unsigned char,1> red_channel_flatted_sorted(red_channel_flatted.size());
        Tensor<unsigned char,1> green_channel_flatted_sorted(green_channel_flatted.size());
        Tensor<unsigned char,1> blue_channel_flatted_sorted(blue_channel_flatted.size());

        red_channel_flatted_sorted.setZero();
        green_channel_flatted_sorted.setZero();
        blue_channel_flatted_sorted.setZero();

        sort_channel(red_channel_flatted, red_channel_flatted_sorted, columns_number);
        sort_channel(green_channel_flatted, green_channel_flatted_sorted, columns_number);
        sort_channel(blue_channel_flatted, blue_channel_flatted_sorted, columns_number);

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
    */
}


/// @todo bad variables names

void sort_channel(Tensor<unsigned char,1>& original, Tensor<unsigned char,1>& sorted, const int& columns_number)
{
    unsigned char* aux_row = nullptr;

    aux_row = (unsigned char*)malloc(size_t(columns_number*sizeof(unsigned char)));

    const int rows_number = int(original.size()/ columns_number);

    for(int i = 0; i <rows_number; i++)
    {
        copy(/*execution::par,*/ 
            original.data() + columns_number * rows_number - (i + 1) * columns_number,
            original.data() + columns_number * rows_number - i * columns_number,
            aux_row);

        // reverse(aux_row, aux_row + raw_variables_number); //uncomment this if the lower right corner px should be in the upper left corner.

        copy(/*execution::par,*/ 
             aux_row, aux_row + columns_number,
             sorted.data() + columns_number * i);
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
                    output(x, y, channel) = input(int(transformed_coordinates[0]),
                                                  int(transformed_coordinates[1]),
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

        const TensorMap<const Tensor<type, 2>> input_column_map(input.data() + column*height + channel*input_size,
                                                           height,
                                                          1);

        TensorMap<Tensor<type, 2>> output_column_map(output.data() + (column + shift)*height + channel*input_size,
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

    const int channels = 3;

    if(rows_number % 4 == 0)
    {
        copy(/*execution::par,*/ 
             image.data(), 
             image.data() + columns_number * channels * rows_number,
             data_without_padding.data());
    }
    else
    {
        for(int i = 0; i < rows_number; i++)
        {
            if(i == 0)
            {
                copy(/*execution::par,*/ 
                    image.data(), image.data() + columns_number * channels, data_without_padding.data());

            }
            else
            {
                copy(/*execution::par,*/ 
                    image.data() + channels * columns_number * i + padding * i,
                    image.data() + channels * columns_number * (i + 1) + padding * i,
                    data_without_padding.data() + channels * columns_number * i);

            }
        }
    }

    return data_without_padding;
}


Tensor<unsigned char, 1> resize_image(Tensor<unsigned char, 1> &data,
                                      const Index &image_width,
                                      const Index &image_height,
                                      const Index &channels_number)
{
    /*
    const fs::path path = data_source_path;

    if(data_source_path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }

    has_raw_variables_names = true;
    has_rows_labels = true;

    separator = Separator::None;

    vector<fs::path> folder_paths;
    vector<fs::path> image_paths;

    int classes_number = 0;
    int images_total_number = 0;

    for(const auto & entry_path : fs::directory_iterator(path))
    {
        if(entry_path.path().string().find(".DS_Store") != string::npos)
        {
            cout << ".DS_Store found in : " << entry_path << endl;
        }
        else
        {
            fs::path actual_directory = entry_path.path().string();

            folder_paths.emplace_back(actual_directory);
            classes_number++;

            for(const auto & entry_image : fs::directory_iterator(actual_directory))
            {
                if(entry_image.path().string().find(".DS_Store") != string::npos)
                {
                    cout << ".DS_Store found in : " << entry_image.path() << endl;
                }
                else
                {
                    image_paths.emplace_back(entry_image.path().string());
                    images_total_number++;
                }
            }
        }
    }

    images_number = images_total_number;

    const int image_size = width * height * channels;

    Tensor<type, 2> imageDataAux(image_data.dimension(0), image_data.dimension(1));
    imageDataAux = image_data;

    if(classes_number == 2)
    {
        Index binary_raw_variables_number = 1;
        data.resize(images_number, image_size + binary_raw_variables_number);
        imageDataAux.resize(images_number, image_size + binary_raw_variables_number);
    }
    else
    {
        data.resize(images_number, image_size + classes_number);
        imageDataAux.resize(images_number, image_size + classes_number);
    }

//    memcpy(data.data(), image_data.data(), images_number * image_size * sizeof(type));

    copy(/*execution::par,
    image_data.data(), image_data.data() + images_number * image_size, data.data());

    rows_labels.resize(images_number);

    Index row_index = 0;

    for(Index i = 0; i < classes_number; i++)
    {
        vector<string> images_paths;
        Index images_in_folder = 0;

        for(const auto & entry : fs::directory_iterator(folder_paths[i]))
        {
            if(entry.path().string().find(".DS_Store") != string::npos)
            {
                cout << ".DS_Store found in : " << entry << endl;
            }
            else
            {
                images_paths.emplace_back(entry.path().string());
                images_in_folder++;
            }
        }

        for(Index j = 0;  j < images_in_folder; j++)
        {

            if(classes_number == 2 && i == 0)
            {
                data(row_index, image_size) = 1;
            }
            else if(classes_number == 2 && i == 1)
            {
                data(row_index, image_size) = type(0);
            }
            else
            {
                data(row_index, image_size + i) = 1;
            }

            rows_labels(row_index) = images_paths[j];

            row_index++;
        }
    }

    raw_variables.resize(image_size + 1);

    // Input raw_variables

    Index raw_variable_index = 0;

    for(Index i = 0; i < channels; i++)
    {
        for(Index j = 0; j < width; j++)
        {
            for(Index k = 0; k < height ; k++)
            {
                raw_variables(raw_variable_index).name= "pixel_" + to_string(i+1)+ "_" + to_string(j+1) + "_" + to_string(k+1);
                raw_variables(raw_variable_index).type = ColumnType::Numeric;
                raw_variables(raw_variable_index).raw_variable_use = VariableUse::Input;
                raw_variables(raw_variable_index).scaler = Scaler::MinimumMaximum;
                raw_variable_index++;
            }
        }
    }

    // Target raw_variables

    raw_variables(image_size).name = "class";

    if(classes_number == 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Invalid number of categories. The minimum is 2 and you have 1.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<string, 1> categories(classes_number);

    for(Index i = 0 ; i < classes_number; i++)
    {
        categories(i) = folder_paths[i].filename().string();
    }

    raw_variables(image_size).raw_variable_use = VariableUse::Target;
    raw_variables(image_size).categories = categories;

    raw_variables(image_size).categories_uses.resize(classes_number);
    raw_variables(image_size).categories_uses.setConstant(VariableUse::Target);

    if(classes_number == 2)
    {
        raw_variables(image_size).type = ColumnType::Binary;
    }
    else
    {
        raw_variables(image_size).type = ColumnType::Categorical;
    }

    samples_uses.resize(images_number);
    split_samples_random();

    image_width = width;
    image_height = height;
    channels_number = channels;

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({channels, width, height});
    */

    return Tensor<unsigned char, 1>();
}

}
