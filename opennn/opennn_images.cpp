#include "opennn_images.h"
#include "tensor_utilities.h"

namespace opennn
{

Tensor<Tensor<type, 1>, 1> read_bmp_image_data(const string& filename)
{
    FILE* f = fopen(filename.data(), "rb");

    if(!f)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp_image() method.\n"
               << "Couldn't open the file.\n";

        throw invalid_argument(buffer.str());
    }

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f);

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
    image_dimensions(0) = static_cast<type>(image_height);
    image_dimensions(1) = static_cast<type>(image_width);
    image_dimensions(2) = static_cast<type>(channels_number);

    image_data(1) = image_dimensions;

    int data_offset = *(int*)(&info[0x0A]);
    fseek(f, (long int)(data_offset - 54), SEEK_CUR);

    fread(image.data(), sizeof(unsigned char), size, f);
    fclose(f);

    if(channels_number == 3)
    {
        const int rows_number = static_cast<int>(image_height);
        const int cols_number = static_cast<int>(image_width);

        const Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, cols_number, padding);

        const Eigen::array<Eigen::Index, 3> dims_3D = {channels, rows_number, cols_number};
        const Eigen::array<Eigen::Index, 1> dims_1D = {rows_number*cols_number};

        Tensor<unsigned char,1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0,0).reshape(dims_1D); // row_major

        Tensor<unsigned char,1> red_channel_flatted_sorted(red_channel_flatted.size());
        Tensor<unsigned char,1> green_channel_flatted_sorted(green_channel_flatted.size());
        Tensor<unsigned char,1> blue_channel_flatted_sorted(blue_channel_flatted.size());

        red_channel_flatted_sorted.setZero();
        green_channel_flatted_sorted.setZero();
        blue_channel_flatted_sorted.setZero();

        sort_channel(red_channel_flatted, red_channel_flatted_sorted, cols_number);
        sort_channel(green_channel_flatted, green_channel_flatted_sorted, cols_number);
        sort_channel(blue_channel_flatted, blue_channel_flatted_sorted,cols_number);

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
}


void sort_channel(Tensor<unsigned char,1>& original, Tensor<unsigned char,1>& sorted, const int& cols_number)
{
    unsigned char* aux_row = nullptr;

    aux_row = (unsigned char*)malloc(static_cast<size_t>(cols_number*sizeof(unsigned char)));

    const int rows_number = static_cast<int>(original.size()/cols_number);

    for(int i = 0; i <rows_number; i++)
    {
        memcpy(aux_row, original.data() + cols_number*rows_number - (i+1)*cols_number , static_cast<size_t>(cols_number)*sizeof(unsigned char));

        //        reverse(aux_row, aux_row + cols_number); //uncomment this if the lower right corner px should be in the upper left corner.

        memcpy(sorted.data() + cols_number*i , aux_row, static_cast<size_t>(cols_number)*sizeof(unsigned char));
    }
}


void reflect_image_x(TensorMap<Tensor<type, 3>>& input,
                     TensorMap<Tensor<type, 3>>& output)
{
    const Eigen::array<bool, 3> reflect_horizontal_dimesions = {false, true, false};

    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    output = input.reverse(reflect_horizontal_dimesions);
}


void reflect_image_y(TensorMap<Tensor<type, 3>>& input,
                     TensorMap<Tensor<type, 3>>& output)
{
    const Eigen::array<bool, 3> reflect_vertical_dimesions = {true, false, false};

    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    output = input.reverse(reflect_vertical_dimesions);
}


void rotate_image(TensorMap<Tensor<type, 3>>& input,
                  TensorMap<Tensor<type, 3>>& output,
                  const type& angle_degree)
{/*
    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    const Index width = input.dimension(0);
    const Index height = input.dimension(1);
    const Index channels = input.dimension(2);

    const type rotation_center_x = width / 2.0;
    const type rotation_center_y = height / 2.0;

    const type angle_rad = -angle_degree * M_PI / 180.0;
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
    rotation_matrix(2, 2) = 1;

    Tensor<type, 1> coordinates(3);
    Tensor<type, 1> transformed_coordinates(3);
    Eigen::array<IndexPair<Index>, 1> contract_dims = {IndexPair<Index>(1,0)};

    for(Index x = 0; x < width; x++)
    {
        for(Index y = 0; y < height; y++)
        {
            coordinates(0) = x;
            coordinates(1) = y;
            coordinates(2) = 1;

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
                    output(x, y, channel) = 0;
                }
            }

        }
    }
*/
}


void translate_image(TensorMap<Tensor<type, 3>>& input,
                     TensorMap<Tensor<type, 3>>& output,
                     const Index& shift)
{
    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    output.setZero();

    Index height = input.dimension(0);
    Index width = input.dimension(1);
    Index channels = input.dimension(2);
    Index input_size = height*width;

    Index limit_column = width - shift;

    for(Index i = 0; i < limit_column * channels; i++)
    {
        Index channel = i % channels;
        Index column = i / channels;

        TensorMap<const Tensor<type, 2>> input_column_map(input.data() + column*height + channel*input_size,
                                                          height,
                                                          1);

        TensorMap<Tensor<type, 2>> output_column_map(output.data() + (column + shift)*height + channel*input_size,
                                                     height,
                                                     1);
        output_column_map= input_column_map;
    }
}


Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>& img, const int& rows_number,const int& cols_number, const int& padding)
{
    Tensor<unsigned char, 1> data_without_padding(img.size() - padding*rows_number);

    const int channels = 3;

    if(rows_number % 4 == 0)
    {
        memcpy(data_without_padding.data(), img.data(), static_cast<size_t>(cols_number*channels*rows_number)*sizeof(unsigned char));
    }
    else
    {
        for(int i = 0; i < rows_number; i++)
        {
            if(i == 0)
            {
                memcpy(data_without_padding.data(), img.data(), static_cast<size_t>(cols_number*channels)*sizeof(unsigned char));
            }
            else
            {
                memcpy(data_without_padding.data() + channels*cols_number*i, img.data() + channels*cols_number*i + padding*i, static_cast<size_t>(cols_number*channels)*sizeof(unsigned char));
            }
        }
    }

    return data_without_padding;
}


Tensor<Tensor<type, 1>, 1> propose_single_random_region(const Tensor<Tensor<type, 1>, 1>& image_data, const Index& width_to_resize, const Index& height_to_resize)
{
    const Index image_height = image_data(1)(0);
    const Index image_width = image_data(1)(1);
    const Index channels_number = image_data(1)(2);

    Index x_center = rand() % image_width;
    Index y_center = rand() % image_height;

    Index x_top_left;
    Index y_top_left;

    if(x_center == 0){x_top_left = 0;}else{x_top_left = rand() % x_center;}
    if(y_center == 0){y_top_left = 0;} else{y_top_left = rand() % y_center;}

    Index x_bottom_right;

    if(x_top_left == 0){x_bottom_right = rand()%(image_width - (x_center + 1) + 1) + (x_center + 1);}
    else{x_bottom_right = rand()%(image_width - x_center + 1) + x_center;}

    Index y_bottom_right;

    if(y_top_left == 0){y_bottom_right = rand()%(image_height - (y_center + 1) + 1) + (y_center + 1);}
    else{y_bottom_right = rand() % (image_height - y_center + 1) + y_center;}

    const Index region_width = abs(x_top_left - x_bottom_right);
    const Index region_height = abs(y_top_left - y_bottom_right);


    Tensor<type, 1> region_parameters(4);
    region_parameters.setValues({static_cast<type>(x_top_left), static_cast<type>(y_top_left),
                                 static_cast<type>(x_bottom_right), static_cast<type>(y_bottom_right)});
    Tensor<type, 1> random_region(channels_number * region_width * region_height);

    // We resize the region after its random proposal

    random_region = get_bounding_box(image_data, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    Tensor<type, 1> resized_random_region(channels_number * width_to_resize * height_to_resize);

    resized_random_region = resize_proposed_region(random_region, channels_number, region_width,
                                                   region_height, width_to_resize, height_to_resize);

    Tensor<Tensor<type, 1>, 1> region(2);
    region(0).resize(resized_random_region.size());
    region(1).resize(region_parameters.size());

    region(0) = resized_random_region;
    region(1) = region_parameters;

    return region;
}


type intersection_over_union(const Index& x_top_left_box_1,
                             const Index& y_top_left_box_1,
                             const Index& x_bottom_right_box_1,
                             const Index& y_bottom_right_box_1,
                             const Index& x_top_left_box_2,
                             const Index& y_top_left_box_2,
                             const Index& x_bottom_right_box_2,
                             const Index& y_bottom_right_box_2)
{
    Index intersection_x_top_left = max(x_top_left_box_1, x_top_left_box_2);
    Index intersection_y_top_left = max(y_top_left_box_1, y_top_left_box_2);
    Index intersection_x_bottom_right = min(x_bottom_right_box_1, x_bottom_right_box_2);
    Index intersection_y_bottom_right = min(y_bottom_right_box_1, y_bottom_right_box_2);

    if((intersection_x_bottom_right < intersection_x_top_left) || (intersection_y_bottom_right < intersection_y_top_left)) return 0;

    type intersection_area = static_cast<type>((intersection_x_bottom_right - intersection_x_top_left) * (intersection_y_bottom_right - intersection_y_top_left));

    type ground_truth_bounding_box_area = (x_bottom_right_box_1 - x_top_left_box_1) *
                                    (y_bottom_right_box_1 - y_top_left_box_1);

    type proposed_bounding_box_area = (x_bottom_right_box_2 - x_top_left_box_2) *
                                        (y_bottom_right_box_2 - y_top_left_box_2);

    type union_area = ground_truth_bounding_box_area + proposed_bounding_box_area - intersection_area;

    type intersection_over_union = static_cast<type>(intersection_area / union_area);

    return intersection_over_union;
}


Tensor<type, 1> get_bounding_box(const Tensor<Tensor<type, 1>, 1>& image,
                                          const Index& x_top_left, const Index& y_top_left,
                                          const Index& x_bottom_right, const Index& y_bottom_right)
{
    const Index channels_number = image(1)(2);
    const Index height = image(1)(0);
    const Index width = image(1)(1);

    const Index image_size_single_channel = height * width;

    const Index bounding_box_width = abs(x_top_left - x_bottom_right);
    const Index bounding_box_height = abs(y_top_left - y_bottom_right);
    const Index bounding_box_single_channel_size = bounding_box_width * bounding_box_height;

    Tensor<type, 1> bounding_box_data;
    bounding_box_data.resize(channels_number * bounding_box_single_channel_size);

    const Index pixel_loop_start = width * (height - y_bottom_right) + x_top_left;
    const Index pixel_loop_end = width * (height - 1 - y_top_left) + x_bottom_right;

    if(channels_number == 3)
    {
        Tensor<type, 1> image_red_channel_flatted_sorted(image_size_single_channel);
        Tensor<type, 1> image_green_channel_flatted_sorted(image_size_single_channel);
        Tensor<type, 1> image_blue_channel_flatted_sorted(image_size_single_channel);

        image_red_channel_flatted_sorted = image(0).slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));
        image_green_channel_flatted_sorted = image(0).slice(Eigen::array<Eigen::Index, 1>({image_size_single_channel}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));
        image_blue_channel_flatted_sorted = image(0).slice(Eigen::array<Eigen::Index, 1>({2 * image_size_single_channel}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));

        Tensor<type, 1> bounding_box_red_channel(bounding_box_single_channel_size);
        Tensor<type, 1> bounding_box_green_channel(bounding_box_single_channel_size);
        Tensor<type, 1> bounding_box_blue_channel(bounding_box_single_channel_size);

        Index data_index = 0;

        for(Index i = pixel_loop_start; i <= pixel_loop_end - 1; i++)
        {
            const int height_number = (int)(i/height);

            const Index left_margin = height_number * width + x_top_left;
            const Index right_margin = height_number * width + x_bottom_right;

            if(i >= left_margin && i < right_margin)
            {
                bounding_box_red_channel(data_index) = image_red_channel_flatted_sorted[i];
                bounding_box_green_channel(data_index) = image_green_channel_flatted_sorted[i];
                bounding_box_blue_channel(data_index) = image_blue_channel_flatted_sorted[i];

                data_index++;
            }
        }

        Tensor<type, 1> red_green_concatenation(bounding_box_red_channel.size() + bounding_box_green_channel.size());
        red_green_concatenation = bounding_box_red_channel.concatenate(bounding_box_green_channel, 0); // To allow a double concatenation
        bounding_box_data = red_green_concatenation.concatenate(bounding_box_blue_channel, 0);
    }
    else
    {
        Index data_index = 0;

        for(Index i = pixel_loop_start; i <= pixel_loop_end - 1; i++)
        {
            const int height_number = (int)(i/height);

            const Index left_margin = height_number * width + x_top_left;
            const Index right_margin = height_number * width + x_bottom_right;

            if(i >= left_margin && i < right_margin)
            {
                bounding_box_data(data_index) = image(0)[i];
                data_index++;
            }
        }
    }

    return bounding_box_data;
}


Tensor<type, 1> get_ground_truth_values(Tensor<unsigned char, 1>& input_image,
                                        Index& x_top_left, Index& y_top_left,
                                        Index& x_bottom_right, Index& y_bottom_right)
{
    Tensor<type, 1> ground_truth_image;
    return ground_truth_image;
}


Tensor<type, 1> resize_image(Tensor<type, 1>& input_image)
{
    Tensor<type, 1> output_image;
    return output_image;
}


Tensor<type, 1> resize_proposed_region(const Tensor<type, 1> region_data,
                                       const Index& channels_number,
                                       const Index& region_width,
                                       const Index& region_height,
                                       const Index& new_width,
                                       const Index& new_height)
{
    Tensor<type, 1> new_resized_region_data(channels_number * new_width * new_height);

    const type scaleWidth =  (type)new_width / (type)region_width;
    const type scaleHeight = (type)new_height / (type)region_height;

    for(Index i = 0; i < new_height; i++)
    {
        for(Index j = 0; j < new_width; j++)
        {
            const int pixel = (i * (new_width * channels_number)) + (j * channels_number);
            const int nearest_match =  (((int)(i / scaleHeight) * (region_width * channels_number)) + ((int)(j / scaleWidth) * channels_number));

            if(channels_number == 3)
            {
                new_resized_region_data[pixel] =  region_data[nearest_match];
                new_resized_region_data[pixel + 1] =  region_data[nearest_match + 1];
                new_resized_region_data[pixel + 2] =  region_data[nearest_match + 2];
            }
            else
            {
                new_resized_region_data[pixel] =  region_data[nearest_match];
            }
        }
    }

    return new_resized_region_data;
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

        throw invalid_argument(buffer.str());
    }
    has_columns_names = true;
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

    Tensor<type, 2> imageDataAux(imageData.dimension(0), imageData.dimension(1));
    imageDataAux = imageData;

    if(classes_number == 2)
    {
        Index binary_columns_number = 1;
        data.resize(images_number, image_size + binary_columns_number);
        imageDataAux.resize(images_number, image_size + binary_columns_number);
    }
    else
    {
        data.resize(images_number, image_size + classes_number);
        imageDataAux.resize(images_number, image_size + classes_number);
    }

    memcpy(data.data(), imageData.data(), images_number * image_size * sizeof(type));

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
                data(row_index, image_size) = 0;
            }
            else
            {
                data(row_index, image_size + i) = 1;
            }

            rows_labels(row_index) = images_paths[j];

            row_index++;
        }
    }

    columns.resize(image_size + 1);

    // Input columns

    Index column_index = 0;

        for(Index i = 0; i < channels; i++)
        {
            for(Index j = 0; j < width; j++)
            {
                for(Index k = 0; k < height ; k++)
                {
                    columns(column_index).name= "pixel_" + to_string(i+1)+ "_" + to_string(j+1) + "_" + to_string(k+1);
                    columns(column_index).type = ColumnType::Numeric;
                    columns(column_index).column_use = VariableUse::Input;
                    columns(column_index).scaler = Scaler::MinimumMaximum;
                    column_index++;
                }
            }
        }

    // Target columns

    columns(image_size).name = "class";

    if(classes_number == 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Invalid number of categories. The minimum is 2 and you have 1.\n";

        throw invalid_argument(buffer.str());


    }

    Tensor<string, 1> categories(classes_number);

    for(Index i = 0 ; i < classes_number; i++)
    {
        categories(i) = folder_paths[i].filename().string();
    }

    columns(image_size).column_use = VariableUse::Target;
    columns(image_size).categories = categories;

    columns(image_size).categories_uses.resize(classes_number);
    columns(image_size).categories_uses.setConstant(VariableUse::Target);

    if(classes_number == 2)
    {
        columns(image_size).type = ColumnType::Binary;
    }
    else
    {
        columns(image_size).type = ColumnType::Categorical;
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
