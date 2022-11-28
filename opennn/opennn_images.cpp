#include "opennn_images.h"

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

            Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, cols_number, padding);

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

    Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>& img, const int& rows_number,const int& cols_number, const int& padding)
    {
        Tensor<unsigned char, 1> data_without_padding(img.size() - padding*rows_number);

        const int channels = 3;

        if (rows_number % 4 ==0)
        {
            memcpy(data_without_padding.data(), img.data(), static_cast<size_t>(cols_number*channels*rows_number)*sizeof(unsigned char));
        }
        else
        {
            for (int i = 0; i<rows_number; i++)
            {
                if(i==0)
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

    Tensor<type, 1> propose_single_random_region(const Tensor<Tensor<type, 1>, 1>& image_data, const Index width_to_resize, const Index height_to_resize)
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

        Tensor<type, 1> random_region(channels_number * region_width * region_height);

        // We resize the region after its random proposal

        random_region = get_bounding_box(image_data, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

        Tensor<type, 1> resized_random_region(channels_number * width_to_resize * height_to_resize);

        resized_random_region = resize_proposed_region(random_region, channels_number, region_width,
                                                       region_height, width_to_resize, height_to_resize);

        return resized_random_region;
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

    Tensor<type, 1> resize_proposed_region(const Tensor<type, 1> region_data, const Index& channels_number,
                                           const Index& region_width, const Index& region_height,
                                           const Index& new_width, const Index& new_height)
    {
        Tensor<type, 1> new_resized_region_data(channels_number * new_width * new_height);

        const type scaleWidth =  (type)new_width / (type)region_width;
        const type scaleHeight = (type)new_height / (type)region_height;

        for(Index i = 0; i < new_height; i++)
        {
            for(Index j = 0; j < new_width; j++)
            {
                const int pixel = (i * (new_width * channels_number)) + (j * channels_number);
                const int nearestMatch =  (((int)(i / scaleHeight) * (region_width * channels_number)) + ((int)(j / scaleWidth) * channels_number));

                if(channels_number == 3)
                {
                    new_resized_region_data[pixel] =  region_data[nearestMatch];
                    new_resized_region_data[pixel + 1] =  region_data[nearestMatch + 1];
                    new_resized_region_data[pixel + 2] =  region_data[nearestMatch + 2];
                }
                else
                {
                    new_resized_region_data[pixel] =  region_data[nearestMatch];
                }
            }
        }

        return new_resized_region_data;
    }
}
