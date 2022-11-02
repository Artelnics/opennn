#include "opennn_images.h"

namespace opennn
{
    Tensor<Tensor<type, 1>, 1> read_bmp_image(const string& filename)
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

/// todo
///
//        Tensor<int, 1> image_dimensions({static_cast<int>(image_height),
//                                         static_cast<int>(image_width),
//                                         static_cast<int>(channels_number)});

        image.setZero();

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

//            image_data(0) = image;
//            image_data(1) = image_dimensions;
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

    /*
    Tensor<type, 1> get_bounding_box(const Tensor<unsigned char, 1>& image,
                                              const Index& x_top_left, const Index& y_top_left,
                                              const Index& x_bottom_right, const Index& y_bottom_right)
    {
        const Index channels_number = get_channels_number();
        const Index height = get_image_height();
        const Index width = get_image_width();
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
            Tensor<unsigned char, 1> image_red_channel_flatted_sorted(image_size_single_channel);
            Tensor<unsigned char, 1> image_green_channel_flatted_sorted(image_size_single_channel);
            Tensor<unsigned char, 1> image_blue_channel_flatted_sorted(image_size_single_channel);

            image_red_channel_flatted_sorted = image.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));
            image_green_channel_flatted_sorted = image.slice(Eigen::array<Eigen::Index, 1>({image_size_single_channel}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));
            image_blue_channel_flatted_sorted = image.slice(Eigen::array<Eigen::Index, 1>({2 * image_size_single_channel}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));

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
                    bounding_box_red_channel(data_index) = static_cast<type>(image_red_channel_flatted_sorted[i]);
                    bounding_box_green_channel(data_index) = static_cast<type>(image_green_channel_flatted_sorted[i]);
                    bounding_box_blue_channel(data_index) = static_cast<type>(image_blue_channel_flatted_sorted[i]);

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
                    bounding_box_data(data_index) = static_cast<type>(image[i]);
                    data_index++;
                }
            }
        }

        return bounding_box_data;
    }
    */
}
