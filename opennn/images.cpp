//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "images.h"
#include "tensors.h"

namespace opennn
{

    
struct RGBQuad
{
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t reserved;
};


uint8_t read_u8(ifstream& f, const string& file_path_str_for_error) 
{
        char val_char;
        f.read(&val_char, 1);
        if (!f) throw runtime_error("BMP read error (uint8_t) in file: " + file_path_str_for_error);
        return static_cast<uint8_t>(val_char);
}


uint16_t read_u16_le(ifstream& f, const string& file_path_str_for_error) 
{
        uint8_t b0 = read_u8(f, file_path_str_for_error);
        uint8_t b1 = read_u8(f, file_path_str_for_error);
        return static_cast<uint16_t>(static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8));
}


uint32_t read_u32_le(ifstream& f, const string& file_path_str_for_error) 
{
        uint8_t b0 = read_u8(f, file_path_str_for_error);
        uint8_t b1 = read_u8(f, file_path_str_for_error);
        uint8_t b2 = read_u8(f, file_path_str_for_error);
        uint8_t b3 = read_u8(f, file_path_str_for_error);
        return static_cast<uint32_t>(static_cast<uint32_t>(b0) |
                                    (static_cast<uint32_t>(b1) << 8) |
                                    (static_cast<uint32_t>(b2) << 16) |
                                    (static_cast<uint32_t>(b3) << 24));
}


int32_t read_s32_le(ifstream& f, const string& file_path_str_for_error) 
{
        uint8_t b0 = read_u8(f, file_path_str_for_error);
        uint8_t b1 = read_u8(f, file_path_str_for_error);
        uint8_t b2 = read_u8(f, file_path_str_for_error);
        uint8_t b3 = read_u8(f, file_path_str_for_error);
        uint32_t u_val = static_cast<uint32_t>(b0) |
                        (static_cast<uint32_t>(b1) << 8) |
                        (static_cast<uint32_t>(b2) << 16) |
                        (static_cast<uint32_t>(b3) << 24);
        return static_cast<int32_t>(u_val);
}


Tensor<type, 3> read_bmp_image(const filesystem::path& image_path_fs) 
{
    string image_path_str = image_path_fs.string();

    ifstream file(image_path_fs, ios::binary);

    if (!file)
        throw runtime_error("Cannot open BMP file: " + image_path_str);

    uint16_t bfType = read_u16_le(file, image_path_str);
    if (bfType != 0x4D42)
        throw runtime_error("Not a BMP file (invalid signature 'BM'): " + image_path_str);

    uint32_t bfSize = read_u32_le(file, image_path_str);    // unused
    uint16_t bfReserved1 = read_u16_le(file, image_path_str); // unused
    uint16_t bfReserved2 = read_u16_le(file, image_path_str); // unused
    uint32_t bfOffBits = read_u32_le(file, image_path_str);

    uint32_t biSize = read_u32_le(file, image_path_str);

    if (biSize != 40)
        throw runtime_error("Unsupported BMP DIB header size: " + to_string(biSize) + " in file: " + image_path_str + ". Expected 40 (BITMAPINFOHEADER).");

    int32_t biWidth = read_s32_le(file, image_path_str);
    int32_t biHeight_signed = read_s32_le(file, image_path_str);
    uint16_t biPlanes = read_u16_le(file, image_path_str);
    uint16_t biBitCount = read_u16_le(file, image_path_str);
    uint32_t biCompression = read_u32_le(file, image_path_str);
    uint32_t biSizeImage = read_u32_le(file, image_path_str); // unused
    int32_t biXPelsPerMeter = read_s32_le(file, image_path_str); // unused
    int32_t biYPelsPerMeter = read_s32_le(file, image_path_str); // unused
    uint32_t biClrUsed = read_u32_le(file, image_path_str);
    uint32_t biClrImportant = read_u32_le(file, image_path_str); // unused

    if (biWidth <= 0)
        throw runtime_error("BMP width must be positive. Got: " + to_string(biWidth) + " in file: " + image_path_str);
    if (biHeight_signed == 0) 
        throw runtime_error("BMP height cannot be zero in file: " + image_path_str);
    if (biPlanes != 1) 
        throw runtime_error("BMP planes must be 1. Got: " + to_string(biPlanes) + " in file: " + image_path_str);
    if (biCompression != 0)
        throw runtime_error("Unsupported BMP compression type: " + to_string(biCompression) + ". Only uncompressed (0 = BI_RGB) is supported. File: " + image_path_str);

    switch (biBitCount) 
    {
    case 8:
    case 24:
    case 32:
        break;
    default:
        throw runtime_error("Unsupported BMP bit count: " + to_string(biBitCount) + " in file: " + image_path_str + ". Supported: 8, 24, 32.");
    }

    Index tensor_height = (biHeight_signed < 0) ? -biHeight_signed : biHeight_signed;
    Index tensor_width = biWidth;
    bool top_down = (biHeight_signed < 0);

    const Index tensor_channels = 3;
    Tensor<float, 3> image_tensor(tensor_height, tensor_width, tensor_channels);

    vector<RGBQuad> palette;

    if (biBitCount <= 8) 
    {
        uint32_t num_palette_colors = biClrUsed;
        if (num_palette_colors == 0)
            num_palette_colors = 1 << biBitCount;
        if (num_palette_colors > 256 && biBitCount == 8)
            throw runtime_error("Invalid palette size for 8-bit BMP: " + to_string(num_palette_colors) + " in file: " + image_path_str);

        palette.resize(num_palette_colors);

        for (uint32_t i = 0; i < num_palette_colors; ++i) 
        {
            palette[i].blue = read_u8(file, image_path_str);
            palette[i].green = read_u8(file, image_path_str);
            palette[i].red = read_u8(file, image_path_str);
            palette[i].reserved = read_u8(file, image_path_str);
        }
    }

    file.seekg(bfOffBits, ios::beg);
    if (!file)
        throw runtime_error("Failed to seek to pixel data offset (" + to_string(bfOffBits) + ") in BMP: " + image_path_str);

    int bytes_per_pixel_in_file;
    if (biBitCount == 32) bytes_per_pixel_in_file = 4;
    else if (biBitCount == 24) bytes_per_pixel_in_file = 3;
    else if (biBitCount == 8) bytes_per_pixel_in_file = 1;
    else
        throw logic_error("Internal error: Unhandled biBitCount in pixel reading stage.");

    long long row_data_bytes = tensor_width * bytes_per_pixel_in_file;
    long long row_stride_in_file = ((row_data_bytes + 3) / 4) * 4;
    vector<unsigned char> row_buffer(row_stride_in_file);

    for (Index y_row = 0; y_row < tensor_height; ++y_row) 
    {
        Index tensor_y_coord = top_down ? y_row : (tensor_height - 1 - y_row);

        file.read(reinterpret_cast<char*>(row_buffer.data()), row_stride_in_file);

        if (file.bad())
            throw runtime_error("Irrecoverable stream error while reading pixel row " + to_string(y_row) + " in BMP: " + image_path_str);

        streamsize bytes_read = file.gcount();

        if (bytes_read < row_data_bytes)
            throw runtime_error("Incomplete pixel data for row " + to_string(y_row) +
                " (got " + to_string(bytes_read) + " bytes, expected at least " +
                to_string(row_data_bytes) + " for pixel data) in BMP: " + image_path_str);

        if (file.eof() && y_row < tensor_height - 1)
            throw runtime_error("Unexpected EOF while reading pixel rows. Reached row " + to_string(y_row + 1) +
                " of " + to_string(tensor_height) + " in BMP: " + image_path_str);

        for (Index x_col = 0; x_col < tensor_width; ++x_col) 
        {
            float r_val = 0.0f, g_val = 0.0f, b_val = 0.0f;

            if (biBitCount == 32) 
            {
                unsigned char b = row_buffer[x_col * 4 + 0];
                unsigned char g = row_buffer[x_col * 4 + 1];
                unsigned char r = row_buffer[x_col * 4 + 2];

                r_val = static_cast<float>(r);
                g_val = static_cast<float>(g);
                b_val = static_cast<float>(b);
            }
            else if (biBitCount == 24) 
            {
                unsigned char b = row_buffer[x_col * 3 + 0];
                unsigned char g = row_buffer[x_col * 3 + 1];
                unsigned char r = row_buffer[x_col * 3 + 2];
                r_val = static_cast<float>(r);
                g_val = static_cast<float>(g);
                b_val = static_cast<float>(b);
            }
            else if (biBitCount == 8) 
            {
                unsigned char index = row_buffer[x_col];

                if (index >= palette.size())
                    throw runtime_error("Palette index " + to_string(index) + " out of bounds (palette size " + to_string(palette.size()) + ") in BMP: " + image_path_str);

                const RGBQuad& color = palette[index];
                r_val = static_cast<float>(color.red);
                g_val = static_cast<float>(color.green);
                b_val = static_cast<float>(color.blue);
            }

            image_tensor(tensor_y_coord, x_col, 0) = r_val;
            image_tensor(tensor_y_coord, x_col, 1) = g_val;
            image_tensor(tensor_y_coord, x_col, 2) = b_val;
        }
    }

    file.close();

    return image_tensor;
}


Tensor<type, 3> resize_image(const Tensor<type, 3>& input_image,
                             const Index& output_height,
                             const Index& output_width)
{
    const Index input_height = input_image.dimension(0);
    const Index input_width = input_image.dimension(1);
    const Index channels = input_image.dimension(2);

    Tensor<type, 3> output_image(output_height, output_width, channels);

    const type scale_y = static_cast<float>(input_height) / output_height;
    const type scale_x = static_cast<float>(input_width) / output_width;

    #pragma omp parallel for collapse(2)

    for (Index c = 0; c < channels; ++c)
    {
        for (Index y = 0; y < output_height; ++y)
        {
            const type in_y = y * scale_y;
            const Index y0 = static_cast<Index>(in_y);
            const type y_weight = in_y - y0;
            const Index y1 = min(y0 + 1, input_height - 1);

            for (Index x = 0; x < output_width; ++x)
            {
                const type in_x = x * scale_x;
                const Index x0 = static_cast<Index>(in_x);
                const type x_weight = in_x - x0;
                const Index x1 = min(x0 + 1, input_width - 1);

                output_image(y, x, c) = (1 - y_weight) * ((1 - x_weight) * input_image(y0, x0, c) + x_weight * input_image(y0, x1, c)) +
                    y_weight * ((1 - x_weight) * input_image(y1, x0, c) + x_weight * input_image(y1, x1, c));
            }
        }
    }

    return output_image;
}


void reflect_image_x(const ThreadPoolDevice* thread_pool_device,
                     Tensor<type, 3>& image)
{
    image/*.device(thread_pool_device)*/ = image.reverse(array<bool, 3>({false, true, false}));
}


void reflect_image_y(const ThreadPoolDevice* thread_pool_device,
                     Tensor<type, 3>& image)
{
    image/*.device(thread_pool_device)*/ = image.reverse(array<bool, 3>({true, false, false}));
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

    rotation_matrix.setValues({
        {cos_angle, -sin_angle, rotation_center_x - cos_angle * rotation_center_x + sin_angle * rotation_center_y},
        {sin_angle, cos_angle, rotation_center_y - sin_angle * rotation_center_x - cos_angle * rotation_center_y},
        {type(0), type(0), type(1)}});

    Tensor<type, 1> coordinates(3);
    Tensor<type, 1> transformed_coordinates(3);

    for(Index x = 0; x < width; x++)
    {
        for(Index y = 0; y < height; y++)
        {
            coordinates(0) = type(x);
            coordinates(1) = type(y);
            coordinates(2) = type(1);

            transformed_coordinates = rotation_matrix.contract(coordinates, axes(1,0));

            if(transformed_coordinates[0] >= 0 && transformed_coordinates[0] < width
            && transformed_coordinates[1] >= 0 && transformed_coordinates[1] < height)
            {
                for(Index channel = 0; channel < channels; channel++)
                    output(x, y, channel) = input(int(transformed_coordinates[0]),
                                                  int(transformed_coordinates[1]),
                                                  channel);
            }
            else
            {
                for(Index channel = 0; channel < channels; channel++)
                    output(x, y, channel) = type(0);
            }
        }
    }
}


void translate_image_x(const ThreadPoolDevice* thread_pool_device,
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


// @todo

void translate_image_y(const ThreadPoolDevice* thread_pool_device,
                       const Tensor<type, 3>& input,
                       Tensor<type, 3>& output,
                       const Index& shift)
{
}


void rescale_image(const ThreadPoolDevice*, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const type&)
{
}

} // namespace opennn
