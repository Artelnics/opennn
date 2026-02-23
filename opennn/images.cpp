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
    if(!f) throw runtime_error("BMP read error (uint8_t) in file: " + file_path_str_for_error);
    return static_cast<uint8_t>(val_char);
}


uint16_t read_u16_le(ifstream& f, const string& file_path_str_for_error)
{
    const uint8_t b0 = read_u8(f, file_path_str_for_error);
    const uint8_t b1 = read_u8(f, file_path_str_for_error);
    return static_cast<uint16_t>(static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8));
}


uint32_t read_u32_le(ifstream& f, const string& file_path_str_for_error)
{
    const uint8_t b0 = read_u8(f, file_path_str_for_error);
    const uint8_t b1 = read_u8(f, file_path_str_for_error);
    const uint8_t b2 = read_u8(f, file_path_str_for_error);
    const uint8_t b3 = read_u8(f, file_path_str_for_error);
    return static_cast<uint32_t>(static_cast<uint32_t>(b0) |
                                 (static_cast<uint32_t>(b1) << 8) |
                                 (static_cast<uint32_t>(b2) << 16) |
                                 (static_cast<uint32_t>(b3) << 24));
}


int32_t read_s32_le(ifstream& f, const string& file_path_str_for_error)
{
    const uint8_t b0 = read_u8(f, file_path_str_for_error);
    const uint8_t b1 = read_u8(f, file_path_str_for_error);
    const uint8_t b2 = read_u8(f, file_path_str_for_error);
    const uint8_t b3 = read_u8(f, file_path_str_for_error);
    const uint32_t u_val = static_cast<uint32_t>(b0) |
                     (static_cast<uint32_t>(b1) << 8) |
                     (static_cast<uint32_t>(b2) << 16) |
                     (static_cast<uint32_t>(b3) << 24);
    return static_cast<int32_t>(u_val);
}


Tensor3 load_image(const filesystem::path& path)
{
    const string image_path_str = path.string();

    ifstream file(path, ios::binary);

    if(!file)
        throw runtime_error("Cannot open BMP file: " + image_path_str);

    const uint16_t bfType = read_u16_le(file, image_path_str);

    if (bfType != 0x4D42)
        throw runtime_error("Not a BMP file (invalid signature 'BM'): " + image_path_str);

    const uint32_t bfSize = read_u32_le(file, image_path_str);
    const uint16_t bfReserved1 = read_u16_le(file, image_path_str);
    const uint16_t bfReserved2 = read_u16_le(file, image_path_str);
    const uint32_t bfOffBits = read_u32_le(file, image_path_str);

    const uint32_t biSize = read_u32_le(file, image_path_str);

    if (biSize != 40)
        throw runtime_error("Unsupported BMP DIB header size: " + to_string(biSize) + " in file: " + image_path_str + ". Expected 40 (BITMAPINFOHEADER).");

    const int32_t biWidth = read_s32_le(file, image_path_str);
    const int32_t biHeight_signed = read_s32_le(file, image_path_str);
    const uint16_t biPlanes = read_u16_le(file, image_path_str);
    const uint16_t biBitCount = read_u16_le(file, image_path_str);
    const uint32_t biCompression = read_u32_le(file, image_path_str);
    const uint32_t biSizeImage = read_u32_le(file, image_path_str);
    const int32_t biXPelsPerMeter = read_s32_le(file, image_path_str);
    const int32_t biYPelsPerMeter = read_s32_le(file, image_path_str);
    const uint32_t biClrUsed = read_u32_le(file, image_path_str);
    const uint32_t biClrImportant = read_u32_le(file, image_path_str);

    if (biWidth <= 0)
        throw runtime_error("BMP width must be positive. Got: " + to_string(biWidth) + " in file: " + image_path_str);
    if (biHeight_signed == 0)
        throw runtime_error("BMP height cannot be zero in file: " + image_path_str);
    if (biPlanes != 1)
        throw runtime_error("BMP planes must be 1. Got: " + to_string(biPlanes) + " in file: " + image_path_str);
    if (biCompression != 0)
        throw runtime_error("Unsupported BMP compression type: " + to_string(biCompression) + ". Only uncompressed (0 = BI_RGB) is supported. File: " + image_path_str);
    if (biBitCount != 8 && biBitCount != 24 && biBitCount != 32)
        throw runtime_error("Unsupported BMP bit count: " + to_string(biBitCount) + " in file: " + image_path_str + ". Supported: 8, 24, 32.");

    vector<RGBQuad> palette;
    bool is_grayscale = false;

    if (biBitCount <= 8)
    {
        const uint32_t num_palette_colors = biClrUsed ? biClrUsed : (1u << biBitCount);

        if (num_palette_colors > 256 && biBitCount == 8)
            throw runtime_error("Invalid palette size for 8-bit BMP: " + to_string(num_palette_colors) + " in file: " + image_path_str);

        palette.resize(num_palette_colors);
        is_grayscale = true;

        for(uint32_t i = 0; i < num_palette_colors; ++i)
        {
            palette[i].blue = read_u8(file, image_path_str);
            palette[i].green = read_u8(file, image_path_str);
            palette[i].red = read_u8(file, image_path_str);
            palette[i].reserved = read_u8(file, image_path_str);

            if (palette[i].red != palette[i].green || palette[i].red != palette[i].blue)
                is_grayscale = false;
        }
    }

    const Index height = (biHeight_signed < 0) ? -biHeight_signed : biHeight_signed;
    const Index width = biWidth;
    const Index channels = (biBitCount == 8 && is_grayscale) ? 1 : 3;

    Tensor3 image(height, width, channels);

    const bool top_down = (biHeight_signed < 0);

    file.seekg(bfOffBits, ios::beg);

    if(!file)
        throw runtime_error("Failed to seek to pixel data offset (" + to_string(bfOffBits) + ") in BMP: " + image_path_str);

    const int bytes_per_pixel_in_file = (biBitCount == 32) ? 4 : (biBitCount == 24) ? 3 : 1;
    const long long row_data_bytes = width * bytes_per_pixel_in_file;
    const long long row_stride_in_file = ((row_data_bytes + 3) / 4) * 4;

    vector<unsigned char> row(row_stride_in_file);

    for(Index y_row = 0; y_row < height; y_row++)
    {
        const Index tensor_y_coord = top_down ? y_row : (height - 1 - y_row);

        file.read(reinterpret_cast<char*>(row.data()), row_stride_in_file);

        if (file.bad())
            throw runtime_error("Irrecoverable stream error while reading pixel row " + to_string(y_row) + " in BMP: " + image_path_str);

        streamsize bytes_read = file.gcount();

        if (bytes_read < row_data_bytes)
            throw runtime_error("Incomplete pixel data for row " + to_string(y_row) +
                                " (got " + to_string(bytes_read) + " bytes, expected at least " +
                                to_string(row_data_bytes) + " for pixel data) in BMP: " + image_path_str);

        if (file.eof() && y_row < height - 1)
            throw runtime_error("Unexpected EOF while reading pixel rows. Reached row " + to_string(y_row + 1) +
                                " of " + to_string(height) + " in BMP: " + image_path_str);

        for(Index x_col = 0; x_col < width; ++x_col)
        {
            float r = 0.0f, g = 0.0f, b = 0.0f;

            if (biBitCount == 32)
            {
                r = static_cast<float>(row[x_col * 4 + 2]);
                g = static_cast<float>(row[x_col * 4 + 1]);
                b = static_cast<float>(row[x_col * 4 + 0]);
            }
            else if (biBitCount == 24)
            {
                r = static_cast<float>(row[x_col * 3 + 2]);
                g = static_cast<float>(row[x_col * 3 + 1]);
                b = static_cast<float>(row[x_col * 3 + 0]);
            }
            else if (biBitCount == 8)
            {
                const unsigned char index = row[x_col];

                if (index >= palette.size())
                    throw runtime_error("Palette index " + to_string(index) + " out of bounds (palette size " + to_string(palette.size()) + ") in BMP: " + image_path_str);

                const RGBQuad& color = palette[index];

                r = static_cast<float>(color.red);
                g = static_cast<float>(color.green);
                b = static_cast<float>(color.blue);
            }

            if (channels == 1)
            {
                image(tensor_y_coord, x_col, 0) = r;
            }
            else
            {
                image(tensor_y_coord, x_col, 0) = r;
                image(tensor_y_coord, x_col, 1) = g;
                image(tensor_y_coord, x_col, 2) = b;
            }
        }
    }

    file.close();

    return image;
}


Tensor3 resize_image(const Tensor3& input_image,
                     Index output_height,
                     Index output_width)
{
    const Index input_height = input_image.dimension(0);
    const Index input_width = input_image.dimension(1);
    const Index channels = input_image.dimension(2);

    Tensor3 output_image(output_height, output_width, channels);

    const float scale_y = static_cast<float>(input_height) / output_height;
    const float scale_x = static_cast<float>(input_width) / output_width;

    vector<Index> x0(output_width), x1(output_width);
    vector<float> x_weight(output_width);

    for(Index x = 0; x < output_width; ++x)
    {
        const float in_x = x * scale_x;
        x0[x] = min<Index>(static_cast<Index>(in_x), input_width - 1); // in_x >= 0 => floor(in_x)
        x1[x] = min<Index>(x0[x] + 1, input_width - 1);
        x_weight[x] = in_x - static_cast<float>(x0[x]);
    }

    #pragma omp parallel for collapse(2)
    for(Index y = 0; y < output_height; ++y)
        for(Index x = 0; x < output_width; ++x)
        {
            const float in_y = y * scale_y;
            const Index y0 = min<Index>(static_cast<Index>(in_y), input_height - 1); // in_y >= 0 => floor(in_y)
            const Index y1 = min<Index>(y0 + 1, input_height - 1);
            const float y_weight = in_y - static_cast<float>(y0);

            const Index x0_value = x0[x];
            const Index x1_value = x1[x];
            const float x_weight_value = x_weight[x];

            for(Index c = 0; c < channels; ++c)
            {
                const float top =
                    (1.0f - x_weight_value) * input_image(y0, x0_value, c) +
                    x_weight_value        * input_image(y0, x1_value, c);

                const float bottom =
                    (1.0f - x_weight_value) * input_image(y1, x0_value, c) +
                    x_weight_value        * input_image(y1, x1_value, c);

                output_image(y, x, c) =
                    (1.0f - y_weight) * top + y_weight * bottom;
            }
        }

    return output_image;
}


void reflect_image_x(Tensor3& image)
{
    image.device(get_device()) = image.reverse(array<bool, 3>({false, true, false}));
}


void reflect_image_y(Tensor3& image)
{
    image.device(get_device()) = image.reverse(array<bool, 3>({true, false, false}));
}


void rotate_image(const Tensor3& input, Tensor3& output, type angle_degree)
{
    const Index width = input.dimension(0);
    const Index height = input.dimension(1);
    const Index channels = input.dimension(2);

    const type rotation_center_x = type(width) / type(2);
    const type rotation_center_y = type(height) / type(2);

    const type angle_rad = -angle_degree * type(3.1415927) / type(180.0);
    const type cos_angle = cos(angle_rad);
    const type sin_angle = sin(angle_rad);

    MatrixR rotation_matrix(3, 3);

    rotation_matrix << cos_angle, -sin_angle, rotation_center_x - cos_angle * rotation_center_x + sin_angle * rotation_center_y,
                       sin_angle, cos_angle, rotation_center_y - sin_angle * rotation_center_x - cos_angle * rotation_center_y,
                       type(0), type(0), type(1);


    using Vector3T = Matrix<type, 3, 1>;

    #pragma omp parallel for collapse(2)

    for(Index x = 0; x < width; x++)
    {
        for(Index y = 0; y < height; y++)
        {
            Vector3T coordinates;
            coordinates << static_cast<type>(x), static_cast<type>(y), 1.0f;

            const Vector3T transformed = rotation_matrix * coordinates;

            if(transformed[0] >= 0 && transformed[0] < width
            && transformed[1] >= 0 && transformed[1] < height)
                for(Index channel = 0; channel < channels; channel++)
                    output(x, y, channel) = input(int(transformed[0]),
                                                  int(transformed[1]),
                                                  channel);
            else
                for(Index channel = 0; channel < channels; channel++)
                    output(x, y, channel) = type(0);
        }
    }
}


void translate_image_x(const Tensor3& input, Tensor3& output, Index shift)
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
        const Index variable = i / channels;

        const TensorMap<const Tensor2> input_column_map(input.data() + variable*height + channel*input_size,
                                                                height,
                                                                1);

        MatrixMap output_column_map(output.data() + (variable + shift)*height + channel*input_size,
                                                     height,
                                                     1);

        output_column_map = input_column_map;
    }
}


void translate_image_y(const Tensor3& input, Tensor3& output, Index shift)
{
    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    output.setZero();

    const Index height = input.dimension(0);

    const Index limit_src_rows = height - shift;

    if (limit_src_rows <= 0)
        return;

    for(Index r_src = 0; r_src < limit_src_rows; ++r_src)
    {
        const Index r_dest = r_src + shift;

        output.template chip<0>(r_dest) = input.template chip<0>(r_src);
    }
}

} 
