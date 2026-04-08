//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_utilities.h"
#include "tensor_utilities.h"

namespace opennn
{

struct RGBQuad
{
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t reserved;
};

Tensor3 load_image(const filesystem::path& path)
{
    const string image_path_str = path.string();

    ifstream file(path, ios::binary | ios::ate);
    if(!file)
        throw runtime_error("Cannot open BMP file: " + image_path_str);

    streamsize size = file.tellg();
    if(size < 54) 
        throw runtime_error("File too small to be a BMP: " + image_path_str);

    file.seekg(0, ios::beg);

    thread_local vector<uint8_t> buffer;
    if (buffer.capacity() < size)
        buffer.reserve(size);
    
    buffer.resize(size);

    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw runtime_error("Error reading BMP file: " + image_path_str);
    file.close();

    auto read_u16 = [&](int offset) { return static_cast<uint16_t>(buffer[offset] | (buffer[offset+1] << 8)); };
    auto read_u32 = [&](int offset) { return static_cast<uint32_t>(buffer[offset] | (buffer[offset+1] << 8) | (buffer[offset+2] << 16) | (buffer[offset+3] << 24)); };
    auto read_s32 = [&](int offset) { return static_cast<int32_t>(read_u32(offset)); };

    if (read_u16(0) != 0x4D42) // 'BM'
        throw runtime_error("Not a BMP file (invalid signature 'BM'): " + image_path_str);

    const uint32_t bfOffBits = read_u32(10);
    const uint32_t biSize    = read_u32(14);

    if (biSize != 40)
        throw runtime_error("Unsupported BMP DIB header size in file: " + image_path_str);

    const int32_t  biWidth         = read_s32(18);
    const int32_t  biHeight_signed = read_s32(22);
    const uint16_t biPlanes        = read_u16(26);
    const uint16_t biBitCount      = read_u16(28);
    const uint32_t biCompression   = read_u32(30);
    const uint32_t biClrUsed       = read_u32(46);

    if (biWidth <= 0 || biHeight_signed == 0 || biPlanes != 1 || biCompression != 0)
        throw runtime_error("Invalid or unsupported BMP format in file: " + image_path_str);
    if (biBitCount != 8 && biBitCount != 24 && biBitCount != 32)
        throw runtime_error("Unsupported BMP bit count: " + to_string(biBitCount));

    vector<RGBQuad> palette;
    bool is_grayscale = false;

    if (biBitCount == 8)
    {
        const uint32_t num_palette_colors = biClrUsed ? biClrUsed : 256;
        if (num_palette_colors > 256)
            throw runtime_error("Invalid palette size for 8-bit BMP.");

        palette.resize(num_palette_colors);
        is_grayscale = true;
        
        uint32_t pal_offset = 14 + biSize; 
        for(uint32_t i = 0; i < num_palette_colors; ++i)
        {
            palette[i].blue     = buffer[pal_offset++];
            palette[i].green    = buffer[pal_offset++];
            palette[i].red      = buffer[pal_offset++];
            palette[i].reserved = buffer[pal_offset++];

            if (palette[i].red != palette[i].green || palette[i].red != palette[i].blue)
                is_grayscale = false;
        }
    }

    const Index height = abs(biHeight_signed);
    const Index width = biWidth;
    const Index channels = (biBitCount == 8 && is_grayscale) ? 1 : 3;
    const bool top_down = (biHeight_signed < 0);

    Tensor3 image(height, width, channels);

    const int bytes_per_pixel = (biBitCount == 32) ? 4 : (biBitCount == 24) ? 3 : 1;
    const long long row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;

    if (bfOffBits + row_stride * height > size)
        throw runtime_error("Corrupted BMP: Pixel data exceeds file size.");

    const uint8_t* pixel_data = buffer.data() + bfOffBits;
    
    auto* img_data = image.data();

    for(Index y = 0; y < height; y++)
    {
        const Index tensor_y = top_down ? y : (height - 1 - y);
        const uint8_t* row_ptr = pixel_data + y * row_stride;
        
        const Index row_start_idx = tensor_y * width * channels;

        if (biBitCount == 24 || biBitCount == 32)
        {
            for(Index x = 0; x < width; ++x)
            {
                const uint8_t* p = row_ptr + x * bytes_per_pixel;
                const Index pixel_idx = row_start_idx + x * channels;
                
                img_data[pixel_idx + 0] = static_cast<float>(p[2]); // R
                img_data[pixel_idx + 1] = static_cast<float>(p[1]); // G
                img_data[pixel_idx + 2] = static_cast<float>(p[0]); // B
            }
        }
        else if (biBitCount == 8)
        {
            if (channels == 1) // 8-bit (1 channel)
            {
                for(Index x = 0; x < width; ++x)
                {
                    img_data[row_start_idx + x] = static_cast<float>(palette[row_ptr[x]].red);
                }
            }
            else // 8-bit RGB (3 channels)
            {
                for(Index x = 0; x < width; ++x)
                {
                    const RGBQuad& color = palette[row_ptr[x]];
                    const Index pixel_idx = row_start_idx + x * 3;
                    
                    img_data[pixel_idx + 0] = static_cast<float>(color.red);
                    img_data[pixel_idx + 1] = static_cast<float>(color.green);
                    img_data[pixel_idx + 2] = static_cast<float>(color.blue);
                }
            }
        }
    }

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

    if (shift >= width) return;

    const Index channel_size = height * width;
    const Index remaining_width = width - shift;

    #pragma omp parallel for
    for (Index c = 0; c < channels; ++c)
    {
        const type* input_channel = input.data() + (c * channel_size);
        type* output_channel = output.data() + (c * channel_size) + (shift * height);

        MatrixMap output_block(output_channel, height, remaining_width);
        const MatrixMap input_block(const_cast<type*>(input_channel), height, remaining_width);

        output_block.noalias() = input_block;
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
