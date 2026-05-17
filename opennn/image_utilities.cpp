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

namespace {

struct BmpHeader
{
    Index height = 0;
    Index width = 0;
    Index channels = 0;
    uint32_t bfOffBits = 0;
    uint16_t biBitCount = 0;
    int bytes_per_pixel = 0;
    long long row_stride = 0;
    bool top_down = false;
    bool is_grayscale = false;
    vector<RGBQuad> palette;
};

void read_bmp_file(const filesystem::path& path, vector<uint8_t>& buffer)
{
    const string path_str = path.string();

    ifstream file(path, ios::binary | ios::ate);
    if (!file)
        throw runtime_error("Cannot open BMP file: " + path_str);

    const streamsize size = file.tellg();
    if (size < 54)
        throw runtime_error("File too small to be a BMP: " + path_str);

    file.seekg(0, ios::beg);

    const size_t byte_count = static_cast<size_t>(size);
    if (buffer.capacity() < byte_count)
        buffer.reserve(byte_count);
    buffer.resize(byte_count);

    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw runtime_error("Error reading BMP file: " + path_str);
}

BmpHeader parse_bmp_header(const vector<uint8_t>& buffer, const string& path_str)
{
    auto read_u16 = [&](int offset) { return static_cast<uint16_t>(buffer[offset] | (buffer[offset+1] << 8)); };
    auto read_u32 = [&](int offset) { return static_cast<uint32_t>(buffer[offset] | (buffer[offset+1] << 8) | (buffer[offset+2] << 16) | (buffer[offset+3] << 24)); };
    auto read_s32 = [&](int offset) { return static_cast<int32_t>(read_u32(offset)); };

    if (read_u16(0) != 0x4D42)
        throw runtime_error("Not a BMP file (invalid signature 'BM'): " + path_str);

    BmpHeader h;
    h.bfOffBits = read_u32(10);
    const uint32_t biSize = read_u32(14);

    if (biSize != 40)
        throw runtime_error("Unsupported BMP DIB header size in file: " + path_str);

    const int32_t biWidth = read_s32(18);
    const int32_t biHeight_signed = read_s32(22);
    const uint16_t biPlanes = read_u16(26);
    h.biBitCount = read_u16(28);
    const uint32_t biCompression = read_u32(30);
    const uint32_t biClrUsed = read_u32(46);

    if (biWidth <= 0 || biHeight_signed == 0 || biPlanes != 1 || biCompression != 0)
        throw runtime_error("Invalid or unsupported BMP format in file: " + path_str);
    if (h.biBitCount != 8 && h.biBitCount != 24 && h.biBitCount != 32)
        throw runtime_error("Unsupported BMP bit count: " + to_string(h.biBitCount));

    h.is_grayscale = false;

    if (h.biBitCount == 8)
    {
        const uint32_t num_palette_colors = biClrUsed ? biClrUsed : 256;
        if (num_palette_colors > 256)
            throw runtime_error("Invalid palette size for 8-bit BMP.");

        h.palette.resize(num_palette_colors);
        h.is_grayscale = true;

        uint32_t pal_offset = 14 + biSize;
        for (uint32_t i = 0; i < num_palette_colors; ++i)
        {
            h.palette[i].blue     = buffer[pal_offset++];
            h.palette[i].green    = buffer[pal_offset++];
            h.palette[i].red      = buffer[pal_offset++];
            h.palette[i].reserved = buffer[pal_offset++];

            if (h.palette[i].red != h.palette[i].green || h.palette[i].red != h.palette[i].blue)
                h.is_grayscale = false;
        }
    }

    h.height = abs(biHeight_signed);
    h.width = biWidth;
    h.channels = (h.biBitCount == 8 && h.is_grayscale) ? 1 : 3;
    h.top_down = (biHeight_signed < 0);
    h.bytes_per_pixel = (h.biBitCount == 32) ? 4 : (h.biBitCount == 24) ? 3 : 1;
    h.row_stride = ((h.width * h.bytes_per_pixel + 3) / 4) * 4;

    if (h.bfOffBits + h.row_stride * h.height > Index(buffer.size()))
        throw runtime_error("Corrupted BMP: Pixel data exceeds file size.");

    return h;
}

void decode_bmp_pixels(const vector<uint8_t>& buffer, const BmpHeader& h, float* dst, bool divide_by_255)
{
    const uint8_t* pixel_data = buffer.data() + h.bfOffBits;
    const float scale = divide_by_255 ? (1.0f / 255.0f) : 1.0f;

    for (Index y = 0; y < h.height; ++y)
    {
        const Index tensor_y = h.top_down ? y : (h.height - 1 - y);
        const uint8_t* row_ptr = pixel_data + y * h.row_stride;
        const Index row_start_index = tensor_y * h.width * h.channels;

        if (h.biBitCount == 24 || h.biBitCount == 32)
        {
            for (Index x = 0; x < h.width; ++x)
            {
                const uint8_t* p = row_ptr + x * h.bytes_per_pixel;
                const Index pi = row_start_index + x * h.channels;
                dst[pi + 0] = float(p[2]) * scale;
                dst[pi + 1] = float(p[1]) * scale;
                dst[pi + 2] = float(p[0]) * scale;
            }
        }
        else if (h.biBitCount == 8)
        {
            if (h.channels == 1)
            {
                for (Index x = 0; x < h.width; ++x)
                    dst[row_start_index + x] = float(h.palette[row_ptr[x]].red) * scale;
            }
            else
            {
                for (Index x = 0; x < h.width; ++x)
                {
                    const RGBQuad& c = h.palette[row_ptr[x]];
                    const Index pi = row_start_index + x * 3;
                    dst[pi + 0] = float(c.red) * scale;
                    dst[pi + 1] = float(c.green) * scale;
                    dst[pi + 2] = float(c.blue) * scale;
                }
            }
        }
    }
}

}

Tensor3 load_image(const filesystem::path& path)
{
    thread_local vector<uint8_t> buffer;

    read_bmp_file(path, buffer);
    const BmpHeader h = parse_bmp_header(buffer, path.string());

    Tensor3 image(h.height, h.width, h.channels);
    decode_bmp_pixels(buffer, h, image.data(), false);

    return image;
}

void load_image(const filesystem::path& path,
                float* dst,
                Index expected_height,
                Index expected_width,
                Index expected_channels,
                bool divide_by_255)
{
    thread_local vector<uint8_t> buffer;

    read_bmp_file(path, buffer);
    const BmpHeader h = parse_bmp_header(buffer, path.string());

    if (h.channels != expected_channels)
        throw runtime_error("Channel mismatch in image: " + path.string());

    if (h.height == expected_height && h.width == expected_width)
    {
        decode_bmp_pixels(buffer, h, dst, divide_by_255);
        return;
    }

    Tensor3 temp(h.height, h.width, h.channels);
    decode_bmp_pixels(buffer, h, temp.data(), false);

    const Tensor3 resized = resize_image(temp, expected_height, expected_width);
    const Index pixels = expected_height * expected_width * expected_channels;

    if (divide_by_255)
        for (Index k = 0; k < pixels; ++k) dst[k] = resized.data()[k] / 255.0f;
    else
        copy_n(resized.data(), pixels, dst);
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

    for (Index x = 0; x < output_width; ++x)
    {
        const float in_x = x * scale_x;
        x0[x] = min<Index>(static_cast<Index>(in_x), input_width - 1);
        x1[x] = min<Index>(x0[x] + 1, input_width - 1);
        x_weight[x] = in_x - static_cast<float>(x0[x]);
    }

    #pragma omp parallel for collapse(2)
    for (Index y = 0; y < output_height; ++y)
        for (Index x = 0; x < output_width; ++x)
        {
            const float in_y = y * scale_y;
            const Index y0 = min<Index>(static_cast<Index>(in_y), input_height - 1);
            const Index y1 = min<Index>(y0 + 1, input_height - 1);
            const float y_weight = in_y - static_cast<float>(y0);

            const Index x0_value = x0[x];
            const Index x1_value = x1[x];
            const float x_weight_value = x_weight[x];

            for (Index c = 0; c < channels; ++c)
            {
                const float top =
                    (1.0f - x_weight_value) * input_image(y0, x0_value, c) +
                    x_weight_value             * input_image(y0, x1_value, c);

                const float bottom =
                    (1.0f - x_weight_value) * input_image(y1, x0_value, c) +
                    x_weight_value             * input_image(y1, x1_value, c);

                output_image(y, x, c) =
                    (1.0f - y_weight) * top + y_weight * bottom;
            }
        }

    return output_image;
}

void reflect_image_horizontal(Tensor3& image)
{
    image.device(get_device()) = image.reverse(array<bool, 3>({false, true, false}));
}

void reflect_image_vertical(Tensor3& image)
{
    image.device(get_device()) = image.reverse(array<bool, 3>({true, false, false}));
}

void rotate_image(const Tensor3& input, Tensor3& output, float angle_degree)
{
    const Index height = input.dimension(0);
    const Index width = input.dimension(1);
    const Index channels = input.dimension(2);

    const float center_x = float(width) / 2.0f;
    const float center_y = float(height) / 2.0f;

    const float angle_rad = -angle_degree * numbers::pi_v<float> / 180.0f;
    const float cos_angle = cos(angle_rad);
    const float sin_angle = sin(angle_rad);

    MatrixR rotation_matrix(3, 3);

    rotation_matrix << cos_angle, -sin_angle, center_x - cos_angle * center_x + sin_angle * center_y,
                       sin_angle, cos_angle, center_y - sin_angle * center_x - cos_angle * center_y,
                       0.0f, 0.0f, 1.0f;

    using Vector3T = Matrix<float, 3, 1>;

    #pragma omp parallel for collapse(2)

    for (Index y = 0; y < height; ++y)
    {
        for (Index x = 0; x < width; ++x)
        {
            Vector3T coordinates;
            coordinates << static_cast<float>(x), static_cast<float>(y), 1.0f;

            const Vector3T transformed = rotation_matrix * coordinates;

            if (transformed[0] >= 0 && transformed[0] < width
            && transformed[1] >= 0 && transformed[1] < height)
                for (Index c = 0; c < channels; ++c)
                    output(y, x, c) = input(int(transformed[1]),
                                            int(transformed[0]),
                                            c);
            else
                for (Index c = 0; c < channels; ++c)
                    output(y, x, c) = 0.0f;
        }
    }
}

template<int Dim>
void translate_image(const Tensor3& input, Tensor3& output, Index shift)
{
    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));
    assert(input.dimension(2) == output.dimension(2));

    output.setZero();

    const Index dim_size = input.dimension(Dim);

    if (abs(shift) >= dim_size) return;

    const Index src_start = (shift >= 0) ? 0 : -shift;
    const Index src_end = (shift >= 0) ? dim_size - shift : dim_size;

    for (Index i = src_start; i < src_end; ++i)
        output.template chip<Dim>(i + shift) = input.template chip<Dim>(i);
}

void translate_image_x(const Tensor3& input, Tensor3& output, Index shift)
{
    translate_image<1>(input, output, shift);
}

void translate_image_y(const Tensor3& input, Tensor3& output, Index shift)
{
    translate_image<0>(input, output, shift);
}

}
