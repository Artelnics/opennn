//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_utilities.h"
#include "tensor_utilities.h"

#include <cctype>
#include <csetjmp>
#include <cstdio>
#include <zlib.h>
extern "C" {
#include <jpeglib.h>
}

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

struct PngHeader
{
    Index height = 0;
    Index width = 0;
    Index channels = 0;
    uint8_t bit_depth = 0;
    uint8_t color_type = 0;
    int bytes_per_pixel = 0;
    vector<uint8_t> palette;
};

void read_image_file(const filesystem::path& path, vector<uint8_t>& buffer)
{
    const string path_str = path.string();

    ifstream file(path, ios::binary | ios::ate);
    if (!file)
        throw runtime_error(format("Cannot open image file: {}", path_str));

    const streamsize size = file.tellg();
    if (size < 8)
        throw runtime_error(format("File too small to be an image: {}", path_str));

    file.seekg(0, ios::beg);

    const size_t byte_count = static_cast<size_t>(size);
    if (buffer.capacity() < byte_count)
        buffer.reserve(byte_count);
    buffer.resize(byte_count);

    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw runtime_error(format("Error reading image file: {}", path_str));
}

bool has_png_signature(const vector<uint8_t>& buffer)
{
    static constexpr uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    return buffer.size() >= 8 && equal(begin(signature), end(signature), buffer.begin());
}

bool has_bmp_signature(const vector<uint8_t>& buffer)
{
    return buffer.size() >= 2 && buffer[0] == 'B' && buffer[1] == 'M';
}

uint32_t read_be32(const uint8_t* p)
{
    return (uint32_t(p[0]) << 24)
         | (uint32_t(p[1]) << 16)
         | (uint32_t(p[2]) << 8)
         |  uint32_t(p[3]);
}

BmpHeader parse_bmp_header(const vector<uint8_t>& buffer, const string& path_str)
{
    if (buffer.size() < 54)
        throw runtime_error(format("File too small to be a BMP: {}", path_str));

    auto read_u16 = [&](int offset) { return static_cast<uint16_t>(buffer[offset] | (buffer[offset+1] << 8)); };
    auto read_u32 = [&](int offset) { return static_cast<uint32_t>(buffer[offset] | (buffer[offset+1] << 8) | (buffer[offset+2] << 16) | (buffer[offset+3] << 24)); };
    auto read_s32 = [&](int offset) { return static_cast<int32_t>(read_u32(offset)); };

    if (read_u16(0) != 0x4D42)
        throw runtime_error(format("Not a BMP file (invalid signature 'BM'): {}", path_str));

    BmpHeader h;
    h.bfOffBits = read_u32(10);
    const uint32_t biSize = read_u32(14);

    if (biSize != 40)
        throw runtime_error(format("Unsupported BMP DIB header size in file: {}", path_str));

    const int32_t biWidth = read_s32(18);
    const int32_t biHeight_signed = read_s32(22);
    const uint16_t biPlanes = read_u16(26);
    h.biBitCount = read_u16(28);
    const uint32_t biCompression = read_u32(30);
    const uint32_t biClrUsed = read_u32(46);

    if (biWidth <= 0 || biHeight_signed == 0 || biPlanes != 1 || biCompression != 0)
        throw runtime_error(format("Invalid or unsupported BMP format in file: {}", path_str));
    if (h.biBitCount != 8 && h.biBitCount != 24 && h.biBitCount != 32)
        throw runtime_error(format("Unsupported BMP bit count: {}", h.biBitCount));

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

uint8_t paeth_predictor(uint8_t a, uint8_t b, uint8_t c)
{
    const int p = int(a) + int(b) - int(c);
    const int pa = abs(p - int(a));
    const int pb = abs(p - int(b));
    const int pc = abs(p - int(c));

    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

PngHeader parse_png_chunks(const vector<uint8_t>& buffer,
                           vector<uint8_t>& compressed,
                           const string& path_str)
{
    if (!has_png_signature(buffer))
        throw runtime_error(format("Not a PNG file: {}", path_str));

    // Reset destination so callers don't have to remember the precondition.
    // Capacity is preserved, so the thread_local buffer doesn't re-allocate.
    compressed.clear();

    PngHeader h;
    bool saw_ihdr = false;
    size_t pos = 8;

    while (pos + 12 <= buffer.size())
    {
        const uint32_t length = read_be32(buffer.data() + pos);
        pos += 4;

        if (pos + 4 + size_t(length) + 4 > buffer.size())
            throw runtime_error(format("Corrupted PNG chunk in file: {}", path_str));

        const string_view type(reinterpret_cast<const char*>(buffer.data() + pos), 4);
        pos += 4;
        const uint8_t* data = buffer.data() + pos;

        if (type == "IHDR")
        {
            if (length != 13)
                throw runtime_error(format("Invalid PNG IHDR in file: {}", path_str));

            h.width = Index(read_be32(data));
            h.height = Index(read_be32(data + 4));
            h.bit_depth = data[8];
            h.color_type = data[9];
            const uint8_t compression = data[10];
            const uint8_t filter = data[11];
            const uint8_t interlace = data[12];

            if (h.width <= 0 || h.height <= 0 || h.bit_depth != 8
             || compression != 0 || filter != 0 || interlace != 0)
                throw runtime_error(format("Unsupported PNG format in file: {}", path_str));

            switch (h.color_type)
            {
                case 0: h.channels = 1; h.bytes_per_pixel = 1; break;
                case 2: h.channels = 3; h.bytes_per_pixel = 3; break;
                case 3: h.channels = 3; h.bytes_per_pixel = 1; break;
                case 4: h.channels = 1; h.bytes_per_pixel = 2; break;
                case 6: h.channels = 3; h.bytes_per_pixel = 4; break;
                default:
                    throw runtime_error(format("Unsupported PNG color type {} in file: {}",
                                               h.color_type, path_str));
            }

            saw_ihdr = true;
        }
        else if (type == "PLTE")
        {
            h.palette.assign(data, data + length);
        }
        else if (type == "IDAT")
        {
            compressed.insert(compressed.end(), data, data + length);
        }
        else if (type == "IEND")
        {
            break;
        }

        pos += size_t(length) + 4;
    }

    if (!saw_ihdr || compressed.empty())
        throw runtime_error(format("Incomplete PNG file: {}", path_str));

    if (h.color_type == 3 && (h.palette.empty() || h.palette.size() % 3 != 0))
        throw runtime_error(format("PNG palette missing or invalid in file: {}", path_str));

    return h;
}

// Resizes `inflated` in place and uncompresses `compressed` into it. No new
// allocation if the thread_local buffer is already large enough.
void inflate_png_data_into(const vector<uint8_t>& compressed,
                            const PngHeader& h,
                            vector<uint8_t>& inflated,
                            const string& path_str)
{
    const size_t row_bytes = size_t(h.width) * size_t(h.bytes_per_pixel);
    const size_t expected_size = size_t(h.height) * (row_bytes + 1);

    inflated.resize(expected_size);
    uLongf actual_size = uLongf(expected_size);

    const int zlib_status = uncompress(inflated.data(), &actual_size,
                                       compressed.data(), uLong(compressed.size()));

    if (zlib_status != Z_OK || actual_size != expected_size)
        throw runtime_error(format("Cannot decompress PNG image: {}", path_str));
}

// Resizes `unfiltered` in place and applies PNG row filters from `inflated`.
// No new allocation if the thread_local buffer is already large enough.
void unfilter_png_rows_into(const vector<uint8_t>& inflated,
                             const PngHeader& h,
                             vector<uint8_t>& unfiltered,
                             const string& path_str)
{
    const Index row_bytes = h.width * h.bytes_per_pixel;
    unfiltered.resize(size_t(h.height) * size_t(row_bytes));

    size_t src = 0;
    for (Index y = 0; y < h.height; ++y)
    {
        const uint8_t filter_type = inflated[src++];
        uint8_t* row = unfiltered.data() + size_t(y) * size_t(row_bytes);
        const uint8_t* prev = y > 0 ? row - row_bytes : nullptr;

        for (Index x = 0; x < row_bytes; ++x)
        {
            const uint8_t raw = inflated[src++];
            const uint8_t left = x >= h.bytes_per_pixel ? row[x - h.bytes_per_pixel] : 0;
            const uint8_t up = prev ? prev[x] : 0;
            const uint8_t up_left = (prev && x >= h.bytes_per_pixel)
                                  ? prev[x - h.bytes_per_pixel]
                                  : 0;

            switch (filter_type)
            {
                case 0: row[x] = raw; break;
                case 1: row[x] = uint8_t(raw + left); break;
                case 2: row[x] = uint8_t(raw + up); break;
                case 3: row[x] = uint8_t(raw + uint8_t((uint16_t(left) + uint16_t(up)) / 2)); break;
                case 4: row[x] = uint8_t(raw + paeth_predictor(left, up, up_left)); break;
                default:
                    throw runtime_error(format("Unsupported PNG row filter in file: {}", path_str));
            }
        }
    }
}

// Orchestrates PNG decode: takes the already-parsed header + IDAT-concatenated
// `compressed`, inflates and unfilters into thread_local scratch, and writes
// the result as floats into `dst` (length h.height*h.width*h.channels).
// Symmetric with decode_bmp_pixels — no heap allocations on the steady state
// because inflated/unfiltered keep their capacity across calls.
void decode_png_pixels(const PngHeader& h,
                       const vector<uint8_t>& compressed,
                       float* dst,
                       bool divide_by_255,
                       const string& path_str)
{
    thread_local vector<uint8_t> inflated;
    thread_local vector<uint8_t> unfiltered;

    inflate_png_data_into(compressed, h, inflated, path_str);
    unfilter_png_rows_into(inflated, h, unfiltered, path_str);

    const float scale = divide_by_255 ? (1.0f / 255.0f) : 1.0f;

    for (Index y = 0; y < h.height; ++y)
    {
        const uint8_t* row = unfiltered.data() + size_t(y) * size_t(h.width) * size_t(h.bytes_per_pixel);

        for (Index x = 0; x < h.width; ++x)
        {
            const uint8_t* p = row + size_t(x) * size_t(h.bytes_per_pixel);
            const Index out = (y * h.width + x) * h.channels;

            switch (h.color_type)
            {
                case 0:
                    dst[out] = float(p[0]) * scale;
                    break;
                case 2:
                    dst[out + 0] = float(p[0]) * scale;
                    dst[out + 1] = float(p[1]) * scale;
                    dst[out + 2] = float(p[2]) * scale;
                    break;
                case 3:
                {
                    const size_t pal = size_t(p[0]) * 3;
                    if (pal + 2 >= h.palette.size())
                        throw runtime_error("PNG palette index out of range.");

                    dst[out + 0] = float(h.palette[pal + 0]) * scale;
                    dst[out + 1] = float(h.palette[pal + 1]) * scale;
                    dst[out + 2] = float(h.palette[pal + 2]) * scale;
                    break;
                }
                case 4:
                    dst[out] = float(p[0]) * scale;
                    break;
                case 6:
                    dst[out + 0] = float(p[0]) * scale;
                    dst[out + 1] = float(p[1]) * scale;
                    dst[out + 2] = float(p[2]) * scale;
                    break;
            }
        }
    }
}

struct JpegHeader
{
    Index height = 0;
    Index width = 0;
    Index channels = 0;
};

struct JpegErrorManager
{
    struct jpeg_error_mgr pub;
    jmp_buf jmp;
    char message[JMSG_LENGTH_MAX] = {0};
};

void jpeg_error_exit_throw(j_common_ptr cinfo)
{
    JpegErrorManager* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, err->message);
    longjmp(err->jmp, 1);
}

void jpeg_output_silent(j_common_ptr) {}

bool has_jpeg_signature(const vector<uint8_t>& buffer)
{
    return buffer.size() >= 3 && buffer[0] == 0xFF && buffer[1] == 0xD8 && buffer[2] == 0xFF;
}

// Decode JPEG bytes in `buffer` into `dst` as interleaved HWC.
// `dst` must hold height*width*channels floats (sized by caller from JpegHeader).
JpegHeader decode_jpeg_pixels(const vector<uint8_t>& buffer,
                              float* dst,
                              bool divide_by_255,
                              const string& path_for_error)
{
    jpeg_decompress_struct cinfo{};
    JpegErrorManager err{};
    cinfo.err = jpeg_std_error(&err.pub);
    err.pub.error_exit = jpeg_error_exit_throw;
    err.pub.output_message = jpeg_output_silent;

    JpegHeader header;
    string error_message;

    if (setjmp(err.jmp))
    {
        jpeg_destroy_decompress(&cinfo);
        throw runtime_error(format("JPEG decode failed for {}: {}", path_for_error, err.message));
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buffer.data(), buffer.size());

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
        throw runtime_error(format("JPEG: missing or corrupt header in {}", path_for_error));

    cinfo.out_color_space = (cinfo.num_components == 1) ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_start_decompress(&cinfo);

    header.height = cinfo.output_height;
    header.width = cinfo.output_width;
    header.channels = cinfo.output_components;

    const float scale = divide_by_255 ? (1.0f / 255.0f) : 1.0f;
    const size_t row_bytes = size_t(header.width) * size_t(header.channels);
    vector<uint8_t> row(row_bytes);
    JSAMPROW row_ptr = row.data();

    while (cinfo.output_scanline < cinfo.output_height)
    {
        const Index y = cinfo.output_scanline;
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
        float* dst_row = dst + y * row_bytes;
        for (size_t i = 0; i < row_bytes; ++i)
            dst_row[i] = float(row[i]) * scale;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return header;
}

}

bool is_supported_image_file(const filesystem::path& path)
{
    string extension = path.extension().string();
    ranges::transform(extension, extension.begin(),
                      [](unsigned char c) { return char(std::tolower(c)); });

    return extension == ".bmp" || extension == ".png"
        || extension == ".jpg" || extension == ".jpeg";
}

Tensor3 load_image(const filesystem::path& path)
{
    thread_local vector<uint8_t> buffer;

    read_image_file(path, buffer);

    if (has_bmp_signature(buffer))
    {
        const BmpHeader h = parse_bmp_header(buffer, path.string());

        Tensor3 image(h.height, h.width, h.channels);
        decode_bmp_pixels(buffer, h, image.data(), false);

        return image;
    }

    if (has_png_signature(buffer))
    {
        thread_local vector<uint8_t> compressed;
        const PngHeader h = parse_png_chunks(buffer, compressed, path.string());

        Tensor3 image(h.height, h.width, h.channels);
        decode_png_pixels(h, compressed, image.data(), false, path.string());

        return image;
    }

    if (has_jpeg_signature(buffer))
    {
        // Two-pass: peek header to size the tensor, then decode in place.
        jpeg_decompress_struct cinfo{};
        JpegErrorManager err{};
        cinfo.err = jpeg_std_error(&err.pub);
        err.pub.error_exit = jpeg_error_exit_throw;
        err.pub.output_message = jpeg_output_silent;

        Index height = 0, width = 0, channels = 0;
        if (setjmp(err.jmp))
        {
            jpeg_destroy_decompress(&cinfo);
            throw runtime_error(format("JPEG header read failed for {}: {}", path.string(), err.message));
        }
        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, buffer.data(), buffer.size());
        jpeg_read_header(&cinfo, TRUE);
        height = cinfo.image_height;
        width = cinfo.image_width;
        channels = (cinfo.num_components == 1) ? 1 : 3;
        jpeg_destroy_decompress(&cinfo);

        Tensor3 image(height, width, channels);
        decode_jpeg_pixels(buffer, image.data(), false, path.string());
        return image;
    }

    throw runtime_error(format("Unsupported image file: {}", path.string()));
}

void load_image(const filesystem::path& path,
                float* dst,
                Index expected_height,
                Index expected_width,
                Index expected_channels,
                bool divide_by_255)
{
    thread_local vector<uint8_t> buffer;

    read_image_file(path, buffer);

    Index height = 0;
    Index width = 0;
    Index channels = 0;

    if (has_bmp_signature(buffer))
    {
        const BmpHeader h = parse_bmp_header(buffer, path.string());
        height = h.height;
        width = h.width;
        channels = h.channels;

        if (channels != expected_channels)
            throw runtime_error(format("Channel mismatch in image: {}", path.string()));

        if (height == expected_height && width == expected_width)
        {
            decode_bmp_pixels(buffer, h, dst, divide_by_255);
            return;
        }

        Tensor3 temp(height, width, channels);
        decode_bmp_pixels(buffer, h, temp.data(), false);

        const Tensor3 resized = resize_image(temp, expected_height, expected_width);
        const Index pixels = expected_height * expected_width * expected_channels;

        if (divide_by_255)
            for (Index k = 0; k < pixels; ++k) dst[k] = resized.data()[k] / 255.0f;
        else
            copy_n(resized.data(), pixels, dst);

        return;
    }

    if (has_png_signature(buffer))
    {
        thread_local vector<uint8_t> compressed;
        const PngHeader h = parse_png_chunks(buffer, compressed, path.string());
        height = h.height;
        width = h.width;
        channels = h.channels;

        if (channels != expected_channels)
            throw runtime_error(format("Channel mismatch in image: {}", path.string()));

        if (height == expected_height && width == expected_width)
        {
            decode_png_pixels(h, compressed, dst, divide_by_255, path.string());
            return;
        }

        Tensor3 temp(height, width, channels);
        decode_png_pixels(h, compressed, temp.data(), false, path.string());

        const Tensor3 resized = resize_image(temp, expected_height, expected_width);
        const Index pixels = expected_height * expected_width * expected_channels;

        if (divide_by_255)
            for (Index k = 0; k < pixels; ++k) dst[k] = resized.data()[k] / 255.0f;
        else
            copy_n(resized.data(), pixels, dst);
        return;
    }

    if (!has_jpeg_signature(buffer))
        throw runtime_error(format("Unsupported image file: {}", path.string()));

    // JPEG: decode into a temp at native size, then resize/scale to match expectations.
    Index jh = 0, jw = 0, jc = 0;
    {
        jpeg_decompress_struct cinfo{};
        JpegErrorManager err{};
        cinfo.err = jpeg_std_error(&err.pub);
        err.pub.error_exit = jpeg_error_exit_throw;
        err.pub.output_message = jpeg_output_silent;
        if (setjmp(err.jmp))
        {
            jpeg_destroy_decompress(&cinfo);
            throw runtime_error(format("JPEG header read failed for {}: {}", path.string(), err.message));
        }
        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, buffer.data(), buffer.size());
        jpeg_read_header(&cinfo, TRUE);
        jh = cinfo.image_height;
        jw = cinfo.image_width;
        jc = (cinfo.num_components == 1) ? 1 : 3;
        jpeg_destroy_decompress(&cinfo);
    }

    if (jc != expected_channels)
        throw runtime_error(format("Channel mismatch in image: {}", path.string()));

    if (jh == expected_height && jw == expected_width)
    {
        decode_jpeg_pixels(buffer, dst, divide_by_255, path.string());
        return;
    }

    Tensor3 temp(jh, jw, jc);
    decode_jpeg_pixels(buffer, temp.data(), false, path.string());

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

void reflect_image_horizontal(TensorMap3& image)
{
    const Index height = image.dimension(0);
    const Index width = image.dimension(1);
    const Index channels = image.dimension(2);

    for (Index y = 0; y < height; ++y)
        for (Index x = 0; x < width / 2; ++x)
            for (Index c = 0; c < channels; ++c)
                swap(image(y, x, c), image(y, width - 1 - x, c));
}

void reflect_image_vertical(TensorMap3& image)
{
    const Index height = image.dimension(0);
    const Index width = image.dimension(1);
    const Index channels = image.dimension(2);
    const Index row_size = width * channels;

    float* data = image.data();

    for (Index y = 0; y < height / 2; ++y)
        swap_ranges(data + y * row_size,
                    data + (y + 1) * row_size,
                    data + (height - 1 - y) * row_size);
}

void rotate_image(const TensorMap3& input, TensorMap3& output, float angle_degree)
{
    const Index height = input.dimension(0);
    const Index width = input.dimension(1);
    const Index channels = input.dimension(2);
    const Index pixels = height * width * channels;

    assert(height == output.dimension(0));
    assert(width == output.dimension(1));
    assert(channels == output.dimension(2));

    if (angle_degree == 0.0f)
    {
        if (input.data() != output.data())
            copy_n(input.data(), pixels, output.data());
        return;
    }

    if (input.data() == output.data())
    {
        Tensor3 copy(height, width, channels);
        copy_n(input.data(), pixels, copy.data());
        TensorMap3 copy_map(copy.data(), height, width, channels);
        rotate_image(copy_map, output, angle_degree);
        return;
    }

    const float center_x = float(width) / 2.0f;
    const float center_y = float(height) / 2.0f;

    const float angle_rad = -angle_degree * numbers::pi_v<float> / 180.0f;
    const float cos_angle = cos(angle_rad);
    const float sin_angle = sin(angle_rad);

    for (Index y = 0; y < height; ++y)
    {
        for (Index x = 0; x < width; ++x)
        {
            const float src_x = cos_angle * (float(x) - center_x)
                              - sin_angle * (float(y) - center_y)
                              + center_x;
            const float src_y = sin_angle * (float(x) - center_x)
                              + cos_angle * (float(y) - center_y)
                              + center_y;

            if (src_x >= 0.0f && src_x < float(width)
            && src_y >= 0.0f && src_y < float(height))
                for (Index c = 0; c < channels; ++c)
                    output(y, x, c) = input(int(src_y), int(src_x), c);
            else
                for (Index c = 0; c < channels; ++c)
                    output(y, x, c) = 0.0f;
        }
    }
}

void translate_image_x(TensorMap3& image, Index shift)
{
    if (shift == 0) return;

    const Index height = image.dimension(0);
    const Index width = image.dimension(1);
    const Index channels = image.dimension(2);
    const Index row_size = width * channels;
    float* data = image.data();

    if (abs(shift) >= width)
    {
        fill(data, data + height * row_size, 0.0f);
        return;
    }

    const Index move_columns = width - abs(shift);
    const Index move_size = move_columns * channels;
    const Index fill_size = abs(shift) * channels;

    for (Index y = 0; y < height; ++y)
    {
        float* row = data + y * row_size;

        if (shift > 0)
        {
            memmove(row + fill_size, row, size_t(move_size) * sizeof(float));
            fill(row, row + fill_size, 0.0f);
        }
        else
        {
            memmove(row, row + fill_size, size_t(move_size) * sizeof(float));
            fill(row + move_size, row + row_size, 0.0f);
        }
    }
}

void translate_image_y(TensorMap3& image, Index shift)
{
    if (shift == 0) return;

    const Index height = image.dimension(0);
    const Index width = image.dimension(1);
    const Index channels = image.dimension(2);
    const Index row_size = width * channels;
    const Index pixels = height * row_size;
    float* data = image.data();

    if (abs(shift) >= height)
    {
        fill(data, data + pixels, 0.0f);
        return;
    }

    const Index move_rows = height - abs(shift);
    const Index move_size = move_rows * row_size;
    const Index fill_size = abs(shift) * row_size;

    if (shift > 0)
    {
        memmove(data + fill_size, data, size_t(move_size) * sizeof(float));
        fill(data, data + fill_size, 0.0f);
    }
    else
    {
        memmove(data, data + fill_size, size_t(move_size) * sizeof(float));
        fill(data + move_size, data + pixels, 0.0f);
    }
}

void reflect_image_horizontal(Tensor3& image)
{
    TensorMap3 image_map(image.data(), image.dimension(0), image.dimension(1), image.dimension(2));
    reflect_image_horizontal(image_map);
}

void reflect_image_vertical(Tensor3& image)
{
    TensorMap3 image_map(image.data(), image.dimension(0), image.dimension(1), image.dimension(2));
    reflect_image_vertical(image_map);
}

void rotate_image(const Tensor3& input, Tensor3& output, float angle_degree)
{
    TensorMap3 input_map(const_cast<float*>(input.data()), input.dimension(0), input.dimension(1), input.dimension(2));
    TensorMap3 output_map(output.data(), output.dimension(0), output.dimension(1), output.dimension(2));
    rotate_image(input_map, output_map, angle_degree);
}

void translate_image_x(const Tensor3& input, Tensor3& output, Index shift)
{
    if (input.data() != output.data())
        copy_n(input.data(), input.size(), output.data());

    TensorMap3 output_map(output.data(), output.dimension(0), output.dimension(1), output.dimension(2));
    translate_image_x(output_map, shift);
}

void translate_image_y(const Tensor3& input, Tensor3& output, Index shift)
{
    if (input.data() != output.data())
        copy_n(input.data(), input.size(), output.data());

    TensorMap3 output_map(output.data(), output.dimension(0), output.dimension(1), output.dimension(2));
    translate_image_y(output_map, shift);
}

}
