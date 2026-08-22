//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T   H E L P E R S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tests/pch.h"

#include "tests/test_helpers.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace opennn_test
{

namespace
{

void write_u16(std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back(std::uint8_t(value & 0xFF));
    bytes.push_back(std::uint8_t((value >> 8) & 0xFF));
}


void write_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(std::uint8_t(value & 0xFF));
    bytes.push_back(std::uint8_t((value >> 8) & 0xFF));
    bytes.push_back(std::uint8_t((value >> 16) & 0xFF));
    bytes.push_back(std::uint8_t((value >> 24) & 0xFF));
}

}


void write_bmp_24(const std::filesystem::path& path,
                  int width,
                  int height,
                  std::uint8_t red,
                  std::uint8_t green,
                  std::uint8_t blue)
{
    constexpr int bytes_per_pixel = 3;
    constexpr std::uint32_t header_size = 54;

    const int row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;
    const std::uint32_t pixel_array_size = std::uint32_t(row_stride * height);

    std::vector<std::uint8_t> bytes;
    bytes.reserve(header_size + pixel_array_size);

    bytes.push_back('B');
    bytes.push_back('M');
    write_u32(bytes, header_size + pixel_array_size);
    write_u16(bytes, 0);
    write_u16(bytes, 0);
    write_u32(bytes, header_size);

    write_u32(bytes, 40);                       // BITMAPINFOHEADER size
    write_u32(bytes, std::uint32_t(width));
    write_u32(bytes, std::uint32_t(height));
    write_u16(bytes, 1);                        // colour planes
    write_u16(bytes, 24);                       // bits per pixel
    write_u32(bytes, 0);                        // BI_RGB, no compression
    write_u32(bytes, pixel_array_size);
    write_u32(bytes, 2835);                     // 72 DPI, horizontal
    write_u32(bytes, 2835);                     // 72 DPI, vertical
    write_u32(bytes, 0);                        // palette colours
    write_u32(bytes, 0);                        // important colours

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            bytes.push_back(blue);
            bytes.push_back(green);
            bytes.push_back(red);
        }
        for (int pad = width * bytes_per_pixel; pad < row_stride; ++pad)
            bytes.push_back(0);
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}


void write_label(const std::filesystem::path& path,
                 int class_id,
                 float centre_x,
                 float centre_y,
                 float width,
                 float height)
{
    std::ofstream out(path);
    out << class_id << ' ' << centre_x << ' ' << centre_y << ' '
        << width << ' ' << height << '\n';
}


void write_classes(const std::filesystem::path& path,
                   std::initializer_list<const char*> names)
{
    std::ofstream out(path);
    for (const char* name : names)
        out << name << '\n';
}


TempDir::TempDir(std::string_view prefix)
{
    const std::filesystem::path base = std::filesystem::temp_directory_path();

    for (int i = 0; i < 10000; ++i)
    {
        const std::filesystem::path candidate = base / (std::string(prefix) + std::to_string(i));

        std::error_code error;

        if (std::filesystem::create_directories(candidate, error) && !error)
        {
            path = candidate;
            return;
        }
    }

    throw std::runtime_error("TempDir: could not create a temporary directory for " + std::string(prefix));
}


TempDir::~TempDir()
{
    std::error_code error;
    std::filesystem::remove_all(path, error);
}

}
