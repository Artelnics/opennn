//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T   H E L P E R S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <string_view>

#include "opennn/core/tensor_types.h"

namespace opennn_test
{

inline bool logical_parameters_are_approx(
    const std::vector<std::vector<opennn::TensorSpec>>& specs,
    const VectorR& left,
    const VectorR& right,
    const float tolerance)
{
    if(left.size() != right.size()) return false;

    Index offset = 0;
    for(const auto& layer_specs : specs)
        for(const opennn::TensorSpec& spec : layer_specs)
        {
            const Index size = spec.shape.size();
            if(size > 0
               && !left.segment(offset, size).isApprox(
                      right.segment(offset, size), tolerance))
            {
                return false;
            }
            offset += opennn::get_aligned_size(size);
        }

    // Alignment padding is deliberately excluded: it is not linked to an
    // operator and therefore is not part of the model's numerical state.
    return offset == left.size();
}

// A solid-colour 24-bit BMP, the fixture image every dataset suite needs.
//
// Seven suites carried their own copy of this and three of those copies had
// drifted apart: two wrote the width into one byte and the height into one or
// two, so any dimension past 255 produced a malformed header. This version
// writes all four bytes of each.
void write_bmp_24(const std::filesystem::path& path,
                  int width,
                  int height,
                  std::uint8_t red,
                  std::uint8_t green,
                  std::uint8_t blue);

// One YOLO label line: a class id and a box as normalised centre and size.
void write_label(const std::filesystem::path& path,
                 int class_id,
                 float centre_x,
                 float centre_y,
                 float width,
                 float height);

// A class-names file, one name per line.
void write_classes(const std::filesystem::path& path,
                   std::initializer_list<const char*> names);

// A directory under temp_directory_path(), removed when it goes out of scope.
// The prefix keeps each suite's directories apart, so one left behind by a
// crash still says which suite made it.
struct TempDir
{
    explicit TempDir(std::string_view prefix);
    ~TempDir();

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::filesystem::path path;
};

}
