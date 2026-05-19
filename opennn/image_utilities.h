//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

/// @brief Loads an image from disk into a rank-3 (height, width, channels) tensor.
Tensor3 load_image(const filesystem::path&);

/// @brief Loads an image into a pre-allocated float buffer at the given shape.
/// @param path Image file on disk.
/// @param dst Destination buffer with capacity expected_height * expected_width * expected_channels.
/// @param expected_height Required output height (image is resized if it differs).
/// @param expected_width Required output width.
/// @param expected_channels Required number of channels.
/// @param divide_by_255 Divide pixel values by 255 to produce a [0, 1] range.
void load_image(const filesystem::path&,
                float* dst,
                Index expected_height,
                Index expected_width,
                Index expected_channels,
                bool divide_by_255 = false);

/// @brief Returns a resized copy of an image at the requested height and width.
Tensor3 resize_image(const Tensor3&, Index, Index);

/// @brief Mirrors the image horizontally (left-right) in place.
void reflect_image_horizontal(Tensor3&);

/// @brief Mirrors the image vertically (top-bottom) in place.
void reflect_image_vertical(Tensor3&);

/// @brief Rotates the image by the given angle (radians) into the destination tensor.
void rotate_image(const Tensor3&, Tensor3&, float);

/// @brief Translates the image along the X axis by the given number of pixels.
void translate_image_x(const Tensor3&, Tensor3&, Index);

/// @brief Translates the image along the Y axis by the given number of pixels.
void translate_image_y(const Tensor3&, Tensor3&, Index);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
