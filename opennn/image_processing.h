//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   P R O C E S S I N G   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"

namespace opennn
{

Tensor3 load_image(const filesystem::path&);

bool is_supported_image_file(const filesystem::path&);

void load_image(const filesystem::path&,
                float*,
                Index,
                Index,
                Index);

Tensor3 resize_image(const Tensor3&, Index, Index);

void reflect_image_horizontal(TensorMap3&);
void reflect_image_vertical(TensorMap3&);
void rotate_image(const TensorMap3&, TensorMap3&, float);
void translate_image_x(TensorMap3&, Index);
void translate_image_y(TensorMap3&, Index);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
