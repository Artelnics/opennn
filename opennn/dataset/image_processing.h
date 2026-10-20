// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/opennn_types.h"

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

template <typename Pixel>
inline float bilinear_blend(Pixel v00, Pixel v01, Pixel v10, Pixel v11, float dx, float dy)
{
    const float top    = float(v00) * (1.0f - dx) + float(v01) * dx;
    const float bottom = float(v10) * (1.0f - dx) + float(v11) * dx;
    return top * (1.0f - dy) + bottom * dy;
}

void reflect_image_horizontal(TensorMap3&);
void reflect_image_vertical(TensorMap3&);
void rotate_image(const TensorMap3&, TensorMap3&, float);
void translate_image_x(TensorMap3&, Index);
void translate_image_y(TensorMap3&, Index);
}
