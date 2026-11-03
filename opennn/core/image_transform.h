// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

struct ImageColorTransform
{
    float exposure = 1.0f;
    float saturation = 1.0f;
    float hue = 0.0f;
};

struct ImageCropTransform
{
    float crop_left = 0.0f;
    float crop_top = 0.0f;
    float crop_right = 0.0f;
    float crop_bottom = 0.0f;
    bool flip = false;
    float exposure_mul = 1.0f;
    float saturation_mul = 1.0f;
    float hue_shift = 0.0f;
};

struct ImageAffineTransform
{
    float angle = 0.0f;
    float cosine = 1.0f;
    float sine = 0.0f;
    Index shift_x = 0;
    Index shift_y = 0;
    bool flip_x = false;
    bool flip_y = false;
};

struct ImageRegion
{
    Index x = 0;
    Index y = 0;
    Index width = 0;
    Index height = 0;
};

struct ImageTransformParameters
{
    ImageCropTransform crop;
    ImageAffineTransform affine;
    Index sources[4]{};
    ImageColorTransform colors[4];
    Index cut_x = 0;
    Index cut_y = 0;
};

struct ImageChannelScale
{
    float scale = 1.0f / 255.0f;
    float offset = 0.0f;
};

struct ImageTransformBatch
{
    enum class Kind { Affine, Crop };
    Kind kind = Kind::Affine;
    Index samples = 0;
    Index sources = 0;
    Index height = 0;
    Index width = 0;
    Index channels = 0;
    Index output_height = 0;
    Index output_width = 0;
    Index cutouts = 0;
    bool float_input = false;
    bool enabled = false;
    bool mosaic = false;

    Index pixels() const { return height * width * channels; }
    Index source_bytes() const { return sources * pixels() * (float_input ? sizeof(float) : 1); }
    Index scales_offset() const { return samples * sizeof(ImageTransformParameters); }
    Index cutouts_offset() const { return scales_offset() + channels * sizeof(ImageChannelScale); }
    Index parameters_bytes() const { return cutouts_offset() + samples * cutouts * sizeof(ImageRegion); }
    bool resize() const { return height != output_height || width != output_width; }
    Index scratch_bytes() const { return samples * pixels() * (Index(mosaic) + Index(resize())); }
};

}
