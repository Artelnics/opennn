// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/dataset/dataset.h"

namespace opennn
{

// Binary semantic segmentation: one image per sample and one mask of the same
// spatial size. Both are decoded once into the in-memory data matrix, input
// columns first and mask columns second, so batch filling, fold splitting and
// device residency work through the base class unchanged.
//
// Masks are nearest-neighbour resampled and thresholded, so a mask stays a
// binary image instead of picking up the grey border bilinear interpolation
// would leave along every class boundary.
class SegmentationDataset final : public Dataset
{

public:

    SegmentationDataset() = default;

    SegmentationDataset(const filesystem::path& images_directory,
                        const filesystem::path& masks_directory,
                        const Shape& input_shape = {});

    void read_images();

    Index get_masks_number() const noexcept { return ssize(mask_paths); }

    const vector<filesystem::path>& get_image_paths() const noexcept { return image_paths; }

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     FillMode,
                     ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      FillMode,
                      ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    FeatureScaling prepare_training_scaling(VariableRole, const FeatureScaling&, Index) override;
    FeatureScaling calculate_used_feature_scaling(VariableRole) const override;

private:

    filesystem::path masks_directory;

    vector<filesystem::path> image_paths;
    vector<filesystem::path> mask_paths;

    Shape requested_input_shape;
};

}