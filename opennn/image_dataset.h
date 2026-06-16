//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"
#include "io_utilities.h"
#include "variable.h"

namespace opennn
{

struct AugmentationSettings
{
    bool enabled = false;
    bool reflection_axis_x = false;
    bool reflection_axis_y = false;
    float rotation_minimum = 0.0f;
    float rotation_maximum = 0.0f;
    float horizontal_translation_minimum = 0.0f;
    float horizontal_translation_maximum = 0.0f;
    float vertical_translation_minimum = 0.0f;
    float vertical_translation_maximum = 0.0f;
};

class ImageDataset : public Dataset
{

public:

    ImageDataset() = default;
    ImageDataset(const filesystem::path&);

    Index get_channels_number() const;

    void set_augmentation(const AugmentationSettings& new_augmentation) { augmentation = new_augmentation; }
    void set_input_scaling(const vector<Descriptives>&,
                           const vector<ScalerMethod>&,
                           float min_range,
                           float max_range);

    VectorI calculate_target_distribution() const override;

    void enable_device_residency() override;

    void read_images();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     int = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      int = -1) const override;

    void augment_inputs(float*, Index) const override;

private:

    void write_image_cache(const vector<filesystem::path>& paths);

    AugmentationSettings augmentation;

    vector<float> input_scale;
    vector<float> input_offset;

    filesystem::path cache_path;
    mutable FileReader cache_reader;

    uint64_t pixel_number = 0;
    uint32_t classes_number = 0;
    vector<int32_t> sample_labels;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
