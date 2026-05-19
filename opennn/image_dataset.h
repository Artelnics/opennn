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

    ImageDataset(const Index = 0, const Shape& = {0, 0, 0}, const Shape& = {0});

    ImageDataset(const filesystem::path&);

    Index get_channels_number() const;

    const AugmentationSettings& get_augmentation() const { return augmentation; }
    void set_augmentation(const AugmentationSettings& new_augmentation) { augmentation = new_augmentation; }

    Index get_samples_number() const override;
    using Dataset::get_samples_number;

    void set_data_random() override;

    void set_image_padding(int new_padding) { padding = new_padding; }

    using Dataset::unscale_features;

    vector<Descriptives> scale_features(const string&) override;
    void unscale_features(const string&);

    void read_bmp(const Shape& new_input_shape = { 0, 0, 0 });

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     bool parallelize = true,
                     int = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int = -1) const override;

    void augment_inputs(float*, Index) const override;

private:

    Index padding = 0;

    AugmentationSettings augmentation;

    // Binary streaming cache: <data_path>/.cache/images.bin
    // Pixels stored as uint8 [0..255]; labels as int32_t.
    filesystem::path cache_path;
    mutable FileReader cache_reader;
    uint64_t record_bytes_ = 0;
    uint64_t labels_off_   = 0;
    uint32_t num_classes_  = 0;
    vector<int32_t> labels_ram;       // small; one entry per sample

    vector<string> labels_tokens;

    Index width_no_padding = 0;

    Index regions_number = 1000;
    Index region_rows = 6;
    Index region_variables = 6;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
