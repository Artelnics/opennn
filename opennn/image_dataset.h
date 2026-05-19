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

/// @brief Image augmentation parameters: reflections, rotations and translations applied at training.
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

/// @brief Image dataset that streams BMP images from disk with optional augmentation.
class ImageDataset : public Dataset
{

public:

    /// @brief Creates an image dataset with the given sample count, image shape and target shape.
    /// @param sample_count Number of samples to allocate.
    /// @param input_shape Image shape (height, width, channels).
    /// @param target_shape Shape of the target tensor.
    ImageDataset(const Index = 0, const Shape& = {0, 0, 0}, const Shape& = {0});

    /// @brief Creates an image dataset by reading the BMP archive at the given path.
    ImageDataset(const filesystem::path&);

    /// @brief Returns the number of image channels (typically 1 or 3).
    Index get_channels_number() const;

    const AugmentationSettings& get_augmentation() const { return augmentation; }
    void set_augmentation(const AugmentationSettings& new_augmentation) { augmentation = new_augmentation; }

    /// @brief Returns the number of image samples in the dataset.
    Index get_samples_number() const override;
    using Dataset::get_samples_number;

    /// @brief Generates random image data, replacing the current contents.
    void set_data_random() override;

    void set_image_padding(int new_padding) { padding = new_padding; }

    using Dataset::unscale_features;

    /// @brief Scales image pixel values for the given feature role.
    vector<Descriptives> scale_features(const string&) override;
    /// @brief Reverts the pixel scaling for the given feature role.
    void unscale_features(const string&);

    /// @brief Reads BMP images into the dataset, resizing to @p new_input_shape when non-zero.
    void read_bmp(const Shape& new_input_shape = { 0, 0, 0 });

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    /// @brief Streams pixel data of the selected samples into the destination buffer.
    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     bool parallelize = true,
                     int = -1) const override;

    /// @brief Writes one-hot encoded labels of the selected samples into the destination buffer.
    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int = -1) const override;

    /// @brief Applies the configured augmentations in place to a batch of images.
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
