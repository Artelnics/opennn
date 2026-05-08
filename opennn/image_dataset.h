//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file image_dataset.h
 * @brief Declares the ImageDataset specialization of Dataset for
 *        image data, plus the AugmentationSettings configuration struct.
 */

#pragma once

#include "dataset.h"

namespace opennn
{

/**
 * @struct AugmentationSettings
 * @brief Optional data-augmentation transforms applied at training time.
 *
 * All augmentations are disabled by default; set @ref enabled to true and
 * configure the per-transform fields to activate them. Each rotation /
 * translation field defines a uniform sampling range whose lower bound is
 * the *_minimum field and upper bound is the *_maximum field.
 */
struct AugmentationSettings
{
    /** @brief Master switch; when false no augmentation is applied. */
    bool enabled = false;
    /** @brief Random horizontal reflections (along the X axis) when true. */
    bool reflection_axis_x = false;
    /** @brief Random vertical reflections (along the Y axis) when true. */
    bool reflection_axis_y = false;
    /** @brief Lower bound of the uniform rotation range, in degrees. */
    float rotation_minimum = 0.0f;
    /** @brief Upper bound of the uniform rotation range, in degrees. */
    float rotation_maximum = 0.0f;
    /** @brief Lower bound of the uniform horizontal translation range, in pixels. */
    float horizontal_translation_minimum = 0.0f;
    /** @brief Upper bound of the uniform horizontal translation range, in pixels. */
    float horizontal_translation_maximum = 0.0f;
    /** @brief Lower bound of the uniform vertical translation range, in pixels. */
    float vertical_translation_minimum = 0.0f;
    /** @brief Upper bound of the uniform vertical translation range, in pixels. */
    float vertical_translation_maximum = 0.0f;
};

/**
 * @class ImageDataset
 * @brief Dataset specialization for image data.
 *
 * Stores raw image pixels plus per-image labels and supports BMP loading,
 * per-channel padding and on-the-fly data augmentation. Inherits the
 * common partitioning, scaling and statistics machinery from Dataset.
 */
class ImageDataset : public Dataset
{

public:

    /**
     * @brief Constructs an empty ImageDataset of given dimensions.
     * @param samples_number Number of samples (images).
     * @param input_shape Per-image input shape (height, width, channels).
     * @param target_shape Per-image target shape (typically class count for classification).
     */
    ImageDataset(const Index samples_number = 0,
                 const Shape& input_shape = {0, 0, 0},
                 const Shape& target_shape = {0});

    /**
     * @brief Constructs an ImageDataset by loading from a directory of BMP files.
     * @param directory Path to a directory whose subdirectories define class labels.
     */
    ImageDataset(const filesystem::path& directory);

    /**
     * @brief Number of color channels per image.
     * @return Trailing dimension of the input shape.
     */
    Index get_channels_number() const;

    /** @brief Read-only access to the augmentation configuration. */
    const AugmentationSettings& get_augmentation() const { return augmentation; }
    /**
     * @brief Replaces the augmentation configuration.
     * @param new_augmentation New AugmentationSettings value.
     */
    void set_augmentation(const AugmentationSettings& new_augmentation) { augmentation = new_augmentation; }

    /** @brief Fills the dataset with random pixel values (debug helper). */
    void set_data_random() override;

    /**
     * @brief Sets the per-image padding in pixels.
     * @param new_padding Padding added on every side of each image.
     */
    void set_image_padding(const int& new_padding) { padding = new_padding; }

    /**
     * @brief Computes per-channel descriptives and rescales pixel values in place.
     *
     * Receives the scaler method name; see ScalerMethod for the supported
     * set. Returns the per-channel descriptives produced before scaling.
     */
    vector<Descriptives> scale_features(const string&) override;
    /**
     * @brief Inverse of scale_features(); restores raw pixel values.
     *
     * Receives the scaler method name used in scale_features().
     */
    void unscale_features(const string&);

    /**
     * @brief Loads BMP files from the dataset directory into memory.
     * @param new_input_shape Optional override for the per-image input shape.
     */
    void read_bmp(const Shape& new_input_shape = { 0, 0, 0 });

    /**
     * @brief Loads dataset metadata from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&) override;
    /**
     * @brief Writes dataset metadata to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const override;

    /**
     * @brief Applies random augmentations in place to the input batch buffer.
     *
     * Receives the device-resident batch buffer pointer and the batch size.
     */
    void augment_inputs(float*, Index) const override;

private:

    /** @brief Padding added on every side of each image, in pixels. */
    Index padding = 0;

    /** @brief Augmentation transforms applied at training time. */
    AugmentationSettings augmentation;

    /** @brief Class labels recovered from subdirectory names. */
    vector<string> labels_tokens;

    /** @brief Image width before padding is applied. */
    Index width_no_padding = 0;

    /** @brief Number of region proposals (object detection). */
    Index regions_number = 1000;
    /** @brief Number of rows per region proposal. */
    Index region_rows = 6;
    /** @brief Number of variables per region proposal. */
    Index region_variables = 6;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
