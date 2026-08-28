//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <cstdio>

#include "opennn/dataset/image_dataset.h"

namespace opennn
{

struct YoloDetection
{
    float center_x = 0.0f;
    float center_y = 0.0f;
    float width    = 0.0f;
    float height   = 0.0f;
    float score    = 0.0f;
    Index class_id = 0;
};

vector<YoloDetection> decode_yolo_detections(span<const float>,
                                             Index,
                                             Index,
                                             Index,
                                             Index);

struct YoloFpnHead
{
    span<const float> data;
    Index grid_size = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
};

vector<YoloDetection> decode_yolo_fpn_detections(const vector<YoloFpnHead>&,
                                                 Index,
                                                 Index,
                                                 Index,
                                                 Index,
                                                 float confidence_threshold = 0.25f,
                                                 float iou_threshold = 0.45f);

vector<YoloDetection> decode_yolo_v8_fpn_detections(const vector<YoloFpnHead>&,
                                                     Index,
                                                     Index,
                                                     Index,
                                                     Index,
                                                     float confidence_threshold = 0.25f,
                                                     float iou_threshold = 0.45f,
                                                     Index reg_max = 1);

class YoloDataset final : public ImageDataset
{
public:

    struct Box
    {
        Index class_id = 0;
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
    };

    YoloDataset() = default;

    YoloDataset(const filesystem::path&,
                const filesystem::path&,
                const Shape& input_shape = {416, 416, 3},
                Index grid_size = 13,
                Index boxes_per_cell = 5,
                const vector<array<float, 2>>& anchors = {});

    Index get_samples_number() const noexcept override { return samples_number; }
    using Dataset::get_samples_number;

    Index get_grid_size() const noexcept { return grid_size; }
    Index get_boxes_per_cell() const noexcept { return boxes_per_cell; }
    Index get_classes_number() const noexcept { return ssize(class_names); }
    const vector<array<float, 2>>& get_anchors() const noexcept { return anchors; }
    const vector<string>& get_class_names() const noexcept { return class_names; }
    const filesystem::path& get_image_path(Index i) const { return image_filenames[size_t(i)]; }
    const filesystem::path& get_images_directory() const { return images_directory; }
    const filesystem::path& get_labels_directory() const { return labels_directory; }
    const Shape& get_input_shape() const { return cache_input_shape; }

    float get_display_confidence_threshold() const noexcept { return display_confidence_threshold; }
    void  set_display_confidence_threshold(float t) { display_confidence_threshold = t; }

    bool is_multi_scale() const noexcept { return !head_grid_sizes.empty(); }
    Index get_boxes_per_head() const noexcept { return boxes_per_head; }
    void set_multi_scale_heads(const vector<Index>&,
                               const vector<vector<array<float, 2>>>&);

    static constexpr Index MAX_GT_BOXES = 100;

    bool is_v8_mode() const noexcept { return v8_mode; }
    void set_v8_mode(bool enabled);
    Index get_target_record_floats() const noexcept { return target_record_floats; }

    void set(const filesystem::path&,
             const filesystem::path&,
             const Shape& input_shape = {416, 416, 3},
             Index grid_size = 13,
             Index boxes_per_cell = 5,
             const vector<array<float, 2>>& anchors = {});
    using Dataset::set_storage_mode;
    void set_storage_mode(StorageMode) override;
    void enable_device_residency() override;

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

    struct AugmentationPolicy
    {
        float jitter = 0.2f;
        float exposure = 1.5f;
        float saturation = 1.5f;
        float hue = 0.1f;
        bool flip = true;
        bool enabled = true;
        bool mosaic = false;
    };

    void set_augmentation_policy(const AugmentationPolicy&);
    const AugmentationPolicy& get_augmentation_policy() const { return augmentation_policy; }

    static Index convert_voc_to_yolo(const filesystem::path&,
                                     const string&,
                                     const filesystem::path&,
                                     const vector<string>& class_filter = {});

private:

    filesystem::path images_directory;
    filesystem::path labels_directory;
    filesystem::path image_cache_path;
    filesystem::path target_cache_path;
    filesystem::path boxes_cache_path;

    mutable FileReader image_cache_reader;
    mutable FileReader target_cache_reader;
    mutable FileReader boxes_cache_reader;

    Index samples_number = 0;
    Index grid_size = 13;
    Index boxes_per_cell = 5;
    Index classes_number = 0;
    Index image_record_bytes = 0;
    Index target_record_floats = 0;

    Shape cache_input_shape;
    Index cache_grid_size = 0;
    Index cache_image_record_bytes = 0;
    Index cache_target_record_floats = 0;
    uint64_t target_data_offset = 0;
    uint64_t boxes_data_offset = 0;
    vector<uint64_t> boxes_offsets;

    AugmentationPolicy augmentation_policy{};
    float display_confidence_threshold = 0.25f;

    vector<filesystem::path> image_filenames;
    vector<array<float, 2>> anchors;
    vector<string> class_names;

    vector<Index> head_grid_sizes;
    vector<vector<array<float, 2>>> head_anchors;
    Index boxes_per_head = 0;

    bool v8_mode = false;

    void open_or_build_cache(const vector<array<float, 2>>&);
    bool try_open_cache(const vector<array<float, 2>>&);
    bool try_rebuild_target_from_boxes(const vector<array<float, 2>>&);
    void build_cache(const vector<array<float, 2>>&);
    void setup_metadata(Index);
    void load_cache_to_ram();

    vector<uint8_t> images_ram;
    vector<float> targets_ram;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
