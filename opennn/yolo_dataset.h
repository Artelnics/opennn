//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <atomic>

#include "image_dataset.h"

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

vector<YoloDetection> decode_yolo_detections(const float* nms_output,
                                             Index max_boxes,
                                             Index original_height,
                                             Index original_width,
                                             Index network_height,
                                             Index network_width);

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

    YoloDataset(const filesystem::path& images_dir,
                const filesystem::path& labels_dir,
                const Shape& input_shape = {416, 416, 3},
                Index grid_size = 13,
                Index boxes_per_cell = 5,
                const vector<array<float, 2>>& anchors = {});

    Index get_samples_number() const override { return samples_number; }
    using Dataset::get_samples_number;

    Index get_grid_size() const { return grid_size; }
    Index get_boxes_per_cell() const { return boxes_per_cell; }
    Index get_classes_number() const { return ssize(class_names); }
    const vector<array<float, 2>>& get_anchors() const { return anchors; }
    const vector<string>& get_class_names() const { return class_names; }
    const filesystem::path& get_image_path(Index i) const { return image_filenames[size_t(i)]; }

    bool is_multi_scale() const { return !head_grid_sizes.empty(); }
    Index get_boxes_per_head() const { return boxes_per_head; }
    void set_multi_scale_heads(const vector<Index>& grid_sizes,
                               const vector<vector<array<float, 2>>>& per_head_anchors);

    void set(const filesystem::path& images_dir,
             const filesystem::path& labels_dir,
             const Shape& input_shape = {416, 416, 3},
             Index grid_size = 13,
             Index boxes_per_cell = 5,
             const vector<array<float, 2>>& anchors = {});
    using Dataset::set_storage_mode;
    void set_storage_mode(StorageMode) override;

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     int contiguous = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      int contiguous = -1) const override;

    void augment_inputs(float*, Index) const override {}

    struct AugmentationConfig
    {
        float jitter = 0.2f;
        float exposure = 1.5f;
        float saturation = 1.5f;
        float hue = 0.1f;
        bool flip = true;
        bool enabled = true;
    };

    void set_augmentation(const AugmentationConfig& cfg) { augmentation = cfg; }

    static Index convert_voc_to_yolo(const filesystem::path& voc_root,
                                     const string& image_set,
                                     const filesystem::path& output_labels_dir);

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

    AugmentationConfig augmentation{};
    mutable atomic<uint64_t> augmentation_counter{0};

    vector<filesystem::path> image_filenames;
    vector<array<float, 2>> anchors;
    vector<string> class_names;

    vector<Index> head_grid_sizes;
    vector<vector<array<float, 2>>> head_anchors;
    Index boxes_per_head = 0;

    void open_or_build_cache(const vector<array<float, 2>>& requested_anchors);
    bool try_open_cache(const vector<array<float, 2>>& requested_anchors);
    void build_cache(const vector<array<float, 2>>& requested_anchors);
    void setup_metadata(Index new_samples_number);
    void read_sample_boxes(Index sample_index, vector<Box>& out) const;
    void load_images_to_ram() const;
    void load_targets_to_ram() const;

    mutable vector<uint8_t> images_ram;
    mutable vector<float> targets_ram;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
