//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <atomic>
#include <cstdio>

#include "image_dataset.h"

namespace opennn
{

class NeuralNetwork;

struct YoloDetection
{
    float center_x = 0.0f;
    float center_y = 0.0f;
    float width    = 0.0f;
    float height   = 0.0f;
    float score    = 0.0f;
    Index class_id = 0;
};

vector<YoloDetection> decode_yolo_detections(const float*,
                                             Index,
                                             Index,
                                             Index,
                                             Index,
                                             Index);

// Single Detection head's output, post-DetectionOperator (x/y sigmoid, w/h = anchor*exp,
// objectness sigmoid, class probs sigmoid/softmax). Used as input to cross-scale NMS.
struct YoloFpnHead
{
    const float* data = nullptr;     // shape implicit: [grid_size, grid_size, boxes_per_cell * (5+classes)]
    Index grid_size = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
};

// Cross-scale NMS for FPN-style YOLO inference. Decodes candidates from each
// head's already-decoded output, merges them in normalized image coords,
// runs unified class-aware greedy NMS, then letterbox-unwarps to the original
// image size. Single-sample (no batch dimension).
vector<YoloDetection> decode_yolo_fpn_detections(const vector<YoloFpnHead>&,
                                                 Index,
                                                 Index,
                                                 Index,
                                                 Index,
                                                 float confidence_threshold = 0.25f,
                                                 float iou_threshold = 0.45f);

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

    bool is_multi_scale() const noexcept { return !head_grid_sizes.empty(); }
    Index get_boxes_per_head() const noexcept { return boxes_per_head; }
    void set_multi_scale_heads(const vector<Index>&,
                               const vector<vector<array<float, 2>>>&);

    void set(const filesystem::path&,
             const filesystem::path&,
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
                     bool,
                     int contiguous = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool,
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
        bool mosaic = false;
    };

    void set_augmentation(const AugmentationConfig& cfg) { augmentation = cfg; }

    // class_filter: if non-empty, only convert objects whose class name is in the list
    // and remap class IDs to 0-indexed within the filter (writes a custom .names file).
    static Index convert_voc_to_yolo(const filesystem::path&,
                                     const string&,
                                     const filesystem::path&,
                                     const vector<string>& class_filter = {});

    // Load the first n_backbone_convs convolutional layers of network from a
    // Darknet binary weights file (e.g. yolov3-tiny.weights).  The file header
    // (20 bytes: 3×int32 + 1×int64) is consumed before walking layers.
    // Returns the number of conv layers actually loaded.
    static Index load_darknet_backbone(NeuralNetwork&,
                                       const filesystem::path&,
                                       Index);

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

    void open_or_build_cache(const vector<array<float, 2>>&);
    bool try_open_cache(const vector<array<float, 2>>&);
    void build_cache(const vector<array<float, 2>>&);
    void setup_metadata(Index);
    void read_sample_boxes(Index, vector<Box>&) const;
    void load_images_to_ram() const;
    void load_targets_to_ram() const;

    mutable vector<uint8_t> images_ram;
    mutable vector<float> targets_ram;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
