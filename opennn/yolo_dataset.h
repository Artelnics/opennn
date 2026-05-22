//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "image_dataset.h"

namespace opennn
{

struct YoloDetection
{
    float center_x = 0.0f;   // pixel coords in the original image
    float center_y = 0.0f;
    float width    = 0.0f;
    float height   = 0.0f;
    float score    = 0.0f;
    Index class_id = 0;
};

// Decode one batch element of a NonMaxSuppression layer output into bounding
// boxes expressed in the original (pre-letterbox) image's pixel coordinates.
// nms_output rows are (x, y, w, h, score, class_id) where x,y,w,h are
// normalized to the letterboxed network input and zero rows are empty slots.
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
    const string& get_class_name(Index i) const { return class_names[size_t(i)]; }

    void set(const filesystem::path& images_dir,
             const filesystem::path& labels_dir,
             const Shape& input_shape = {416, 416, 3},
             Index grid_size = 13,
             Index boxes_per_cell = 5,
             const vector<array<float, 2>>& anchors = {});

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     bool parallelize = true,
                     int contiguous = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int contiguous = -1) const override;

    void augment_inputs(float*, Index) const override {}

private:

    filesystem::path images_directory;
    filesystem::path labels_directory;
    filesystem::path image_cache_path;
    filesystem::path target_cache_path;

    mutable FileReader image_cache_reader;
    mutable FileReader target_cache_reader;

    Index samples_number = 0;
    Index grid_size = 13;
    Index boxes_per_cell = 5;
    Index classes_number = 0;
    Index image_record_bytes = 0;
    Index target_record_floats = 0;
    uint64_t target_data_offset = 0;

    vector<array<float, 2>> anchors;
    vector<string> class_names;

    void open_or_build_cache(const vector<array<float, 2>>& requested_anchors);
    bool try_open_cache(const vector<array<float, 2>>& requested_anchors);
    void build_cache(const vector<array<float, 2>>& requested_anchors);
    void setup_metadata(Index new_samples_number);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
