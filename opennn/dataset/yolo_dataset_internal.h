// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/dataset/yolo_dataset.h"

namespace opennn::yolo_detail
{


inline float clamp_unit(const float value)
{
    return min(1.0f, max(0.0f, value));
}
#pragma pack(push, 1)
struct YoloImageCacheHeader
{
    char magic[8];
    uint32_t version;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint64_t samples;
    uint64_t record_bytes;
    uint64_t sources_hash;
    uint8_t pad[16];
};

struct YoloTargetCacheHeader
{
    char magic[8];
    uint32_t version;
    uint32_t grid_size;
    uint32_t boxes_per_cell;
    uint32_t classes_number;
    uint64_t samples;
    uint64_t target_floats;
    uint64_t anchors_hash;
    uint64_t anchors_offset;
    uint64_t targets_offset;
};

struct YoloBoxesCacheHeader
{
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t samples;
    uint64_t total_boxes;
    uint64_t offsets_byte_offset;
    uint64_t boxes_byte_offset;
    uint8_t pad[16];
};

struct YoloBoxRecord
{
    int32_t class_id;
    float x;
    float y;
    float w;
    float h;
};
#pragma pack(pop)

static_assert(sizeof(YoloImageCacheHeader) == 64);
static_assert(sizeof(YoloTargetCacheHeader) == 64);
static_assert(sizeof(YoloBoxesCacheHeader) == 64);
static_assert(sizeof(YoloBoxRecord) == 20);

constexpr uint32_t YOLO_CACHE_VERSION = 3;
constexpr char YOLO_IMAGE_MAGIC[8] = {'O','P','E','N','N','Y','I','M'};
constexpr char YOLO_TARGET_MAGIC[8] = {'O','P','E','N','N','Y','T','G'};
constexpr char YOLO_BOXES_MAGIC[8] = {'O','P','E','N','N','Y','B','X'};
inline float yolo_iou_wh(const array<float, 2>& box, const array<float, 2>& anchor)
{
    const float inter = min(box[0], anchor[0]) * min(box[1], anchor[1]);
    const float area = box[0] * box[1] + anchor[0] * anchor[1] - inter;
    return area > 0.0f ? inter / area : 0.0f;
}
inline Index best_anchor_for_box(const YoloDataset::Box& box,
                          const vector<array<float, 2>>& anchors)
{
    float best_iou = -1.0f;
    Index best = 0;

    for (Index i = 0; i < ssize(anchors); ++i)
    {
        const float iou = yolo_iou_wh({box.w, box.h}, anchors[size_t(i)]);
        if (iou > best_iou)
        {
            best_iou = iou;
            best = i;
        }
    }

    return best;
}
inline Index grid_cell(float coordinate, Index grid)
{
    return min<Index>(grid - 1, max<Index>(0, Index(floor(coordinate * grid))));
}

inline void write_box_target(float* cell, const YoloDataset::Box& box,
                      Index grid, Index col, Index row, float objectness)
{
    cell[0] = box.x * grid - float(col);
    cell[1] = box.y * grid - float(row);
    cell[2] = box.w;
    cell[3] = box.h;
    cell[4] = objectness;
    cell[5 + box.class_id] = 1.0f;
}

inline void mark_ignored_anchors(const YoloDataset::Box& box,
                          const vector<array<float, 2>>& anchors,
                          Index grid, Index head_offset,
                          Index values_per_box, Index head_channels,
                          Index skip_anchor, float* target)
{
    for (Index j = 0; j < ssize(anchors); ++j)
    {
        if (j == skip_anchor) continue;
        if (yolo_iou_wh({box.w, box.h}, anchors[size_t(j)]) < 0.5f) continue;
        const Index col = grid_cell(box.x, grid);
        const Index row = grid_cell(box.y, grid);
        const Index base = head_offset + (row * grid + col) * head_channels + j * values_per_box;
        if (target[base + 4] < 0.5f)
            target[base + 4] = -1.0f;
    }
}

struct AnchorMatch
{
    float iou = -1.0f;
    size_t head = 0;
    Index anchor = 0;
};

inline AnchorMatch best_anchor_across_heads(const YoloDataset::Box& box,
                                     const vector<vector<array<float, 2>>>& head_anchors)
{
    AnchorMatch match;
    for (size_t i = 0; i < head_anchors.size(); ++i)
    {
        const vector<array<float, 2>>& anchors_h = head_anchors[i];
        for (Index j = 0; j < ssize(anchors_h); ++j)
        {
            const float iou = yolo_iou_wh({box.w, box.h}, anchors_h[size_t(j)]);
            if (iou > match.iou)
                match = {iou, i, j};
        }
    }
    return match;
}

inline void make_target(const vector<YoloDataset::Box>& boxes,
                 const vector<array<float, 2>>& anchors,
                 Index grid_size,
                 Index boxes_per_cell,
                 Index classes_number,
                 float* target)
{
    if (boxes_per_cell <= 0) return;

    const Index values_per_box = 5 + classes_number;
    const Index channels = boxes_per_cell * values_per_box;
    fill_n(target, grid_size * grid_size * channels, 0.0f);

    for (const auto& box : boxes)
    {
        if (box.class_id < 0 || box.class_id >= classes_number)
            continue;

        const Index col = grid_cell(box.x, grid_size);
        const Index row = grid_cell(box.y, grid_size);
        const Index anchor = best_anchor_for_box(box, anchors);
        const float best_iou = yolo_iou_wh({box.w, box.h}, anchors[size_t(anchor)]);
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        write_box_target(target + base, box, grid_size, col, row,
                         max(best_iou, 0.5f));
        mark_ignored_anchors(box, anchors, grid_size, 0,
                             values_per_box, channels, anchor, target);
    }
}

inline void make_target_multi_scale(const vector<YoloDataset::Box>& boxes,
                             const vector<vector<array<float, 2>>>& head_anchors,
                             const vector<Index>& head_grid_sizes,
                             Index boxes_per_head,
                             Index classes_number,
                             float* target)
{
    const Index values_per_box = 5 + classes_number;
    const Index head_channels = boxes_per_head * values_per_box;

    Index total_floats = 0;
    vector<Index> head_offsets(head_grid_sizes.size() + 1, 0);
    for (size_t i = 0; i < head_grid_sizes.size(); ++i)
    {
        const Index head_floats = head_grid_sizes[i] * head_grid_sizes[i] * head_channels;
        head_offsets[i + 1] = head_offsets[i] + head_floats;
        total_floats += head_floats;
    }
    fill_n(target, total_floats, 0.0f);

    for (const auto& box : boxes)
    {
        if (box.class_id < 0 || box.class_id >= classes_number)
            continue;

        const AnchorMatch best = best_anchor_across_heads(box, head_anchors);

        const Index grid_h = head_grid_sizes[best.head];
        const Index col = grid_cell(box.x, grid_h);
        const Index row = grid_cell(box.y, grid_h);
        const Index base = head_offsets[best.head]
                         + (row * grid_h + col) * head_channels
                         + best.anchor * values_per_box;

        write_box_target(target + base, box, grid_h, col, row,
                         max(best.iou, 0.5f));

        for (size_t i = 0; i < head_anchors.size(); ++i)
            mark_ignored_anchors(box, head_anchors[i], head_grid_sizes[i], head_offsets[i],
                                 values_per_box, head_channels,
                                 i == best.head ? best.anchor : Index(-1), target);
    }
}

inline void make_target_v8_gtlist(const vector<YoloDataset::Box>& boxes,
                                   Index classes_number,
                                   float* target)
{
    constexpr Index MAX_GT = YoloDataset::MAX_GT_BOXES;
    fill_n(target, MAX_GT * 5, 0.0f);
    const Index n = min<Index>(ssize(boxes), MAX_GT);
    for (Index i = 0; i < n; ++i)
    {
        const auto& b = boxes[size_t(i)];
        if (b.class_id < 0 || b.class_id >= classes_number) continue;
        target[i*5 + 0] = b.x;
        target[i*5 + 1] = b.y;
        target[i*5 + 2] = b.w;
        target[i*5 + 3] = b.h;
        target[i*5 + 4] = float(b.class_id + 1);
    }
}

}
