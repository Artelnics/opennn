// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/dataset/yolo_dataset_internal.h"

#include <utility>

#include "opennn/core/io_utilities.h"
#include "opennn/core/json.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/string_utilities.h"
#include "opennn/dataset/image_processing.h"
#include "opennn/network/detection_head.h"

namespace opennn
{

using namespace yolo_detail;

namespace
{

uint8_t round_to_byte(const float value)
{
    return uint8_t(min(255.0f, max(0.0f, value + 0.5f)));
}

using AugmentationTransform = ImageCropTransform;

void validate_augmentation_scale(string_view name, float value)
{
    throw_if(!isfinite(value) || value <= 0.0f,
             "YOLO augmentation {} must be finite and positive; 1 disables the change.", name);
    const double largest_scale = exp(abs(double(log(value))));
    throw_if(!isfinite(largest_scale) || largest_scale > double(numeric_limits<float>::max()),
             "YOLO augmentation {} is too extreme for finite FP32 log-space sampling and reciprocals.", name);
}

AugmentationTransform sample_augmentation_transform(uint64_t epoch_seed,
                                                    uint64_t sample_index,
                                                    const YoloDataset::AugmentationPolicy& policy)
{
    RandomStream random(derive_random_seed(epoch_seed, sample_index));
    const auto rand_scale = [&](float max_scale) -> float
    {
        return exp(random.uniform(-1.0f, 1.0f) * log(max_scale));
    };

    AugmentationTransform transform{};
    transform.crop_left   = random.uniform(-policy.jitter, policy.jitter);
    transform.crop_right  = random.uniform(-policy.jitter, policy.jitter);
    transform.crop_top    = random.uniform(-policy.jitter, policy.jitter);
    transform.crop_bottom = random.uniform(-policy.jitter, policy.jitter);
    transform.flip = policy.flip && random.uniform() < 0.5f;
    transform.exposure_mul = rand_scale(policy.exposure);
    transform.saturation_mul = rand_scale(policy.saturation);
    transform.hue_shift = random.uniform(-policy.hue, policy.hue);
    return transform;
}

void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v)
{
    const float mx = max({r, g, b});
    const float mn = min({r, g, b});
    v = mx;
    const float d = mx - mn;
    s = mx > 0.0f ? d / mx : 0.0f;
    if (d <= 1e-6f) { h = 0.0f; return; }
    if (mx == r)      h = ((g - b) / d) / 6.0f;
    else if (mx == g) h = ((b - r) / d + 2.0f) / 6.0f;
    else              h = ((r - g) / d + 4.0f) / 6.0f;
    if (h < 0.0f) h += 1.0f;
}

void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b)
{
    if (s <= 0.0f) { r = g = b = v; return; }
    h = h - floor(h);
    const float h6 = h * 6.0f;
    const int i = int(floor(h6));
    const float f = h6 - i;
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * f);
    const float t = v * (1.0f - s * (1.0f - f));
    switch (i % 6)
    {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q;
    }
}

ImageRegion sample_cutout(RandomStream& random, Index width, Index height, Index max_side)
{
    const Index pw = 1 + Index(random.next() % uint64_t(min(width, max_side)));
    const Index ph = 1 + Index(random.next() % uint64_t(min(height, max_side)));
    return {Index(random.next() % uint64_t(width - pw + 1)),
            Index(random.next() % uint64_t(height - ph + 1)), pw, ph};
}

static const uint8_t* apply_cutout(uint8_t* img, const Shape& shape,
                                   const YoloDataset::AugmentationPolicy& policy,
                                   uint64_t epoch_seed, Index sample_index)
{
    if (policy.cutout_count == 0 || policy.cutout_max_size_ratio == 0.0f) return img;
    const Index h = shape[0], w = shape[1], ch = shape[2];
    RandomStream random(derive_random_seed(derive_random_seed(epoch_seed, uint64_t(sample_index)),
                                           0x4355544F5554ull));
    const Index max_side = max<Index>(1, Index(double(max(h, w)) * policy.cutout_max_size_ratio));
    for (int n = 0; n < policy.cutout_count; ++n)
    {
        const auto region = sample_cutout(random, w, h, max_side);
        for (Index dy = 0; dy < region.height; ++dy)
            fill_n(img + ((region.y + dy) * w + region.x) * ch, region.width * ch, uint8_t(0));
    }
    return img;
}

void apply_color_jitter(uint8_t* rgb, Index height, Index width, Index channels,
                        const AugmentationTransform& transform)
{
    if (channels < 3) return;
    const Index pixels = height * width;
    for (Index i = 0; i < pixels; ++i)
    {
        const Index base = i * channels;
        float r = float(rgb[base + 0]) / 255.0f;
        float g = float(rgb[base + 1]) / 255.0f;
        float b = float(rgb[base + 2]) / 255.0f;
        float h, s, v;
        rgb_to_hsv(r, g, b, h, s, v);
        h += transform.hue_shift;
        s = clamp_unit(s * transform.saturation_mul);
        v = clamp_unit(v * transform.exposure_mul);
        hsv_to_rgb(h, s, v, r, g, b);
        rgb[base + 0] = round_to_byte(r * 255.0f);
        rgb[base + 1] = round_to_byte(g * 255.0f);
        rgb[base + 2] = round_to_byte(b * 255.0f);
    }
}

void apply_geometric_to_image(const uint8_t* src, uint8_t* dst,
                              Index height, Index width, Index channels,
                              const AugmentationTransform& transform)
{
    const float original_width = float(width);
    const float original_height = float(height);
    const float source_x0 = transform.crop_left * original_width;
    const float source_y0 = transform.crop_top * original_height;
    const float crop_width = (1.0f - transform.crop_left - transform.crop_right) * original_width;
    const float crop_height = (1.0f - transform.crop_top - transform.crop_bottom) * original_height;

    for (Index dy = 0; dy < height; ++dy)
    {
        for (Index dx = 0; dx < width; ++dx)
        {
            const Index out_x = transform.flip ? (width - 1 - dx) : dx;
            const float fx = source_x0 + (float(dx) + 0.5f) * crop_width / original_width - 0.5f;
            const float fy = source_y0 + (float(dy) + 0.5f) * crop_height / original_height - 0.5f;

            const Index x0 = Index(floor(fx));
            const Index y0 = Index(floor(fy));
            const Index x1 = x0 + 1;
            const Index y1 = y0 + 1;
            const float ax = fx - float(x0);
            const float ay = fy - float(y0);

            auto sample = [&](Index sx, Index sy, Index c) -> float
            {
                if (sx < 0 || sx >= width || sy < 0 || sy >= height) return 128.0f;
                return float(src[(sy * width + sx) * channels + c]);
            };

            for (Index c = 0; c < channels; ++c)
            {
                const float v = bilinear_blend(sample(x0, y0, c), sample(x1, y0, c),
                                               sample(x0, y1, c), sample(x1, y1, c),
                                               ax, ay);
                dst[(dy * width + out_x) * channels + c] = round_to_byte(v);
            }
        }
    }
}

void resize_bilinear_into(const uint8_t* src, Index src_h, Index src_w,
                          uint8_t* destination, Index destination_w,
                          Index dst_x, Index dst_y,
                          Index out_w, Index out_h,
                          Index channels)
{
    for (Index oy = 0; oy < out_h; ++oy)
    {
        const float sy_f = (float(oy) + 0.5f) * float(src_h) / float(out_h) - 0.5f;
        const Index sy0 = min<Index>(src_h - 1, max<Index>(0, Index(floor(sy_f))));
        const Index sy1 = min<Index>(sy0 + 1, src_h - 1);
        const float dy = sy_f - float(sy0);

        for (Index ox = 0; ox < out_w; ++ox)
        {
            const float sx_f = (float(ox) + 0.5f) * float(src_w) / float(out_w) - 0.5f;
            const Index sx0 = min<Index>(src_w - 1, max<Index>(0, Index(floor(sx_f))));
            const Index sx1 = min<Index>(sx0 + 1, src_w - 1);
            const float dx = sx_f - float(sx0);

            const Index destination_offset =
                ((dst_y + oy) * destination_w + (dst_x + ox)) * channels;

            for (Index c = 0; c < channels; ++c)
            {
                const float v = bilinear_blend(src[(sy0 * src_w + sx0) * channels + c],
                                               src[(sy0 * src_w + sx1) * channels + c],
                                               src[(sy1 * src_w + sx0) * channels + c],
                                               src[(sy1 * src_w + sx1) * channels + c],
                                               dx, dy);
                destination[destination_offset + c] = round_to_byte(v);
            }
        }
    }
}

void bilinear_resize_uint8(const uint8_t* src,
                           Index src_h, Index src_w,
                           uint8_t* dst,
                           Index dst_h, Index dst_w,
                           Index channels)
{
    if (src_h == dst_h && src_w == dst_w)
    {
        memcpy(dst, src, size_t(src_h) * size_t(src_w) * size_t(channels));
        return;
    }

    resize_bilinear_into(src, src_h, src_w, dst, dst_w, 0, 0, dst_w, dst_h, channels);
}

void apply_geometric_to_boxes(vector<YoloDataset::Box>& boxes,
                              const AugmentationTransform& transform)
{
    const float crop_width = 1.0f - transform.crop_left - transform.crop_right;
    const float crop_height = 1.0f - transform.crop_top - transform.crop_bottom;

    vector<YoloDataset::Box> out;
    out.reserve(boxes.size());

    for (auto box : boxes)
    {
        const float box_x0 = box.x - 0.5f * box.w;
        const float box_y0 = box.y - 0.5f * box.h;
        const float box_x1 = box.x + 0.5f * box.w;
        const float box_y1 = box.y + 0.5f * box.h;

        float nx0 = (box_x0 - transform.crop_left) / crop_width;
        float ny0 = (box_y0 - transform.crop_top)  / crop_height;
        float nx1 = (box_x1 - transform.crop_left) / crop_width;
        float ny1 = (box_y1 - transform.crop_top)  / crop_height;

        nx0 = max(0.0f, min(1.0f, nx0));
        ny0 = max(0.0f, min(1.0f, ny0));
        nx1 = max(0.0f, min(1.0f, nx1));
        ny1 = max(0.0f, min(1.0f, ny1));

        const float nw = nx1 - nx0;
        const float nh = ny1 - ny0;
        if (nw <= 1e-3f || nh <= 1e-3f) continue;

        box.x = 0.5f * (nx0 + nx1);
        box.y = 0.5f * (ny0 + ny1);
        box.w = nw;
        box.h = nh;
        if (transform.flip) box.x = 1.0f - box.x;

        out.push_back(box);
    }

    boxes = std::move(out);
}



}


namespace
{

struct LetterboxUnwarp
{
    float scale;
    float offset_x;
    float offset_y;
    float original_width;
    float original_height;
    float network_width;
    float network_height;
};

LetterboxUnwarp make_letterbox_unwarp(Index original_height, Index original_width,
                                      Index network_height, Index network_width)
{
    LetterboxUnwarp unwarp;
    unwarp.scale = min(float(network_width)  / float(original_width),
                       float(network_height) / float(original_height));
    const float scaled_width  = float(original_width)  * unwarp.scale;
    const float scaled_height = float(original_height) * unwarp.scale;
    unwarp.offset_x = (float(network_width)  - scaled_width)  * 0.5f;
    unwarp.offset_y = (float(network_height) - scaled_height) * 0.5f;
    unwarp.original_width  = float(original_width);
    unwarp.original_height = float(original_height);
    unwarp.network_width   = float(network_width);
    unwarp.network_height  = float(network_height);
    return unwarp;
}

YoloDetection unwarp_candidate(const float* candidate, const LetterboxUnwarp& unwarp)
{
    const float cx_net_px = candidate[0] * unwarp.network_width;
    const float cy_net_px = candidate[1] * unwarp.network_height;
    const float w_net_px  = candidate[2] * unwarp.network_width;
    const float h_net_px  = candidate[3] * unwarp.network_height;

    YoloDetection detection;
    detection.center_x = clamp((cx_net_px - unwarp.offset_x) / unwarp.scale, 0.0f, unwarp.original_width);
    detection.center_y = clamp((cy_net_px - unwarp.offset_y) / unwarp.scale, 0.0f, unwarp.original_height);
    detection.width    = w_net_px / unwarp.scale;
    detection.height   = h_net_px / unwarp.scale;
    detection.score    = candidate[4];
    detection.class_id = Index(candidate[5]);
    return detection;
}

bool valid_decode_dimensions(Index original_height, Index original_width,
                             Index network_height, Index network_width)
{
    return original_height > 0 && original_width > 0
        && network_height > 0 && network_width > 0;
}

vector<YoloDetection> nms_and_unwarp(vector<array<float, 6>>& candidates,
                                     Index original_height, Index original_width,
                                     Index network_height, Index network_width,
                                     float iou_threshold)
{
    ranges::sort(candidates, greater<>{}, [](const array<float, 6>& c) { return c[4]; });

    vector<array<float, 6>> kept;
    kept.reserve(candidates.size());
    for (const array<float, 6>& candidate : candidates)
    {
        const bool suppressed = ranges::any_of(kept, [&](const array<float, 6>& kept_candidate)
        {
            return Index(kept_candidate[5]) == Index(candidate[5])
                && yolo_box_iou(candidate, kept_candidate) > iou_threshold;
        });
        if (!suppressed) kept.push_back(candidate);
    }

    const LetterboxUnwarp unwarp = make_letterbox_unwarp(original_height, original_width,
                                                         network_height, network_width);

    vector<YoloDetection> detections(kept.size());
    ranges::transform(kept, detections.begin(),
                      [&unwarp](const array<float, 6>& candidate)
                      { return unwarp_candidate(candidate.data(), unwarp); });

    return detections;
}

}

vector<YoloDetection> decode_yolo_fpn_detections(const vector<YoloFpnHead>& heads,
                                                 Index original_height,
                                                 Index original_width,
                                                 Index network_height,
                                                 Index network_width,
                                                 float confidence_threshold,
                                                 float iou_threshold)
{
    if (!valid_decode_dimensions(original_height, original_width, network_height, network_width))
        throw runtime_error("decode_yolo_fpn_detections: dimensions must be positive.");

    vector<array<float, 6>> candidates;

    for (const YoloFpnHead& head : heads)
    {
        if (head.data.empty() || head.grid_size <= 0 || head.boxes_per_cell <= 0
        ||  head.classes_number <= 0)
            continue;

        const Index values_per_box = 5 + head.classes_number;
        const Index channels = head.boxes_per_cell * values_per_box;
        throw_if(head.data.size() < size_t(head.grid_size * head.grid_size * channels),
                 "decode_yolo_fpn_detections: head data is smaller than its shape.");
        const float inv_grid = 1.0f / float(head.grid_size);

        for (Index row = 0; row < head.grid_size; ++row)
            for (Index col = 0; col < head.grid_size; ++col)
            {
                const Index cell = (row * head.grid_size + col) * channels;

                for (Index box = 0; box < head.boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    Index best_class = 0;
                    float best_probability = head.data[base + 5];
                    for (Index c = 1; c < head.classes_number; ++c)
                        if (head.data[base + 5 + c] > best_probability)
                        {
                            best_probability = head.data[base + 5 + c];
                            best_class = c;
                        }

                    const float score = head.data[base + 4] * best_probability;
                    if (score < confidence_threshold) continue;

                    candidates.push_back({
                        (float(col) + head.data[base + 0]) * inv_grid,
                        (float(row) + head.data[base + 1]) * inv_grid,
                        head.data[base + 2],
                        head.data[base + 3],
                        score,
                        float(best_class)
                    });
                }
            }
    }

    return nms_and_unwarp(candidates, original_height, original_width,
                          network_height, network_width, iou_threshold);
}

vector<YoloDetection> decode_yolo_v8_fpn_detections(const vector<YoloFpnHead>& heads,
                                                     Index original_height,
                                                     Index original_width,
                                                     Index network_height,
                                                     Index network_width,
                                                     float confidence_threshold,
                                                     float iou_threshold,
                                                     Index reg_max)
{
    if (!valid_decode_dimensions(original_height, original_width, network_height, network_width))
        throw runtime_error("decode_yolo_v8_fpn_detections: dimensions must be positive.");

    const Index box_ch = 4 * max(reg_max, Index(1));

    vector<array<float, 6>> candidates;

    for (const YoloFpnHead& head : heads)
    {
        if (head.data.empty() || head.grid_size <= 0 || head.classes_number <= 0) continue;

        const Index G  = head.grid_size;
        const Index ch = box_ch + head.classes_number;
        throw_if(head.data.size() < size_t(G * G * ch),
                 "decode_yolo_v8_fpn_detections: head data is smaller than its shape.");
        const float inv_grid = 1.0f / float(G);

        for (Index row = 0; row < G; ++row)
            for (Index col = 0; col < G; ++col)
            {
                const Index base = (row * G + col) * ch;

                Index best_class = 0;
                float best_prob = head.data[base + box_ch];
                for (Index c = 1; c < head.classes_number; ++c)
                    if (head.data[base + box_ch + c] > best_prob)
                    {
                        best_prob = head.data[base + box_ch + c];
                        best_class = c;
                    }

                if (best_prob < confidence_threshold) continue;

                float pred_cx, pred_cy, pred_w, pred_h;
                if (reg_max > 1)
                {
                    const float cell_cx = (float(col) + 0.5f) * inv_grid;
                    const float cell_cy = (float(row) + 0.5f) * inv_grid;
                    const float d_l = dfl_decode(head.data.data() + base, reg_max);
                    const float d_t = dfl_decode(head.data.data() + base + reg_max, reg_max);
                    const float d_r = dfl_decode(head.data.data() + base + 2 * reg_max, reg_max);
                    const float d_b = dfl_decode(head.data.data() + base + 3 * reg_max, reg_max);
                    pred_cx = cell_cx + (d_r - d_l) * inv_grid * 0.5f;
                    pred_cy = cell_cy + (d_b - d_t) * inv_grid * 0.5f;
                    pred_w  = (d_l + d_r) * inv_grid;
                    pred_h  = (d_t + d_b) * inv_grid;
                }
                else
                {
                    pred_cx = (float(col) + head.data[base + 0]) * inv_grid;
                    pred_cy = (float(row) + head.data[base + 1]) * inv_grid;
                    pred_w  = head.data[base + 2];
                    pred_h  = head.data[base + 3];
                }

                candidates.push_back({pred_cx, pred_cy, pred_w, pred_h, best_prob, float(best_class)});
            }
    }

    return nms_and_unwarp(candidates, original_height, original_width,
                          network_height, network_width, iou_threshold);
}

vector<YoloDetection> decode_yolo_detections(span<const float> nms_output,
                                             Index original_height,
                                             Index original_width,
                                             Index network_height,
                                             Index network_width)
{
    if (!valid_decode_dimensions(original_height, original_width, network_height, network_width))
        throw runtime_error("decode_yolo_detections: dimensions must be positive.");
    throw_if(nms_output.size() % 6 != 0,
             "decode_yolo_detections: output size must be a multiple of 6.");

    const Index max_boxes = Index(nms_output.size() / 6);

    const LetterboxUnwarp unwarp = make_letterbox_unwarp(original_height, original_width,
                                                         network_height, network_width);

    vector<YoloDetection> detections;
    detections.reserve(max_boxes);

    for (Index i = 0; i < max_boxes; ++i)
    {
        const float* row = nms_output.data() + i * 6;

        if (row[4] <= 0.0f) break;

        detections.push_back(unwarp_candidate(row, unwarp));
    }

    return detections;
}


void YoloDataset::enable_device_residency()
{
    if (augmentation_policy.enabled)
    {
        if (is_device_resident()) disable_device_residency();
        return;
    }

    upload_device_samples();
}

void YoloDataset::set_augmentation_policy(const AugmentationPolicy& new_policy)
{
    throw_if(!isfinite(new_policy.jitter) || new_policy.jitter < 0.0f || new_policy.jitter >= 0.5f,
             "YOLO augmentation jitter must be finite and in [0, 0.5) so crop dimensions stay positive.");
    validate_augmentation_scale("exposure", new_policy.exposure);
    validate_augmentation_scale("saturation", new_policy.saturation);
    throw_if(!isfinite(new_policy.hue) || new_policy.hue < 0.0f || new_policy.hue > 0.5f,
             "YOLO augmentation hue must be finite and in [0, 0.5] (fractions of a hue cycle).");
    throw_if(new_policy.cutout_count < 0,
             "Cutout count must be nonnegative.");
    throw_if(!isfinite(new_policy.cutout_max_size_ratio)
             || new_policy.cutout_max_size_ratio < 0.0f || new_policy.cutout_max_size_ratio > 1.0f,
             "Cutout maximum size ratio must be finite and in [0, 1].");
    if (is_device_resident()) disable_device_residency();

    augmentation_policy = new_policy;
    if (v8_mode) update_target_layout();
}


void YoloDataset::set_multi_scale_heads(const vector<Index>& grid_sizes,
                                        const vector<vector<array<float, 2>>>& per_head_anchors)
{
    throw_if(grid_sizes.empty() || grid_sizes.size() != per_head_anchors.size(),
             "YoloDataset::set_multi_scale_heads: head counts must match and be non-zero.");

    const Index per_head = ssize(per_head_anchors[0]);
    throw_if(per_head <= 0,
             "YoloDataset::set_multi_scale_heads: each head needs at least one anchor.");
    for (const auto& a : per_head_anchors)
        throw_if(ssize(a) != per_head,
                 "YoloDataset::set_multi_scale_heads: all heads must have the same boxes_per_head.");
    for (Index g : grid_sizes)
        throw_if(g <= 0,
                 "YoloDataset::set_multi_scale_heads: grid sizes must be positive.");

    head_grid_sizes = grid_sizes;
    head_anchors = per_head_anchors;
    boxes_per_head = per_head;
    update_target_layout();
}

void YoloDataset::set_v8_mode(bool enabled)
{
    v8_mode = enabled;
    update_target_layout();
}

void YoloDataset::update_target_layout()
{
    disable_device_residency();
    if (v8_mode)
        target_shape = {max(MAX_GT_BOXES, max_boxes_per_image
            * (augmentation_policy.enabled && augmentation_policy.mosaic ? 4 : 1)) * 5};
    else if (is_multi_scale())
        target_shape = {transform_reduce(head_grid_sizes.begin(), head_grid_sizes.end(), Index(0), plus<>{},
            [&](Index g) { return g * g * boxes_per_head * (5 + classes_number); })};
    else
        target_shape = {grid_size, grid_size, boxes_per_cell * (5 + classes_number)};

    target_record_floats = target_shape.size();
    if (variables.size() >= 2)
        variables[1].features = target_record_floats;
}


struct MosaicParams
{
    Index companions[3];
    float cx_frac;
    float cy_frac;
};

MosaicParams derive_mosaic_params(uint64_t epoch_seed, uint64_t sample_index,
                                  span<const Index> training_indices)
{
    // Salted re-mix of the per-sample seed: the mosaic draws from its own stream,
    // independent of the one sample_augmentation_transform consumes.
    RandomStream random(derive_random_seed(derive_random_seed(epoch_seed, sample_index),
                                           0xCAFEBABEDEADBEEFull));

    MosaicParams params{};

    for (Index& companion : params.companions)
        companion = training_indices[size_t(random.next() % training_indices.size())];

    params.cx_frac = random.uniform(0.3f, 0.7f);
    params.cy_frac = random.uniform(0.3f, 0.7f);

    return params;
}

struct MosaicQuad
{
    Index si;
    Index dst_x, dst_y, qw, qh;
};

array<MosaicQuad, 4> compute_mosaic_layout(uint64_t epoch_seed, Index sample_index,
                                           span<const Index> training_indices, Index height, Index width)
{
    const MosaicParams mp = derive_mosaic_params(epoch_seed, uint64_t(sample_index), training_indices);
    const Index cut_x = max<Index>(1, min(width - 1, Index(mp.cx_frac * float(width))));
    const Index cut_y = max<Index>(1, min(height - 1, Index(mp.cy_frac * float(height))));

    return {{
        {sample_index,         0,     0,         cut_x,          cut_y},
        {mp.companions[0], cut_x,     0, width - cut_x,          cut_y},
        {mp.companions[1],     0, cut_y,         cut_x, height - cut_y},
        {mp.companions[2], cut_x, cut_y, width - cut_x, height - cut_y},
    }};
}

struct YoloDataset::AugmentationBatch
{
    AugmentationPolicy policy;
    uint64_t seed;
    span<const Index> training_indices;
};

span<const Index> YoloDataset::get_mosaic_training_indices() const
{
    const std::lock_guard lock(mosaic_indices_mutex);
    if (!mosaic_indices_valid)
    {
        mosaic_training_indices = get_sample_indices(SampleRole::Training);
        mosaic_indices_valid = true;
    }
    return mosaic_training_indices;
}

YoloDataset::AugmentationBatch YoloDataset::prepare_augmentation_batch(
    const vector<Index>& sample_indices, FillMode mode) const
{
    AugmentationPolicy policy = augmentation_policy;
    policy.enabled = mode == FillMode::Training && policy.enabled;
    policy.mosaic = policy.enabled && policy.mosaic;
    if (policy.enabled) validate_augmentation_samples(sample_indices);
    return {policy, get_augmentation_seed(),
            policy.mosaic ? get_mosaic_training_indices() : span<const Index>{}};
}

void YoloDataset::read_image_record(Index sample, uint8_t* destination) const
{
    throw_if(sample < 0 || sample >= samples_number, "YoloDataset input sample index is out of range.");
    if (storage_mode == StorageMode::Matrix)
        copy_n(images_ram.data() + size_t(sample) * size_t(cache_image_record_bytes),
               size_t(cache_image_record_bytes), destination);
    else
        image_cache_reader.read_at(span(destination, size_t(cache_image_record_bytes)),
                                   sizeof(YoloImageCacheHeader)
                                   + uint64_t(sample) * uint64_t(cache_image_record_bytes));
}

optional<ImageTransformBatch> YoloDataset::get_gpu_image_transform(Index samples) const
{
    ImageTransformBatch transform;
    transform.kind = ImageTransformBatch::Kind::Crop;
    transform.samples = samples;
    transform.enabled = augmentation_policy.enabled;
    transform.mosaic = transform.enabled && augmentation_policy.mosaic;
    transform.sources = samples * (transform.mosaic ? 4 : 1);
    transform.height = cache_input_shape[0];
    transform.width = cache_input_shape[1];
    transform.channels = cache_input_shape[2];
    transform.output_height = input_shape[0];
    transform.output_width = input_shape[1];
    if (transform.enabled && augmentation_policy.cutout_max_size_ratio > 0.0f)
        transform.cutouts = augmentation_policy.cutout_count;
    return transform;
}

void YoloDataset::fill_batch(Batch& batch, const vector<Index>& sample_indices,
                              const FeatureSelection& features, FillMode mode) const
{
    batch.image_transform.reset();
    if (!batch.uses_cuda() || mode != FillMode::Training
        || (!augmentation_policy.enabled && !batch.pack_image_input))
        return Dataset::fill_batch(batch, sample_indices, features, mode);
    throw_if(ssize(sample_indices) != batch.batch_size,
             "fill_batch sample count does not match the batch size.");
    const auto augmentation = prepare_augmentation_batch(sample_indices, mode);
    const auto transform = *get_gpu_image_transform(batch.batch_size);
    const auto parameters = batch.prepare_image_transform(transform);
    const auto cutouts = batch.image_cutouts();
    vector<Index> sources;
    unordered_map<Index, Index> source_indices;
    const auto source_slot = [&](Index sample)
    {
        const auto [entry, added] = source_indices.try_emplace(sample, ssize(sources));
        if (added) sources.push_back(sample);
        return entry->second;
    };
    auto color_policy = augmentation.policy;
    color_policy.jitter = 0.0f;
    color_policy.flip = false;
    for (Index i = 0; i < batch.batch_size; ++i)
    {
        auto& parameter = parameters[size_t(i)];
        const Index sample = sample_indices[size_t(i)];
        if (transform.enabled)
            parameter.crop = sample_augmentation_transform(augmentation.seed, uint64_t(sample), augmentation.policy);
        if (transform.mosaic)
        {
            const auto quads = compute_mosaic_layout(augmentation.seed, sample, augmentation.training_indices,
                                                     transform.height, transform.width);
            parameter.cut_x = quads[0].qw;
            parameter.cut_y = quads[0].qh;
            for (size_t q = 0; q < quads.size(); ++q)
            {
                parameter.sources[q] = source_slot(quads[q].si);
                const auto color = sample_augmentation_transform(augmentation.seed, uint64_t(quads[q].si), color_policy);
                parameter.colors[q] = {color.exposure_mul, color.saturation_mul, color.hue_shift};
            }
        }
        else
            parameter.sources[0] = source_slot(sample);
        RandomStream random(derive_random_seed(derive_random_seed(augmentation.seed, uint64_t(sample)),
                                               0x4355544F5554ull));
        const Index max_side = max<Index>(1, Index(double(max(transform.height, transform.width))
                                                   * augmentation.policy.cutout_max_size_ratio));
        for (Index n = 0; n < transform.cutouts; ++n)
            cutouts[size_t(i * transform.cutouts + n)] =
                sample_cutout(random, transform.width, transform.height, max_side);
    }
    batch.image_transform->sources = ssize(sources);
    for (size_t i = 0; i < sources.size(); ++i)
        read_image_record(sources[i], batch.input.host.as<uint8_t>() + i * size_t(cache_image_record_bytes));
    fill_targets(sample_indices, features.targets, batch.target.host.as<float>(), mode);
}

void YoloDataset::fill_inputs(const vector<Index>& sample_indices,
                              const vector<Index>&,
                              float* input_data,
                              FillMode mode,
                              ColumnContiguity) const
{
    const Index batch_size = ssize(sample_indices);
    const float scale = 1.0f / 255.0f;
    const Index output_values = input_shape.size();

    const auto augmentation = prepare_augmentation_batch(sample_indices, mode);
    const auto& policy = augmentation.policy;
    string omp_error;

    const bool resize_needed = (input_shape[0] != cache_input_shape[0])
                            || (input_shape[1] != cache_input_shape[1]);

    const int workers = max(1, min(omp_get_max_threads(), to_int(batch_size)));

    #pragma omp parallel num_threads(workers)
    {
        vector<uint8_t> pixels(static_cast<size_t>(cache_image_record_bytes));
        vector<uint8_t> augmented(policy.enabled ? size_t(cache_image_record_bytes) : 0);
        vector<uint8_t> resized(resize_needed ? size_t(output_values) : 0);
        vector<uint8_t> mosaic_source(policy.mosaic ? size_t(cache_image_record_bytes) : 0);

        #pragma omp for schedule(dynamic)
        for (Index i = 0; i < batch_size; ++i)
        {
            try
            {
                const Index sample_index = sample_indices[size_t(i)];
                throw_if(sample_index < 0 || sample_index >= samples_number,
                         "YoloDataset input sample index is out of range.");

                if (policy.mosaic)
                {
                    const Index H = cache_input_shape[0];
                    const Index W = cache_input_shape[1];
                    const Index C = cache_input_shape[2];
                    const array<MosaicQuad, 4> quads =
                        compute_mosaic_layout(augmentation.seed, sample_index, augmentation.training_indices, H, W);

                    AugmentationPolicy color_policy = policy;
                    color_policy.jitter = 0.0f;
                    color_policy.flip = false;

                    for (const MosaicQuad& q : quads)
                    {
                        read_image_record(q.si, mosaic_source.data());

                        const AugmentationTransform transform = sample_augmentation_transform(
                            augmentation.seed, uint64_t(q.si), color_policy);
                        apply_color_jitter(mosaic_source.data(), H, W, C, transform);

                        resize_bilinear_into(mosaic_source.data(), H, W,
                                             pixels.data(), W,
                                             q.dst_x, q.dst_y, q.qw, q.qh, C);
                    }
                }
                else
                    read_image_record(sample_index, pixels.data());

                const uint8_t* image_bytes = pixels.data();
                if (policy.enabled)
                {
                    const AugmentationTransform transform = sample_augmentation_transform(
                        augmentation.seed, uint64_t(sample_index), policy);

                    apply_geometric_to_image(pixels.data(), augmented.data(),
                                             cache_input_shape[0], cache_input_shape[1],
                                             cache_input_shape[2], transform);
                    if (!policy.mosaic)
                        apply_color_jitter(augmented.data(),
                                           cache_input_shape[0], cache_input_shape[1],
                                           cache_input_shape[2], transform);
                    image_bytes = apply_cutout(augmented.data(), cache_input_shape, policy, augmentation.seed, sample_index);
                }

                if (resize_needed)
                {
                    bilinear_resize_uint8(image_bytes,
                                          cache_input_shape[0], cache_input_shape[1],
                                          resized.data(),
                                          input_shape[0], input_shape[1],
                                          input_shape[2]);
                    image_bytes = resized.data();
                }

                Map<Array<float, Dynamic, 1>>(
                    input_data + i * output_values, output_values) =
                    Map<const Array<uint8_t, Dynamic, 1>>(
                        image_bytes, output_values).cast<float>() * scale;
            }
            catch (const exception& e)
            {
                #pragma omp critical
                { if (omp_error.empty()) omp_error = e.what(); }
            }
        }
    }

    throw_if(!omp_error.empty(),
             omp_error);
}

void YoloDataset::fill_targets(const vector<Index>& sample_indices,
                               const vector<Index>&,
                               float* target_data,
                               FillMode mode,
                               ColumnContiguity) const
{
    const Index batch_size = ssize(sample_indices);

    const auto augmentation = prepare_augmentation_batch(sample_indices, mode);
    const auto& policy = augmentation.policy;
    const bool matrix_storage = storage_mode == StorageMode::Matrix;

    const bool grid_changed = (grid_size != cache_grid_size);
    const bool reencode = policy.enabled || grid_changed || is_multi_scale() || v8_mode;

    string omp_error;

    const auto read_sample_boxes = [&](Index sample_index, vector<Box>& output,
                                       vector<YoloBoxRecord>& records)
    {
        throw_if(sample_index < 0 || sample_index >= samples_number,
                 "YoloDataset box sample index is out of range.");

        const uint64_t begin = boxes_offsets[size_t(sample_index)];
        const uint64_t end = boxes_offsets[size_t(sample_index) + 1];
        throw_if(end < begin, "YoloDataset box cache offsets are invalid.");

        const size_t count = size_t(end - begin);
        output.resize(count);
        records.resize(count);
        if (count == 0) return;

        boxes_cache_reader.read_at(span(records),
                                   boxes_data_offset + begin * sizeof(YoloBoxRecord));
        for (size_t box = 0; box < count; ++box)
            output[box] = {records[box].class_id, records[box].x, records[box].y,
                           records[box].w, records[box].h};
    };

    const int workers = max(1, min(omp_get_max_threads(), to_int(batch_size)));

    #pragma omp parallel num_threads(workers)
    {
        vector<Box> boxes;
        vector<Box> quad_boxes;
        vector<YoloBoxRecord> box_records;

        #pragma omp for
        for (Index i = 0; i < batch_size; ++i)
        {
            try
            {
                const Index sample_index = sample_indices[size_t(i)];
                throw_if(sample_index < 0 || sample_index >= samples_number,
                         "YoloDataset target sample index is out of range.");

                if (reencode)
                {
                    float* const target_ptr = target_data + i * target_record_floats;

                    if (policy.mosaic)
                    {
                        const Index H = cache_input_shape[0];
                        const Index W = cache_input_shape[1];
                        const array<MosaicQuad, 4> quads =
                            compute_mosaic_layout(augmentation.seed, sample_index, augmentation.training_indices, H, W);

                        boxes.clear();
                        for (const MosaicQuad& q : quads)
                        {
                            read_sample_boxes(q.si, quad_boxes, box_records);
                            append_mosaic_boxes(quad_boxes,
                                                {float(q.dst_x) / float(W), float(q.dst_y) / float(H)},
                                                {float(q.qw) / float(W), float(q.qh) / float(H)}, boxes);
                        }
                    }
                    else
                        read_sample_boxes(sample_index, boxes, box_records);

                    if (policy.enabled)
                    {
                        const AugmentationTransform transform = sample_augmentation_transform(
                            augmentation.seed, uint64_t(sample_index), policy);
                        apply_geometric_to_boxes(boxes, transform);
                    }

                    if (v8_mode)
                        make_target_v8_gtlist(boxes, classes_number, target_ptr, target_record_floats / 5);
                    else if (is_multi_scale())
                        make_target_multi_scale(boxes, head_anchors, head_grid_sizes,
                                                boxes_per_head, classes_number, target_ptr);
                    else
                        make_target(boxes, anchors, grid_size, boxes_per_cell,
                                    classes_number, target_ptr);
                }
                else if (matrix_storage)
                {
                    copy_n(targets_ram.data()
                               + size_t(sample_index) * size_t(cache_target_record_floats),
                           cache_target_record_floats,
                           target_data + i * target_record_floats);
                }
                else
                {
                    target_cache_reader.read_at(
                        span(target_data + i * target_record_floats,
                             size_t(cache_target_record_floats)),
                        target_data_offset
                            + uint64_t(sample_index)
                            * uint64_t(cache_target_record_floats) * sizeof(float));
                }
            }
            catch (const exception& e)
            {
                #pragma omp critical
                { if (omp_error.empty()) omp_error = e.what(); }
            }
        }
    }

    throw_if(!omp_error.empty(),
             omp_error);
}


}
