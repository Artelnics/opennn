//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/dataset/yolo_dataset_internal.h"

#include <utility>

#include "opennn/core/io_utilities.h"
#include "opennn/core/json.h"
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






struct AugmentationTransform
{
    float crop_left;
    float crop_top;
    float crop_right;
    float crop_bottom;
    bool flip;
    float exposure_mul;
    float saturation_mul;
    float hue_shift;
};

uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

uint64_t batch_augmentation_seed(const vector<Index>& sample_indices, bool augment)
{
    if (!augment) return 0;

    uint64_t seed = 14695981039346656037ULL;
    for (const Index sample_index : sample_indices)
    {
        seed ^= uint64_t(sample_index);
        seed *= 1099511628211ULL;
    }
    return seed ? seed : 1;
}

AugmentationTransform sample_augmentation_transform(uint64_t epoch_counter,
                                                    uint64_t sample_index,
                                                    const YoloDataset::AugmentationPolicy& policy)
{
    auto rand_unit = [](uint64_t& rng_state) -> float
    {
        rng_state = splitmix64(rng_state);
        return float(rng_state >> 40) / float(1u << 24);
    };

    auto rand_signed = [&](uint64_t& rng_state, float range) -> float
    {
        return (rand_unit(rng_state) * 2.0f - 1.0f) * range;
    };

    auto rand_scale = [&](uint64_t& rng_state, float max_scale) -> float
    {
        const float r = rand_unit(rng_state);
        const float t = (r * 2.0f - 1.0f) * log(max_scale);
        return exp(t);
    };

    uint64_t state = splitmix64(epoch_counter * 0x9E3779B97F4A7C15ull + sample_index);

    AugmentationTransform transform{};
    transform.crop_left   = rand_signed(state, policy.jitter);
    transform.crop_right  = rand_signed(state, policy.jitter);
    transform.crop_top    = rand_signed(state, policy.jitter);
    transform.crop_bottom = rand_signed(state, policy.jitter);
    transform.flip = policy.flip && (rand_unit(state) < 0.5f);
    transform.exposure_mul = rand_scale(state, policy.exposure);
    transform.saturation_mul = rand_scale(state, policy.saturation);
    transform.hue_shift = rand_signed(state, policy.hue);
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
    const float source_x1 = (1.0f - transform.crop_right) * original_width;
    const float source_y1 = (1.0f - transform.crop_bottom) * original_height;
    const float crop_width = max(1.0f, source_x1 - source_x0);
    const float crop_height = max(1.0f, source_y1 - source_y0);

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
        const Index sy0 = max<Index>(0, min(src_h - 1, Index(sy_f)));
        const Index sy1 = min(sy0 + 1, src_h - 1);
        const float dy  = sy_f - float(sy0);

        for (Index ox = 0; ox < out_w; ++ox)
        {
            const float sx_f = (float(ox) + 0.5f) * float(src_w) / float(out_w) - 0.5f;
            const Index sx0 = max<Index>(0, min(src_w - 1, Index(sx_f)));
            const Index sx1 = min(sx0 + 1, src_w - 1);
            const float dx  = sx_f - float(sx0);

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
    const float crop_width = max(1e-6f, 1.0f - transform.crop_left - transform.crop_right);
    const float crop_height = max(1e-6f, 1.0f - transform.crop_top - transform.crop_bottom);

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

    ImageDataset::enable_device_residency();
}

void YoloDataset::set_augmentation_policy(const AugmentationPolicy& new_policy)
{
    if (is_device_resident()) disable_device_residency();

    augmentation_policy = new_policy;
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

    const Index values_per_box = 5 + classes_number;
    target_record_floats = transform_reduce(grid_sizes.begin(), grid_sizes.end(), Index(0), plus<>{},
                                            [&](Index g) { return g * g * boxes_per_head * values_per_box; });
    target_shape = {target_record_floats};
}

void YoloDataset::set_v8_mode(bool enabled)
{
    v8_mode = enabled;
    if (!enabled) return;

    target_record_floats = MAX_GT_BOXES * 5;
    target_shape = {target_record_floats};
    if (variables.size() >= 2)
        variables[1].features = target_record_floats;
}


void blit_resized_into_canvas(const uint8_t* src, Index src_h, Index src_w,
                               uint8_t* canvas, Index canvas_w,
                               Index dst_x, Index dst_y, Index qw, Index qh,
                               Index channels)
{
    resize_bilinear_into(src, src_h, src_w, canvas, canvas_w, dst_x, dst_y, qw, qh, channels);
}

struct MosaicParams
{
    Index companions[3];
    float cx_frac;
    float cy_frac;
};

MosaicParams derive_mosaic_params(uint64_t epoch_seed, uint64_t sample_index,
                                  Index samples_number)
{
    uint64_t state = splitmix64(
        splitmix64(epoch_seed * 0x9E3779B97F4A7C15ull + sample_index)
        + 0xCAFEBABEDEADBEEFull);

    auto rand_idx = [&]() -> Index {
        state = splitmix64(state);
        return Index(state % uint64_t(samples_number));
    };
    auto rand_frac = [&](float lo, float hi) -> float {
        state = splitmix64(state);
        return lo + (hi - lo) * float(state >> 40) / float(1ull << 24);
    };

    MosaicParams mp{};
    mp.companions[0] = rand_idx();
    mp.companions[1] = rand_idx();
    mp.companions[2] = rand_idx();
    mp.cx_frac = rand_frac(0.3f, 0.7f);
    mp.cy_frac = rand_frac(0.3f, 0.7f);
    return mp;
}

struct MosaicQuad
{
    Index si;
    Index dst_x, dst_y, qw, qh;
};

array<MosaicQuad, 4> compute_mosaic_layout(uint64_t epoch_seed, Index sample_index,
                                           Index samples_number, Index height, Index width)
{
    const MosaicParams mp = derive_mosaic_params(epoch_seed, uint64_t(sample_index), samples_number);
    const Index cut_x = max<Index>(1, min(width - 1, Index(mp.cx_frac * float(width))));
    const Index cut_y = max<Index>(1, min(height - 1, Index(mp.cy_frac * float(height))));

    return {{
        {sample_index,         0,     0,         cut_x,          cut_y},
        {mp.companions[0], cut_x,     0, width - cut_x,          cut_y},
        {mp.companions[1],     0, cut_y,         cut_x, height - cut_y},
        {mp.companions[2], cut_x, cut_y, width - cut_x, height - cut_y},
    }};
}

void YoloDataset::fill_inputs(const vector<Index>& sample_indices,
                              const vector<Index>&,
                              float* input_data,
                              FillMode mode,
                              ColumnContiguity) const
{
    const Index batch_size = ssize(sample_indices);
    const float scale = 1.0f / 255.0f;

    const bool augment = mode == FillMode::Training && augmentation_policy.enabled;
    const bool matrix_storage = storage_mode == StorageMode::Matrix;
    const uint64_t epoch_seed = batch_augmentation_seed(sample_indices, augment);

    const AugmentationPolicy policy = augmentation_policy;
    const bool mosaic = augment && policy.mosaic;

    string omp_error;

    const bool resize_needed = (input_shape[0] != cache_input_shape[0])
                            || (input_shape[1] != cache_input_shape[1]);

    auto read_image_record = [&](Index sample, uint8_t* destination)
    {
        if (matrix_storage)
            copy_n(images_ram.data() + size_t(sample) * size_t(cache_image_record_bytes),
                   size_t(cache_image_record_bytes), destination);
        else
            image_cache_reader.read_at(span(destination, size_t(cache_image_record_bytes)),
                                       sizeof(YoloImageCacheHeader)
                                       + uint64_t(sample) * uint64_t(cache_image_record_bytes));
    };

    const int workers = max(1, min(omp_get_max_threads(), to_int(batch_size)));

    #pragma omp parallel num_threads(workers)
    {
        vector<uint8_t> pixels(static_cast<size_t>(cache_image_record_bytes));
        vector<uint8_t> augmented(augment ? size_t(cache_image_record_bytes) : 0);
        vector<uint8_t> resized(resize_needed ? size_t(image_record_bytes) : 0);
        vector<uint8_t> mosaic_source(mosaic ? size_t(cache_image_record_bytes) : 0);

        #pragma omp for schedule(dynamic)
        for (Index i = 0; i < batch_size; ++i)
        {
            try
            {
                const Index sample_index = sample_indices[size_t(i)];
                throw_if(sample_index < 0 || sample_index >= samples_number,
                         "YoloDataset input sample index is out of range.");

                read_image_record(sample_index, pixels.data());

                const uint8_t* image_bytes = pixels.data();

                if (mosaic)
                {
                    const Index H = cache_input_shape[0];
                    const Index W = cache_input_shape[1];
                    const Index C = cache_input_shape[2];
                    const array<MosaicQuad, 4> quads =
                        compute_mosaic_layout(epoch_seed, sample_index, samples_number, H, W);

                    AugmentationPolicy color_policy = policy;
                    color_policy.jitter = 0.0f;
                    color_policy.flip = false;

                    for (const MosaicQuad& q : quads)
                    {
                        read_image_record(q.si, mosaic_source.data());

                        const AugmentationTransform transform = sample_augmentation_transform(
                            epoch_seed, uint64_t(q.si), color_policy);
                        apply_color_jitter(mosaic_source.data(), H, W, C, transform);

                        blit_resized_into_canvas(mosaic_source.data(), H, W,
                                                 augmented.data(), W,
                                                 q.dst_x, q.dst_y, q.qw, q.qh, C);
                    }
                    image_bytes = augmented.data();
                }
                else if (augment)
                {
                    const AugmentationTransform transform = sample_augmentation_transform(
                        epoch_seed, uint64_t(sample_index), policy);

                    apply_geometric_to_image(pixels.data(), augmented.data(),
                                             cache_input_shape[0], cache_input_shape[1],
                                             cache_input_shape[2], transform);
                    apply_color_jitter(augmented.data(),
                                       cache_input_shape[0], cache_input_shape[1],
                                       cache_input_shape[2], transform);
                    image_bytes = augmented.data();
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
                    input_data + i * image_record_bytes, image_record_bytes) =
                    Map<const Array<uint8_t, Dynamic, 1>>(
                        image_bytes, image_record_bytes).cast<float>() * scale;
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

    const bool augment = mode == FillMode::Training && augmentation_policy.enabled;
    const bool matrix_storage = storage_mode == StorageMode::Matrix;
    const uint64_t epoch_seed = batch_augmentation_seed(sample_indices, augment);

    const AugmentationPolicy policy = augmentation_policy;
    const bool mosaic = augment && policy.mosaic;

    const bool grid_changed = (grid_size != cache_grid_size);
    const bool reencode = augment || grid_changed || is_multi_scale() || v8_mode;

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
        vector<Box> mosaic_boxes;
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

                    if (mosaic)
                    {
                        const Index H = cache_input_shape[0];
                        const Index W = cache_input_shape[1];
                        const array<MosaicQuad, 4> quads =
                            compute_mosaic_layout(epoch_seed, sample_index, samples_number, H, W);

                        mosaic_boxes.clear();
                        for (const MosaicQuad& q : quads)
                        {
                            read_sample_boxes(q.si, quad_boxes, box_records);
                            const float qw_frac = float(q.qw) / float(W);
                            const float qh_frac = float(q.qh) / float(H);
                            const float ox_frac = float(q.dst_x) / float(W);
                            const float oy_frac = float(q.dst_y) / float(H);

                            for (const Box& src : quad_boxes)
                            {
                                const float raw_cx = src.x * qw_frac + ox_frac;
                                const float raw_cy = src.y * qh_frac + oy_frac;
                                const float raw_w  = src.w * qw_frac;
                                const float raw_h  = src.h * qh_frac;

                                const float x0 = max(raw_cx - raw_w * 0.5f, ox_frac);
                                const float y0 = max(raw_cy - raw_h * 0.5f, oy_frac);
                                const float x1 = min(raw_cx + raw_w * 0.5f, ox_frac + qw_frac);
                                const float y1 = min(raw_cy + raw_h * 0.5f, oy_frac + qh_frac);

                                if (x1 - x0 < 1e-3f || y1 - y0 < 1e-3f) continue;

                                Box transformed;
                                transformed.class_id = src.class_id;
                                transformed.x = 0.5f * (x0 + x1);
                                transformed.y = 0.5f * (y0 + y1);
                                transformed.w = x1 - x0;
                                transformed.h = y1 - y0;
                                mosaic_boxes.push_back(transformed);
                            }
                        }

                        if (v8_mode)
                            make_target_v8_gtlist(mosaic_boxes, classes_number, target_ptr);
                        else if (is_multi_scale())
                            make_target_multi_scale(mosaic_boxes, head_anchors, head_grid_sizes,
                                                    boxes_per_head, classes_number, target_ptr);
                        else
                            make_target(mosaic_boxes, anchors, grid_size, boxes_per_cell,
                                        classes_number, target_ptr);
                    }
                    else
                    {
                        read_sample_boxes(sample_index, boxes, box_records);

                        if (augment)
                        {
                            const AugmentationTransform transform = sample_augmentation_transform(
                                epoch_seed, uint64_t(sample_index), policy);
                            apply_geometric_to_boxes(boxes, transform);
                        }

                        if (v8_mode)
                            make_target_v8_gtlist(boxes, classes_number, target_ptr);
                        else if (is_multi_scale())
                            make_target_multi_scale(boxes, head_anchors, head_grid_sizes,
                                                    boxes_per_head, classes_number, target_ptr);
                        else
                            make_target(boxes, anchors, grid_size, boxes_per_cell,
                                        classes_number, target_ptr);
                    }
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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
