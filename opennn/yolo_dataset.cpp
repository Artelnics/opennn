//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "yolo_dataset.h"
#include "convolutional_layer.h"
#include "neural_network.h"
#include "image_processing.h"
#include "io_utilities.h"
#include "json.h"
#include "string_utilities.h"

namespace opennn
{

namespace
{

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

constexpr uint32_t YOLO_CACHE_VERSION = 3;  // bumped: anchor-IoU soft objectness targets
constexpr char YOLO_IMAGE_MAGIC[8] = {'O','P','E','N','N','Y','I','M'};
constexpr char YOLO_TARGET_MAGIC[8] = {'O','P','E','N','N','Y','T','G'};
constexpr char YOLO_BOXES_MAGIC[8] = {'O','P','E','N','N','Y','B','X'};

vector<filesystem::path> list_files(const filesystem::path& directory,
                                    bool (*predicate)(const filesystem::path&))
{
    vector<filesystem::path> paths;
    for (const auto& entry : filesystem::directory_iterator(directory))
        if (entry.is_regular_file() && predicate(entry.path()))
            paths.push_back(entry.path());

    ranges::sort(paths);
    return paths;
}

vector<string> read_yolo_classes(const filesystem::path& labels_directory)
{
    // Search the labels directory, then the parent (dataset root) for a .names file.
    vector<filesystem::path> search_dirs = { labels_directory };
    if (labels_directory.has_parent_path())
        search_dirs.push_back(labels_directory.parent_path());

    for (const auto& dir : search_dirs)
    {
        if (!filesystem::is_directory(dir)) continue;
        for (const auto& entry : filesystem::directory_iterator(dir))
        {
            if (!entry.is_regular_file() || entry.path().extension() != ".names")
                continue;

            ifstream file(entry.path());
            throw_if(!file,
                     format("Cannot open YOLO classes file: {}", entry.path().string()));

            vector<string> classes;
            string line;
            while (getline(file, line))
                if (!line.empty())
                    classes.push_back(line);

            if (!classes.empty())
                return classes;
        }
    }

    return {};
}

vector<YoloDataset::Box> read_yolo_boxes(const filesystem::path& label_path)
{
    ifstream file(label_path);
    if (!file)
        return {};

    vector<YoloDataset::Box> boxes;
    string line;

    while (getline(file, line))
    {
        if (line.empty()) continue;

        istringstream stream(line);
        YoloDataset::Box box;
        throw_if(!(stream >> box.class_id >> box.x >> box.y >> box.w >> box.h),
                 format("Invalid YOLO label line in {}: {}", label_path.string(), line));

        if (box.x < 0.0f || box.x > 1.0f || box.y < 0.0f || box.y > 1.0f
        ||  box.w < 0.0f || box.w > 1.0f || box.h < 0.0f || box.h > 1.0f)
            throw runtime_error(format("YOLO label values out of range in {}: {}", label_path.string(), line));

        boxes.push_back(box);
    }

    return boxes;
}

uint64_t hash_anchors(const vector<array<float, 2>>& anchors)
{
    uint64_t hash_value = 1469598103934665603ull;

    auto mix = [&](uint32_t value)
    {
        hash_value ^= value;
        hash_value *= 1099511628211ull;
    };

    mix(uint32_t(anchors.size()));

    for (const auto& anchor : anchors)
    {
        mix(bit_cast<uint32_t>(anchor[0]));
        mix(bit_cast<uint32_t>(anchor[1]));
    }

    return hash_value;
}

uint64_t hash_sources(const filesystem::path& images_dir,
                      const filesystem::path& labels_dir)
{
    uint64_t hash_value = 1469598103934665603ull;

    auto mix_u64 = [&](uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            hash_value ^= (value >> (i * 8)) & 0xff;
            hash_value *= 1099511628211ull;
        }
    };

    auto mix_bytes = [&](const void* data, size_t n)
    {
        const uint8_t* byte_pointer = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < n; ++i)
        {
            hash_value ^= byte_pointer[i];
            hash_value *= 1099511628211ull;
        }
    };

    auto mix_path = [&](const filesystem::path& path)
    {
        const string s = path.filename().string();
        mix_u64(uint64_t(s.size()));
        mix_bytes(s.data(), s.size());

        error_code ec;
        const auto size = filesystem::file_size(path, ec);
        mix_u64(ec ? 0ull : uint64_t(size));

        const auto mtime = filesystem::last_write_time(path, ec);
        if (ec)
            mix_u64(0ull);
        else
            mix_u64(uint64_t(mtime.time_since_epoch().count()));
    };

    vector<filesystem::path> image_paths = list_files(images_dir, is_supported_image_file);
    mix_u64(uint64_t(image_paths.size()));

    for (const auto& image_path : image_paths)
    {
        mix_path(image_path);

        filesystem::path label_path = labels_dir / image_path.filename();
        label_path.replace_extension(".txt");
        if (filesystem::exists(label_path))
            mix_path(label_path);
        else
            mix_u64(0ull);
    }

    return hash_value;
}

float yolo_iou_wh(const array<float, 2>& box, const array<float, 2>& anchor)
{
    const float inter = min(box[0], anchor[0]) * min(box[1], anchor[1]);
    const float area = box[0] * box[1] + anchor[0] * anchor[1] - inter;
    return area > 0.0f ? inter / area : 0.0f;
}

vector<array<float, 2>> calculate_yolo_anchors(const vector<vector<YoloDataset::Box>>& labels,
                                               Index boxes_per_cell)
{
    vector<array<float, 2>> boxes;

    for (const auto& sample : labels)
        for (const auto& box : sample)
            if (box.w > 0.0f && box.h > 0.0f)
                boxes.push_back({box.w, box.h});

    if (boxes.empty())
        return vector<array<float, 2>>(size_t(boxes_per_cell), {0.1f, 0.1f});

    vector<array<float, 2>> anchors(static_cast<size_t>(boxes_per_cell));
    for (Index i = 0; i < boxes_per_cell; ++i)
        anchors[size_t(i)] = boxes[size_t(i % ssize(boxes))];

    vector<Index> assignments(boxes.size());

    for (Index iteration = 0; iteration < 100; ++iteration)
    {
        bool changed = false;

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            float best_iou = -1.0f;
            Index best_anchor = 0;

            for (Index j = 0; j < boxes_per_cell; ++j)
            {
                const float iou = yolo_iou_wh(boxes[i], anchors[size_t(j)]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_anchor = j;
                }
            }

            changed = changed || assignments[i] != best_anchor;
            assignments[i] = best_anchor;
        }

        vector<array<float, 2>> sums(size_t(boxes_per_cell), {0.0f, 0.0f});
        vector<Index> counts(size_t(boxes_per_cell), 0);

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const Index a = assignments[i];
            sums[size_t(a)][0] += boxes[i][0];
            sums[size_t(a)][1] += boxes[i][1];
            counts[size_t(a)]++;
        }

        for (Index i = 0; i < boxes_per_cell; ++i)
            if (counts[size_t(i)] > 0)
                anchors[size_t(i)] = {
                    sums[size_t(i)][0] / float(counts[size_t(i)]),
                    sums[size_t(i)][1] / float(counts[size_t(i)])
                };

        if (!changed) break;
    }

    return anchors;
}

Tensor3 letterbox_image(const Tensor3& image,
                        Index target_height,
                        Index target_width,
                        float& scale,
                        Index& offset_x,
                        Index& offset_y)
{
    const Index original_height = image.dimension(0);
    const Index original_width = image.dimension(1);
    const Index channels = image.dimension(2);

    scale = min(float(target_width) / float(original_width),
                float(target_height) / float(original_height));

    const Index scaled_width = max<Index>(1, Index(round(float(original_width) * scale)));
    const Index scaled_height = max<Index>(1, Index(round(float(original_height) * scale)));

    offset_x = (target_width - scaled_width) / 2;
    offset_y = (target_height - scaled_height) / 2;

    Tensor3 output(target_height, target_width, channels);
    output.setZero();

    const Tensor3 resized = resize_image(image, scaled_height, scaled_width);

    output.slice(array<Index, 3>{offset_y, offset_x, 0},
                 array<Index, 3>{scaled_height, scaled_width, channels}) = resized;

    return output;
}

void adjust_boxes_to_letterbox(vector<YoloDataset::Box>& boxes,
                               Index original_height,
                               Index original_width,
                               Index target_height,
                               Index target_width,
                               float scale,
                               Index offset_x,
                               Index offset_y)
{
    for (auto& box : boxes)
    {
        const float x_abs = box.x * float(original_width);
        const float y_abs = box.y * float(original_height);
        const float w_abs = box.w * float(original_width);
        const float h_abs = box.h * float(original_height);

        box.x = (x_abs * scale + float(offset_x)) / float(target_width);
        box.y = (y_abs * scale + float(offset_y)) / float(target_height);
        box.w = w_abs * scale / float(target_width);
        box.h = h_abs * scale / float(target_height);
    }
}

Index best_anchor_for_box(const YoloDataset::Box& box,
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

struct AugmentationParams
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

struct AugmentationConfig
{
    float jitter = 0.2f;
    float exposure = 1.5f;
    float saturation = 1.5f;
    float hue = 0.1f;
    bool flip = true;
};

AugmentationParams derive_augmentation_params(uint64_t epoch_counter,
                                              uint64_t sample_index,
                                              const AugmentationConfig& cfg)
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

    AugmentationParams p{};
    p.crop_left   = rand_signed(state, cfg.jitter);
    p.crop_right  = rand_signed(state, cfg.jitter);
    p.crop_top    = rand_signed(state, cfg.jitter);
    p.crop_bottom = rand_signed(state, cfg.jitter);
    p.flip = cfg.flip && (rand_unit(state) < 0.5f);
    p.exposure_mul = rand_scale(state, cfg.exposure);
    p.saturation_mul = rand_scale(state, cfg.saturation);
    p.hue_shift = rand_signed(state, cfg.hue);
    return p;
}

void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v)
{
    const float mx = max(max(r, g), b);
    const float mn = min(min(r, g), b);
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
                        const AugmentationParams& p)
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
        h += p.hue_shift;
        s = min(1.0f, max(0.0f, s * p.saturation_mul));
        v = min(1.0f, max(0.0f, v * p.exposure_mul));
        hsv_to_rgb(h, s, v, r, g, b);
        rgb[base + 0] = uint8_t(min(255.0f, max(0.0f, r * 255.0f + 0.5f)));
        rgb[base + 1] = uint8_t(min(255.0f, max(0.0f, g * 255.0f + 0.5f)));
        rgb[base + 2] = uint8_t(min(255.0f, max(0.0f, b * 255.0f + 0.5f)));
    }
}

void apply_geometric_to_image(const uint8_t* src, uint8_t* dst,
                              Index height, Index width, Index channels,
                              const AugmentationParams& p)
{
    const float original_width = float(width);
    const float original_height = float(height);
    const float source_x0 = p.crop_left * original_width;
    const float source_y0 = p.crop_top * original_height;
    const float source_x1 = (1.0f - p.crop_right) * original_width;
    const float source_y1 = (1.0f - p.crop_bottom) * original_height;
    const float crop_width = max(1.0f, source_x1 - source_x0);
    const float crop_height = max(1.0f, source_y1 - source_y0);

    for (Index dy = 0; dy < height; ++dy)
    {
        for (Index dx = 0; dx < width; ++dx)
        {
            const Index out_x = p.flip ? (width - 1 - dx) : dx;
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
                const float v00 = sample(x0, y0, c);
                const float v10 = sample(x1, y0, c);
                const float v01 = sample(x0, y1, c);
                const float v11 = sample(x1, y1, c);
                const float v0 = v00 * (1.0f - ax) + v10 * ax;
                const float v1 = v01 * (1.0f - ax) + v11 * ax;
                const float v = v0 * (1.0f - ay) + v1 * ay;
                dst[(dy * width + out_x) * channels + c] =
                    uint8_t(min(255.0f, max(0.0f, v + 0.5f)));
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

    const float scale_y = (dst_h <= 1) ? 0.0f : float(src_h - 1) / float(dst_h - 1);
    const float scale_x = (dst_w <= 1) ? 0.0f : float(src_w - 1) / float(dst_w - 1);

    for (Index y = 0; y < dst_h; ++y)
    {
        const float sy = float(y) * scale_y;
        const Index y0 = Index(sy);
        const Index y1 = min(y0 + 1, src_h - 1);
        const float dy = sy - float(y0);

        for (Index x = 0; x < dst_w; ++x)
        {
            const float sx = float(x) * scale_x;
            const Index x0 = Index(sx);
            const Index x1 = min(x0 + 1, src_w - 1);
            const float dx = sx - float(x0);

            const uint8_t* p00 = src + (y0 * src_w + x0) * channels;
            const uint8_t* p01 = src + (y0 * src_w + x1) * channels;
            const uint8_t* p10 = src + (y1 * src_w + x0) * channels;
            const uint8_t* p11 = src + (y1 * src_w + x1) * channels;

            uint8_t* dst_pixel = dst + (y * dst_w + x) * channels;

            for (Index c = 0; c < channels; ++c)
            {
                const float top = float(p00[c]) * (1.0f - dx) + float(p01[c]) * dx;
                const float bot = float(p10[c]) * (1.0f - dx) + float(p11[c]) * dx;
                const float v = top * (1.0f - dy) + bot * dy;
                dst_pixel[c] = uint8_t(min(255.0f, max(0.0f, v + 0.5f)));
            }
        }
    }
}

void apply_geometric_to_boxes(vector<YoloDataset::Box>& boxes,
                              const AugmentationParams& p)
{
    const float crop_width = max(1e-6f, 1.0f - p.crop_left - p.crop_right);
    const float crop_height = max(1e-6f, 1.0f - p.crop_top - p.crop_bottom);

    vector<YoloDataset::Box> out;
    out.reserve(boxes.size());

    for (auto box : boxes)
    {
        const float box_x0 = box.x - 0.5f * box.w;
        const float box_y0 = box.y - 0.5f * box.h;
        const float box_x1 = box.x + 0.5f * box.w;
        const float box_y1 = box.y + 0.5f * box.h;

        float nx0 = (box_x0 - p.crop_left) / crop_width;
        float ny0 = (box_y0 - p.crop_top)  / crop_height;
        float nx1 = (box_x1 - p.crop_left) / crop_width;
        float ny1 = (box_y1 - p.crop_top)  / crop_height;

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
        if (p.flip) box.x = 1.0f - box.x;

        out.push_back(box);
    }

    boxes = move(out);
}

void make_target(const vector<YoloDataset::Box>& boxes,
                 const vector<array<float, 2>>& anchors,
                 Index grid_size,
                 Index boxes_per_cell,
                 Index classes_number,
                 float* target)
{
    const Index values_per_box = 5 + classes_number;
    const Index channels = boxes_per_cell * values_per_box;
    fill_n(target, grid_size * grid_size * channels, 0.0f);

    for (const auto& box : boxes)
    {
        if (box.class_id < 0 || box.class_id >= classes_number)
            continue;

        const Index col = min<Index>(grid_size - 1, max<Index>(0, Index(floor(box.x * grid_size))));
        const Index row = min<Index>(grid_size - 1, max<Index>(0, Index(floor(box.y * grid_size))));
        const Index anchor = best_anchor_for_box(box, anchors);
        const float best_iou = yolo_iou_wh({box.w, box.h}, anchors[size_t(anchor)]);
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        target[base + 0] = box.x * grid_size - float(col);
        target[base + 1] = box.y * grid_size - float(row);
        target[base + 2] = box.w;
        target[base + 3] = box.h;
        target[base + 4] = max(best_iou, 0.5f);  // soft anchor-IoU objectness target
        target[base + 5 + box.class_id] = 1.0f;

        // Mark non-best anchors with IoU > 0.5 as ignore (no noobj gradient)
        for (Index j = 0; j < boxes_per_cell; ++j)
        {
            if (j == anchor) continue;
            if (yolo_iou_wh({box.w, box.h}, anchors[size_t(j)]) < 0.5f) continue;
            const Index base_j = (row * grid_size + col) * channels + j * values_per_box;
            if (target[base_j + 4] < 0.5f)
                target[base_j + 4] = -1.0f;
        }
    }
}

void make_target_multi_scale(const vector<YoloDataset::Box>& boxes,
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

        float best_iou = -1.0f;
        size_t best_head = 0;
        Index best_anchor_in_head = 0;
        for (size_t i = 0; i < head_anchors.size(); ++i)
        {
            const vector<array<float, 2>>& anchors_h = head_anchors[i];
            for (Index j = 0; j < ssize(anchors_h); ++j)
            {
                const float iou = yolo_iou_wh({box.w, box.h}, anchors_h[size_t(j)]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_head = i;
                    best_anchor_in_head = j;
                }
            }
        }

        const Index grid_h = head_grid_sizes[best_head];
        const Index col = min<Index>(grid_h - 1, max<Index>(0, Index(floor(box.x * grid_h))));
        const Index row = min<Index>(grid_h - 1, max<Index>(0, Index(floor(box.y * grid_h))));
        const Index base = head_offsets[best_head]
                         + (row * grid_h + col) * head_channels
                         + best_anchor_in_head * values_per_box;

        target[base + 0] = box.x * grid_h - float(col);
        target[base + 1] = box.y * grid_h - float(row);
        target[base + 2] = box.w;
        target[base + 3] = box.h;
        target[base + 4] = max(best_iou, 0.5f);  // soft anchor-IoU objectness target
        target[base + 5 + box.class_id] = 1.0f;

        // Mark non-best anchors with IoU > 0.5 as ignore (no noobj gradient)
        for (size_t i = 0; i < head_anchors.size(); ++i)
        {
            const vector<array<float, 2>>& anchors_i = head_anchors[i];
            for (Index j = 0; j < ssize(anchors_i); ++j)
            {
                if (i == best_head && j == best_anchor_in_head) continue;
                if (yolo_iou_wh({box.w, box.h}, anchors_i[size_t(j)]) < 0.5f) continue;
                const Index grid_i  = head_grid_sizes[i];
                const Index col_i   = min<Index>(grid_i-1, max<Index>(0, Index(floor(box.x * grid_i))));
                const Index row_i   = min<Index>(grid_i-1, max<Index>(0, Index(floor(box.y * grid_i))));
                const Index base_i  = head_offsets[i]
                                    + (row_i * grid_i + col_i) * head_channels
                                    + j * values_per_box;
                if (target[base_i + 4] < 0.5f)
                    target[base_i + 4] = -1.0f;
            }
        }
    }
}

string read_voc_tag(const string& xml, const string& tag, size_t from = 0)
{
    const string open = "<" + tag + ">";
    const string close = "</" + tag + ">";
    const size_t a = xml.find(open, from);
    if (a == string::npos) return {};
    const size_t b = xml.find(close, a + open.size());
    if (b == string::npos) return {};
    string value = xml.substr(a + open.size(), b - a - open.size());
    const size_t s = value.find_first_not_of(" \t\r\n");
    const size_t e = value.find_last_not_of(" \t\r\n");
    return (s == string::npos) ? string{} : value.substr(s, e - s + 1);
}

struct VocBox
{
    string class_name;
    float xmin, ymin, xmax, ymax;
};

struct VocAnnotation
{
    float width = 0.0f;
    float height = 0.0f;
    vector<VocBox> boxes;
};

VocAnnotation parse_voc_xml(const filesystem::path& xml_path)
{
    ifstream file(xml_path);
    throw_if(!file,
             format("Cannot open VOC annotation: {}", xml_path.string()));

    stringstream buffer;
    buffer << file.rdbuf();
    const string xml = buffer.str();

    VocAnnotation annotation;
    const string size_block = read_voc_tag(xml, "size");
    annotation.width  = parse_float(read_voc_tag(size_block, "width"),  "VOC annotation: width");
    annotation.height = parse_float(read_voc_tag(size_block, "height"), "VOC annotation: height");

    throw_if(annotation.width <= 0.0f || annotation.height <= 0.0f,
             format("VOC annotation has invalid size: {}", xml_path.string()));

    size_t cursor = 0;
    const string obj_open = "<object>";
    while ((cursor = xml.find(obj_open, cursor)) != string::npos)
    {
        const size_t obj_end = xml.find("</object>", cursor);
        throw_if(obj_end == string::npos,
                 format("Unterminated <object> in {}", xml_path.string()));

        const string obj = xml.substr(cursor, obj_end - cursor);
        const string bbox = read_voc_tag(obj, "bndbox");
        VocBox box;
        box.class_name = read_voc_tag(obj, "name");
        box.xmin = parse_float(read_voc_tag(bbox, "xmin"), "VOC bndbox: xmin");
        box.ymin = parse_float(read_voc_tag(bbox, "ymin"), "VOC bndbox: ymin");
        box.xmax = parse_float(read_voc_tag(bbox, "xmax"), "VOC bndbox: xmax");
        box.ymax = parse_float(read_voc_tag(bbox, "ymax"), "VOC bndbox: ymax");
        annotation.boxes.push_back(box);

        cursor = obj_end + strlen("</object>");
    }

    return annotation;
}

const vector<string>& voc_class_names()
{
    static const vector<string> names = {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };
    return names;
}

} // namespace

Index YoloDataset::convert_voc_to_yolo(const filesystem::path& voc_root,
                                       const string& image_set,
                                       const filesystem::path& output_labels_dir,
                                       const vector<string>& class_filter)
{
    throw_if(!filesystem::is_directory(voc_root),
             format("VOC root is not a directory: {}", voc_root.string()));

    const filesystem::path image_set_path =
        voc_root / "ImageSets" / "Main" / (image_set + ".txt");
    throw_if(!filesystem::is_regular_file(image_set_path),
             format("VOC image-set file not found: {}",
                    image_set_path.string()));

    const filesystem::path annotations_dir = voc_root / "Annotations";
    throw_if(!filesystem::is_directory(annotations_dir),
             format("VOC Annotations dir not found: {}",
                    annotations_dir.string()));

    filesystem::create_directories(output_labels_dir);

    // Build class→id mapping. With a filter, only the listed classes are kept
    // and IDs are remapped to 0-indexed within the filter.
    const vector<string>& active_classes = class_filter.empty() ? voc_class_names() : class_filter;
    unordered_map<string, Index> class_index;
    for (Index i = 0; i < ssize(active_classes); ++i)
        class_index[active_classes[size_t(i)]] = i;

    ofstream names_file(output_labels_dir / "voc.names");
    throw_if(!names_file,
             format("Cannot write VOC names file in {}",
                    output_labels_dir.string()));
    for (const string& name : active_classes)
        names_file << name << '\n';
    names_file.close();

    ifstream id_file(image_set_path);
    string image_id;
    Index converted = 0;

    while (id_file >> image_id)
    {
        if (image_id.empty()) continue;

        const filesystem::path xml_path = annotations_dir / (image_id + ".xml");
        if (!filesystem::is_regular_file(xml_path))
            continue;

        const VocAnnotation ann = parse_voc_xml(xml_path);

        // Collect surviving boxes (matching the class filter)
        vector<pair<Index, array<float,4>>> kept_boxes;
        for (const VocBox& box : ann.boxes)
        {
            const auto it = class_index.find(box.class_name);
            if (it == class_index.end())
                continue;
            auto clamp01 = [](float v) { return min(1.0f, max(0.0f, v)); };
            const float cx = clamp01(0.5f * (box.xmin + box.xmax) / ann.width);
            const float cy = clamp01(0.5f * (box.ymin + box.ymax) / ann.height);
            const float bw = clamp01((box.xmax - box.xmin) / ann.width);
            const float bh = clamp01((box.ymax - box.ymin) / ann.height);
            kept_boxes.push_back({it->second, {cx, cy, bw, bh}});
        }

        // With a filter, skip images that have no objects of the requested classes
        if (!class_filter.empty() && kept_boxes.empty())
            continue;

        const filesystem::path out_path = output_labels_dir / (image_id + ".txt");
        ofstream out(out_path);
        throw_if(!out, format("Cannot write YOLO label: {}", out_path.string()));
        for (const auto& [id, b] : kept_boxes)
            out << id << ' ' << b[0] << ' ' << b[1] << ' ' << b[2] << ' ' << b[3] << '\n';
        ++converted;
    }

    return converted;
}

namespace
{

float candidate_iou(const array<float, 6>& a, const array<float, 6>& b)
{
    const float a_left = a[0] - 0.5f * a[2];
    const float a_top = a[1] - 0.5f * a[3];
    const float a_right = a[0] + 0.5f * a[2];
    const float a_bottom = a[1] + 0.5f * a[3];

    const float b_left = b[0] - 0.5f * b[2];
    const float b_top = b[1] - 0.5f * b[3];
    const float b_right = b[0] + 0.5f * b[2];
    const float b_bottom = b[1] + 0.5f * b[3];

    const float inter_w = max(0.0f, min(a_right, b_right) - max(a_left, b_left));
    const float inter_h = max(0.0f, min(a_bottom, b_bottom) - max(a_top, b_top));
    const float inter = inter_w * inter_h;
    const float area = a[2] * a[3] + b[2] * b[3] - inter;

    return area > 0.0f ? inter / area : 0.0f;
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
    if (original_height <= 0 || original_width <= 0
    ||  network_height  <= 0 || network_width  <= 0)
        throw runtime_error("decode_yolo_fpn_detections: dimensions must be positive.");

    vector<array<float, 6>> candidates;

    // Each candidate: (cx_norm, cy_norm, w_norm, h_norm, score, class_id).
    // Decoded output from DetectionOperator: x,y are sigmoid([0,1]) cell-relative
    // offsets; w,h are already image-normalized (anchor * exp(raw)).
    for (const YoloFpnHead& head : heads)
    {
        if (!head.data || head.grid_size <= 0 || head.boxes_per_cell <= 0
        ||  head.classes_number <= 0)
            continue;

        const Index values_per_box = 5 + head.classes_number;
        const Index channels = head.boxes_per_cell * values_per_box;
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

    ranges::sort(candidates, greater<>{}, [](const array<float, 6>& c) { return c[4]; });

    vector<array<float, 6>> kept;
    kept.reserve(candidates.size());
    for (const array<float, 6>& candidate : candidates)
    {
        bool suppressed = false;
        for (const array<float, 6>& k : kept)
            if (Index(k[5]) == Index(candidate[5])
            &&  candidate_iou(candidate, k) > iou_threshold)
            {
                suppressed = true;
                break;
            }
        if (!suppressed) kept.push_back(candidate);
    }

    const float scale = min(float(network_width)  / float(original_width),
                            float(network_height) / float(original_height));
    const float scaled_width  = float(original_width)  * scale;
    const float scaled_height = float(original_height) * scale;
    const float offset_x = (float(network_width)  - scaled_width)  * 0.5f;
    const float offset_y = (float(network_height) - scaled_height) * 0.5f;

    vector<YoloDetection> detections;
    detections.reserve(kept.size());
    for (const array<float, 6>& k : kept)
    {
        const float cx_net_px = k[0] * float(network_width);
        const float cy_net_px = k[1] * float(network_height);
        const float w_net_px  = k[2] * float(network_width);
        const float h_net_px  = k[3] * float(network_height);

        YoloDetection detection;
        detection.center_x = clamp((cx_net_px - offset_x) / scale, 0.0f, float(original_width));
        detection.center_y = clamp((cy_net_px - offset_y) / scale, 0.0f, float(original_height));
        detection.width    = w_net_px / scale;
        detection.height   = h_net_px / scale;
        detection.score    = k[4];
        detection.class_id = Index(k[5]);
        detections.push_back(detection);
    }

    return detections;
}

vector<YoloDetection> decode_yolo_detections(const float* nms_output,
                                             Index max_boxes,
                                             Index original_height,
                                             Index original_width,
                                             Index network_height,
                                             Index network_width)
{
    if (original_height <= 0 || original_width <= 0
    ||  network_height  <= 0 || network_width  <= 0)
        throw runtime_error("decode_yolo_detections: dimensions must be positive.");

    const float scale = min(float(network_width)  / float(original_width),
                            float(network_height) / float(original_height));
    const float scaled_width  = float(original_width)  * scale;
    const float scaled_height = float(original_height) * scale;
    const float offset_x = (float(network_width)  - scaled_width)  * 0.5f;
    const float offset_y = (float(network_height) - scaled_height) * 0.5f;

    vector<YoloDetection> detections;
    detections.reserve(max_boxes);

    for (Index i = 0; i < max_boxes; ++i)
    {
        const float* row = nms_output + i * 6;
        const float score = row[4];

        if (score <= 0.0f) break;

        const float cx_net_px = row[0] * float(network_width);
        const float cy_net_px = row[1] * float(network_height);
        const float w_net_px  = row[2] * float(network_width);
        const float h_net_px  = row[3] * float(network_height);

        YoloDetection detection;
        detection.center_x = (cx_net_px - offset_x) / scale;
        detection.center_y = (cy_net_px - offset_y) / scale;
        detection.width    = w_net_px / scale;
        detection.height   = h_net_px / scale;
        detection.score    = score;
        detection.class_id = Index(row[5]);

        detection.center_x = clamp(detection.center_x, 0.0f, float(original_width));
        detection.center_y = clamp(detection.center_y, 0.0f, float(original_height));

        detections.push_back(detection);
    }

    return detections;
}

YoloDataset::YoloDataset(const filesystem::path& new_images_dir,
                         const filesystem::path& new_labels_dir,
                         const Shape& new_input_shape,
                         Index new_grid_size,
                         Index new_boxes_per_cell,
                         const vector<array<float, 2>>& new_anchors)
{
    set(new_images_dir, new_labels_dir, new_input_shape, new_grid_size,
        new_boxes_per_cell, new_anchors);
}

void YoloDataset::set(const filesystem::path& new_images_dir,
                      const filesystem::path& new_labels_dir,
                      const Shape& new_input_shape,
                      Index new_grid_size,
                      Index new_boxes_per_cell,
                      const vector<array<float, 2>>& new_anchors)
{
    throw_if(new_input_shape.rank != 3,
             "YoloDataset: input_shape must be rank 3.");
    throw_if(new_grid_size <= 0 || new_boxes_per_cell <= 0,
             "YoloDataset: grid_size and boxes_per_cell must be positive.");

    images_directory = new_images_dir;
    labels_directory = new_labels_dir;
    data_path = images_directory;
    storage_mode = StorageMode::BinaryFile;
    input_shape = new_input_shape;
    grid_size = new_grid_size;
    boxes_per_cell = new_boxes_per_cell;
    image_record_bytes = input_shape.size();
    class_names = read_yolo_classes(labels_directory);

    image_cache_path = images_directory / ".cache" / "yolo_images.bin";
    target_cache_path = images_directory / ".cache" / "yolo_targets.bin";
    boxes_cache_path = images_directory / ".cache" / "yolo_boxes.bin";
    images_ram.clear();
    targets_ram.clear();

    image_filenames = list_files(images_directory, is_supported_image_file);

    open_or_build_cache(new_anchors);
}

void YoloDataset::set_storage_mode(StorageMode new_storage_mode)
{
    Dataset::set_storage_mode(new_storage_mode);

    if (new_storage_mode == StorageMode::BinaryFile)
    {
        images_ram.clear();
        targets_ram.clear();
    }
}

void YoloDataset::open_or_build_cache(const vector<array<float, 2>>& requested_anchors)
{
    if (try_open_cache(requested_anchors))
        return;

    image_cache_reader.close();
    target_cache_reader.close();
    boxes_cache_reader.close();
    build_cache(requested_anchors);
}

bool YoloDataset::try_open_cache(const vector<array<float, 2>>& requested_anchors)
{
    if (!filesystem::exists(image_cache_path)
    ||  !filesystem::exists(target_cache_path)
    ||  !filesystem::exists(boxes_cache_path))
        return false;

    try
    {
        image_cache_reader.open(image_cache_path);
        target_cache_reader.open(target_cache_path);
        boxes_cache_reader.open(boxes_cache_path);

        YoloImageCacheHeader image_header{};
        YoloTargetCacheHeader target_header{};
        image_cache_reader.read_at(&image_header, sizeof(image_header), 0);
        target_cache_reader.read_at(&target_header, sizeof(target_header), 0);

        if (memcmp(image_header.magic, YOLO_IMAGE_MAGIC, 8) != 0
        ||  memcmp(target_header.magic, YOLO_TARGET_MAGIC, 8) != 0
        ||  image_header.version != YOLO_CACHE_VERSION
        ||  target_header.version != YOLO_CACHE_VERSION)
            return false;

        if (Index(image_header.height) != input_shape[0]
        ||  Index(image_header.width) != input_shape[1]
        ||  Index(image_header.channels) != input_shape[2]
        ||  Index(target_header.grid_size) != grid_size
        ||  Index(target_header.boxes_per_cell) != boxes_per_cell
        ||  image_header.samples != target_header.samples)
            return false;

        if (image_header.sources_hash != hash_sources(images_directory, labels_directory))
            return false;

        vector<array<float, 2>> cached_anchors(static_cast<size_t>(boxes_per_cell));
        target_cache_reader.read_at(cached_anchors.data(),
                                    cached_anchors.size() * sizeof(array<float, 2>),
                                    target_header.anchors_offset);

        const uint64_t cached_hash = hash_anchors(cached_anchors);
        if (cached_hash != target_header.anchors_hash)
            return false;

        if (!requested_anchors.empty() && hash_anchors(requested_anchors) != cached_hash)
            return false;

        const uint64_t expected_image_size = sizeof(YoloImageCacheHeader)
            + image_header.samples * image_header.record_bytes;
        const uint64_t expected_target_size = target_header.targets_offset
            + target_header.samples * target_header.target_floats * sizeof(float);

        if (image_cache_reader.file_size() != expected_image_size
        ||  target_cache_reader.file_size() != expected_target_size)
            return false;

        anchors = move(cached_anchors);
        classes_number = Index(target_header.classes_number);
        if (class_names.empty())
        {
            class_names.resize(size_t(classes_number));
            for (Index i = 0; i < classes_number; ++i)
                class_names[size_t(i)] = to_string(i);
        }

        target_record_floats = Index(target_header.target_floats);
        target_data_offset = target_header.targets_offset;

        YoloBoxesCacheHeader boxes_header{};
        boxes_cache_reader.read_at(&boxes_header, sizeof(boxes_header), 0);
        if (memcmp(boxes_header.magic, YOLO_BOXES_MAGIC, 8) != 0
        ||  boxes_header.version != YOLO_CACHE_VERSION
        ||  boxes_header.samples != image_header.samples)
            return false;

        boxes_offsets.assign(size_t(boxes_header.samples + 1), 0);
        boxes_cache_reader.read_at(boxes_offsets.data(),
                                   boxes_offsets.size() * sizeof(uint64_t),
                                   boxes_header.offsets_byte_offset);
        if (boxes_offsets.empty()
        ||  boxes_offsets.front() != 0
        ||  boxes_offsets.back() != boxes_header.total_boxes)
            return false;

        for (size_t i = 1; i < boxes_offsets.size(); ++i)
            if (boxes_offsets[i] < boxes_offsets[i - 1])
                return false;

        boxes_data_offset = boxes_header.boxes_byte_offset;

        const uint64_t expected_boxes_size = boxes_header.boxes_byte_offset
            + boxes_header.total_boxes * sizeof(YoloBoxRecord);
        if (boxes_cache_reader.file_size() != expected_boxes_size)
            return false;

        setup_metadata(Index(image_header.samples));
        return true;
    }
    catch (const exception&)
    {
        image_cache_reader.close();
        target_cache_reader.close();
        boxes_cache_reader.close();
        return false;
    }
}

void YoloDataset::build_cache(const vector<array<float, 2>>& requested_anchors)
{
    vector<filesystem::path> image_paths = list_files(images_directory, is_supported_image_file);
    throw_if(image_paths.empty(),
             format("YoloDataset: no images found in {}", images_directory.string()));

    vector<vector<Box>> labels(image_paths.size());
    Index max_class_id = -1;

    filesystem::create_directories(image_cache_path.parent_path());

    const filesystem::path image_tmp_path = image_cache_path.string() + ".tmp";
    FileWriter image_writer;
    image_writer.open(image_tmp_path);

    YoloImageCacheHeader image_header{};
    memcpy(image_header.magic, YOLO_IMAGE_MAGIC, 8);
    image_header.version = YOLO_CACHE_VERSION;
    image_header.height = uint32_t(input_shape[0]);
    image_header.width = uint32_t(input_shape[1]);
    image_header.channels = uint32_t(input_shape[2]);
    image_header.samples = uint64_t(image_paths.size());
    image_header.record_bytes = uint64_t(image_record_bytes);
    image_header.sources_hash = hash_sources(images_directory, labels_directory);
    image_writer.write(&image_header, sizeof(image_header));

    vector<uint8_t> pixels(static_cast<size_t>(image_record_bytes));

    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        Tensor3 image = load_image(image_paths[i]);
        // Convert grayscale → RGB by replicating the single channel.
        if (image.dimension(2) == 1 && input_shape[2] == 3)
        {
            Tensor3 rgb(image.dimension(0), image.dimension(1), 3);
            rgb.chip(0, 2) = image.chip(0, 2);
            rgb.chip(1, 2) = image.chip(0, 2);
            rgb.chip(2, 2) = image.chip(0, 2);
            image = std::move(rgb);
        }
        throw_if(image.dimension(2) != input_shape[2],
                 format("YoloDataset: channel mismatch in {} (got {} channels, expected {})",
                        image_paths[i].string(), image.dimension(2), input_shape[2]));

        float scale = 1.0f;
        Index offset_x = 0;
        Index offset_y = 0;
        const Tensor3 prepared = letterbox_image(image, input_shape[0], input_shape[1],
                                                 scale, offset_x, offset_y);

        filesystem::path label_path = labels_directory / image_paths[i].filename();
        label_path.replace_extension(".txt");
        labels[i] = read_yolo_boxes(label_path);
        adjust_boxes_to_letterbox(labels[i], image.dimension(0), image.dimension(1),
                                  input_shape[0], input_shape[1], scale, offset_x, offset_y);

        for (const Box& box : labels[i])
            max_class_id = max(max_class_id, box.class_id);

        Map<Array<uint8_t, Dynamic, 1>>(pixels.data(), image_record_bytes) =
            (Map<const Array<float, Dynamic, 1>>(prepared.data(), image_record_bytes)
                .max(0.0f).min(255.0f) + 0.5f).cast<uint8_t>();

        image_writer.write(pixels.data(), pixels.size());

        if (display && (i % 1000 == 0 || i + 1 == image_paths.size()))
            display_progress_bar(Index(i + 1), Index(image_paths.size()));
    }

    image_writer.finish_with_rename(image_cache_path);

    classes_number = class_names.empty() ? max_class_id + 1 : ssize(class_names);
    throw_if(classes_number <= 0,
             "YoloDataset: cannot infer classes_number.");

    if (class_names.empty())
    {
        class_names.resize(size_t(classes_number));
        for (Index i = 0; i < classes_number; ++i)
            class_names[size_t(i)] = to_string(i);
    }

    anchors = requested_anchors.empty()
        ? calculate_yolo_anchors(labels, boxes_per_cell)
        : requested_anchors;

    throw_if(ssize(anchors) != boxes_per_cell,
             "YoloDataset: anchors size must equal boxes_per_cell.");

    target_record_floats = grid_size * grid_size * boxes_per_cell * (5 + classes_number);

    const filesystem::path target_tmp_path = target_cache_path.string() + ".tmp";
    FileWriter target_writer;
    target_writer.open(target_tmp_path);

    YoloTargetCacheHeader target_header{};
    memcpy(target_header.magic, YOLO_TARGET_MAGIC, 8);
    target_header.version = YOLO_CACHE_VERSION;
    target_header.grid_size = uint32_t(grid_size);
    target_header.boxes_per_cell = uint32_t(boxes_per_cell);
    target_header.classes_number = uint32_t(classes_number);
    target_header.samples = uint64_t(image_paths.size());
    target_header.target_floats = uint64_t(target_record_floats);
    target_header.anchors_hash = hash_anchors(anchors);
    target_header.anchors_offset = sizeof(YoloTargetCacheHeader);
    target_header.targets_offset = target_header.anchors_offset
        + uint64_t(anchors.size() * sizeof(array<float, 2>));

    target_writer.write(&target_header, sizeof(target_header));
    target_writer.write(anchors.data(), anchors.size() * sizeof(array<float, 2>));

    vector<float> target(static_cast<size_t>(target_record_floats));
    for (const auto& sample_boxes : labels)
    {
        make_target(sample_boxes, anchors, grid_size, boxes_per_cell, classes_number, target.data());
        target_writer.write(target.data(), target.size() * sizeof(float));
    }

    target_writer.finish_with_rename(target_cache_path);

    const filesystem::path boxes_tmp_path = boxes_cache_path.string() + ".tmp";
    FileWriter boxes_writer;
    boxes_writer.open(boxes_tmp_path);

    uint64_t total_boxes = 0;
    for (const auto& sample_boxes : labels) total_boxes += sample_boxes.size();

    YoloBoxesCacheHeader boxes_header{};
    memcpy(boxes_header.magic, YOLO_BOXES_MAGIC, 8);
    boxes_header.version = YOLO_CACHE_VERSION;
    boxes_header.samples = uint64_t(image_paths.size());
    boxes_header.total_boxes = total_boxes;
    boxes_header.offsets_byte_offset = sizeof(YoloBoxesCacheHeader);
    boxes_header.boxes_byte_offset = boxes_header.offsets_byte_offset
        + (boxes_header.samples + 1) * sizeof(uint64_t);

    boxes_writer.write(&boxes_header, sizeof(boxes_header));

    vector<uint64_t> offsets(image_paths.size() + 1, 0);
    for (size_t i = 0; i < image_paths.size(); ++i)
        offsets[i + 1] = offsets[i] + labels[i].size();
    boxes_writer.write(offsets.data(), offsets.size() * sizeof(uint64_t));

    for (const auto& sample_boxes : labels)
        for (const auto& box : sample_boxes)
        {
            const YoloBoxRecord rec{int32_t(box.class_id), box.x, box.y, box.w, box.h};
            boxes_writer.write(&rec, sizeof(rec));
        }

    boxes_writer.finish_with_rename(boxes_cache_path);

    image_cache_reader.open(image_cache_path);
    target_cache_reader.open(target_cache_path);
    boxes_cache_reader.open(boxes_cache_path);
    target_data_offset = target_header.targets_offset;
    boxes_data_offset = boxes_header.boxes_byte_offset;
    boxes_offsets = move(offsets);

    setup_metadata(Index(image_paths.size()));

    if (display)
        cout << "\nYOLO cache built (" << samples_number << " samples).\n";
}

void YoloDataset::setup_metadata(Index new_samples_number)
{
    samples_number = new_samples_number;

    cache_input_shape = input_shape;
    cache_grid_size = grid_size;
    cache_image_record_bytes = image_record_bytes;
    cache_target_record_floats = target_record_floats;

    target_shape = {grid_size, grid_size, boxes_per_cell * (5 + classes_number)};

    variables.assign(2, Variable());

    Variable& image_variable = variables[0];
    image_variable.name = "image";
    image_variable.role = VariableRole::Input;
    image_variable.type = VariableType::Numeric;
    image_variable.scaler = ScalerMethod::None;
    image_variable.features = input_shape.size();

    Variable& target_variable = variables[1];
    target_variable.name = "yolo_target";
    target_variable.role = VariableRole::Target;
    target_variable.type = VariableType::Numeric;
    target_variable.scaler = ScalerMethod::None;
    target_variable.features = target_shape.size();

    sample_roles.assign(size_t(samples_number), SampleRole::Training);
    split_samples_random();
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
    Index total_floats = 0;
    for (Index g : grid_sizes)
        total_floats += g * g * boxes_per_head * values_per_box;
    target_record_floats = total_floats;
    target_shape = {target_record_floats};
}

void YoloDataset::read_sample_boxes(Index sample_index, vector<Box>& out) const
{
    throw_if(sample_index < 0 || sample_index >= samples_number,
             "YoloDataset box sample index is out of range.");

    const uint64_t begin = boxes_offsets[size_t(sample_index)];
    const uint64_t end   = boxes_offsets[size_t(sample_index) + 1];
    throw_if(end < begin,
             "YoloDataset box cache offsets are invalid.");

    const uint64_t count = end - begin;

    out.resize(size_t(count));
    if (count == 0) return;

    thread_local vector<YoloBoxRecord> records;
    records.resize(size_t(count));
    boxes_cache_reader.read_at(records.data(), records.size() * sizeof(YoloBoxRecord),
                               boxes_data_offset + begin * sizeof(YoloBoxRecord));

    for (size_t i = 0; i < records.size(); ++i)
    {
        out[i].class_id = records[i].class_id;
        out[i].x = records[i].x;
        out[i].y = records[i].y;
        out[i].w = records[i].w;
        out[i].h = records[i].h;
    }
}

void YoloDataset::load_images_to_ram() const
{
    if (!images_ram.empty() || samples_number == 0 || cache_image_record_bytes == 0) return;

    throw_if(!image_cache_reader.is_open(),
             "YoloDataset::load_images_to_ram: image cache is not open.");

    images_ram.resize(size_t(samples_number) * size_t(cache_image_record_bytes));
    image_cache_reader.read_at(images_ram.data(), images_ram.size(), sizeof(YoloImageCacheHeader));
}

void YoloDataset::load_targets_to_ram() const
{
    if (!targets_ram.empty() || samples_number == 0 || cache_target_record_floats == 0) return;

    throw_if(!target_cache_reader.is_open(),
             "YoloDataset::load_targets_to_ram: target cache is not open.");

    targets_ram.resize(size_t(samples_number) * size_t(cache_target_record_floats));
    target_cache_reader.read_at(targets_ram.data(),
                                targets_ram.size() * sizeof(float),
                                target_data_offset);
}

// Blit a bilinearly-resized source image into a sub-rectangle of a canvas.
// src:        source pixels, shape [src_h, src_w, channels]
// canvas:     destination buffer, shape [canvas_h, canvas_w, channels]
// canvas_w:   width of the full canvas (stride)
// dst_x/y:   top-left corner of the destination rectangle in the canvas
// qw/qh:     width/height of the destination rectangle
void blit_resized_into_canvas(const uint8_t* src, Index src_h, Index src_w,
                               uint8_t* canvas, Index canvas_w,
                               Index dst_x, Index dst_y, Index qw, Index qh,
                               Index channels)
{
    for (Index oy = 0; oy < qh; ++oy)
    {
        const float sy_f = (float(oy) + 0.5f) * float(src_h) / float(qh) - 0.5f;
        const Index sy0 = max<Index>(0, min(src_h - 1, Index(sy_f)));
        const Index sy1 = min(sy0 + 1, src_h - 1);
        const float dy  = sy_f - float(sy0);

        for (Index ox = 0; ox < qw; ++ox)
        {
            const float sx_f = (float(ox) + 0.5f) * float(src_w) / float(qw) - 0.5f;
            const Index sx0 = max<Index>(0, min(src_w - 1, Index(sx_f)));
            const Index sx1 = min(sx0 + 1, src_w - 1);
            const float dx  = sx_f - float(sx0);

            const Index dst_off = ((dst_y + oy) * canvas_w + (dst_x + ox)) * channels;
            for (Index c = 0; c < channels; ++c)
            {
                const float v00 = float(src[(sy0 * src_w + sx0) * channels + c]);
                const float v01 = float(src[(sy0 * src_w + sx1) * channels + c]);
                const float v10 = float(src[(sy1 * src_w + sx0) * channels + c]);
                const float v11 = float(src[(sy1 * src_w + sx1) * channels + c]);
                const float top = v00 * (1.f - dx) + v01 * dx;
                const float bot = v10 * (1.f - dx) + v11 * dx;
                canvas[dst_off + c] = uint8_t(min(255.f, max(0.f, top * (1.f - dy) + bot * dy + 0.5f)));
            }
        }
    }
}

struct MosaicParams
{
    Index companions[3];
    float cx_frac;
    float cy_frac;
};

// Derive mosaic companion indices and cut point from the same epoch/sample seed
// used by derive_augmentation_params, but with a different hash so the two
// streams are independent.
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

void YoloDataset::fill_inputs(const vector<Index>& sample_indices,
                              const vector<Index>&,
                              float* input_data,
                              FillMode mode,
                              int) const
{
    const Index batch_size = ssize(sample_indices);
    const float scale = 1.0f / 255.0f;

    const bool augment = mode == FillMode::Training && augmentation.enabled;
    const bool matrix_storage = storage_mode == StorageMode::Matrix;
    const uint64_t epoch_seed = augment
        ? augmentation_counter.fetch_add(1, memory_order_relaxed) + 1
        : 0;

    if (matrix_storage)
        load_images_to_ram();

    AugmentationConfig cfg = augmentation;

    string omp_error;

    const bool resize_needed = (input_shape[0] != cache_input_shape[0])
                            || (input_shape[1] != cache_input_shape[1]);

    #pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            thread_local vector<uint8_t> pixels;
            thread_local vector<uint8_t> aug_pixels;
            thread_local vector<uint8_t> resized_pixels;
            pixels.resize(size_t(cache_image_record_bytes));

            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= samples_number,
                     "YoloDataset input sample index is out of range.");

            const uint64_t offset = sizeof(YoloImageCacheHeader)
                + uint64_t(sample_index) * uint64_t(cache_image_record_bytes);

            if (matrix_storage)
            {
                copy_n(images_ram.data() + size_t(sample_index) * size_t(cache_image_record_bytes),
                       pixels.size(),
                       pixels.data());
            }
            else
            {
                image_cache_reader.read_at(pixels.data(), pixels.size(), offset);
            }

            const uint8_t* image_bytes = pixels.data();

            if (augment && augmentation.mosaic)
            {
                const Index H = cache_input_shape[0];
                const Index W = cache_input_shape[1];
                const Index C = cache_input_shape[2];
                const MosaicParams mp = derive_mosaic_params(epoch_seed, uint64_t(sample_index), samples_number);
                const Index cut_x = max<Index>(1, min(W - 1, Index(mp.cx_frac * float(W))));
                const Index cut_y = max<Index>(1, min(H - 1, Index(mp.cy_frac * float(H))));

                struct Quad { Index si; Index dst_x, dst_y, qw, qh; };
                const Quad quads[4] = {
                    {sample_index,         0,     0,     cut_x,   cut_y},
                    {mp.companions[0], cut_x,     0, W - cut_x,   cut_y},
                    {mp.companions[1],     0, cut_y,     cut_x, H - cut_y},
                    {mp.companions[2], cut_x, cut_y, W - cut_x, H - cut_y},
                };

                aug_pixels.resize(size_t(H) * size_t(W) * size_t(C));

                // Color-only cfg for per-quadrant jitter (no geometric crop/flip inside mosaic)
                const ::opennn::AugmentationConfig color_cfg{
                    0.0f, cfg.exposure, cfg.saturation, cfg.hue, false
                };

                thread_local vector<uint8_t> mosaic_src;
                thread_local vector<uint8_t> quad_buf;

                for (const Quad& q : quads)
                {
                    mosaic_src.resize(size_t(cache_image_record_bytes));
                    if (matrix_storage)
                        copy_n(images_ram.data() + size_t(q.si) * size_t(cache_image_record_bytes),
                               size_t(cache_image_record_bytes), mosaic_src.data());
                    else
                    {
                        const uint64_t off = sizeof(YoloImageCacheHeader)
                            + uint64_t(q.si) * uint64_t(cache_image_record_bytes);
                        image_cache_reader.read_at(mosaic_src.data(),
                                                   size_t(cache_image_record_bytes), off);
                    }

                    quad_buf.assign(mosaic_src.begin(), mosaic_src.end());
                    const AugmentationParams qp = derive_augmentation_params(
                        epoch_seed, uint64_t(q.si), color_cfg);
                    apply_color_jitter(quad_buf.data(), H, W, C, qp);

                    blit_resized_into_canvas(quad_buf.data(), H, W,
                                             aug_pixels.data(), W,
                                             q.dst_x, q.dst_y, q.qw, q.qh, C);
                }
                image_bytes = aug_pixels.data();
            }
            else if (augment)
            {
                const ::opennn::AugmentationConfig free_cfg{
                    cfg.jitter, cfg.exposure, cfg.saturation, cfg.hue, cfg.flip
                };
                const AugmentationParams p = derive_augmentation_params(
                    epoch_seed, uint64_t(sample_index), free_cfg);

                aug_pixels.resize(size_t(cache_image_record_bytes));
                apply_geometric_to_image(pixels.data(), aug_pixels.data(),
                                         cache_input_shape[0], cache_input_shape[1], cache_input_shape[2], p);
                apply_color_jitter(aug_pixels.data(),
                                   cache_input_shape[0], cache_input_shape[1], cache_input_shape[2], p);
                image_bytes = aug_pixels.data();
            }

            if (resize_needed)
            {
                resized_pixels.resize(size_t(image_record_bytes));
                bilinear_resize_uint8(image_bytes,
                                      cache_input_shape[0], cache_input_shape[1],
                                      resized_pixels.data(),
                                      input_shape[0], input_shape[1],
                                      input_shape[2]);
                image_bytes = resized_pixels.data();
            }

            Map<Array<float, Dynamic, 1>>(input_data + i * image_record_bytes, image_record_bytes) =
                Map<const Array<uint8_t, Dynamic, 1>>(image_bytes, image_record_bytes).cast<float>() * scale;
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    throw_if(!omp_error.empty(),
             omp_error);
}

void YoloDataset::fill_targets(const vector<Index>& sample_indices,
                               const vector<Index>&,
                               float* target_data,
                               FillMode mode,
                               int) const
{
    const Index batch_size = ssize(sample_indices);

    const bool augment = mode == FillMode::Training && augmentation.enabled;
    const bool matrix_storage = storage_mode == StorageMode::Matrix;
    const uint64_t epoch_seed = augment
        ? augmentation_counter.load(memory_order_relaxed)
        : 0;

    AugmentationConfig cfg = augmentation;

    const bool grid_changed = (grid_size != cache_grid_size);
    const bool reencode = augment || grid_changed || is_multi_scale();

    if (matrix_storage && !reencode)
        load_targets_to_ram();

    string omp_error;

    #pragma omp parallel for
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

                if (augment && augmentation.mosaic)
                {
                    const Index H = cache_input_shape[0];
                    const Index W = cache_input_shape[1];
                    const MosaicParams mp = derive_mosaic_params(epoch_seed, uint64_t(sample_index), samples_number);
                    const Index cut_x = max<Index>(1, min(W - 1, Index(mp.cx_frac * float(W))));
                    const Index cut_y = max<Index>(1, min(H - 1, Index(mp.cy_frac * float(H))));

                    struct Quad { Index si; Index dst_x, dst_y, qw, qh; };
                    const Quad quads[4] = {
                        {sample_index,         0,     0,     cut_x,   cut_y},
                        {mp.companions[0], cut_x,     0, W - cut_x,   cut_y},
                        {mp.companions[1],     0, cut_y,     cut_x, H - cut_y},
                        {mp.companions[2], cut_x, cut_y, W - cut_x, H - cut_y},
                    };

                    vector<Box> mosaic_boxes;
                    vector<Box> quad_boxes;

                    for (const Quad& q : quads)
                    {
                        read_sample_boxes(q.si, quad_boxes);
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

                            Box out;
                            out.class_id = src.class_id;
                            out.x = 0.5f * (x0 + x1);
                            out.y = 0.5f * (y0 + y1);
                            out.w = x1 - x0;
                            out.h = y1 - y0;
                            mosaic_boxes.push_back(out);
                        }
                    }

                    if (is_multi_scale())
                        make_target_multi_scale(mosaic_boxes, head_anchors, head_grid_sizes,
                                                boxes_per_head, classes_number, target_ptr);
                    else
                        make_target(mosaic_boxes, anchors, grid_size, boxes_per_cell,
                                    classes_number, target_ptr);
                }
                else
                {
                    vector<Box> boxes;
                    read_sample_boxes(sample_index, boxes);

                    if (augment)
                    {
                        const ::opennn::AugmentationConfig free_cfg{
                            cfg.jitter, cfg.exposure, cfg.saturation, cfg.hue, cfg.flip
                        };
                        const AugmentationParams p = derive_augmentation_params(
                            epoch_seed, uint64_t(sample_index), free_cfg);
                        apply_geometric_to_boxes(boxes, p);
                    }

                    if (is_multi_scale())
                        make_target_multi_scale(boxes, head_anchors, head_grid_sizes,
                                                boxes_per_head, classes_number, target_ptr);
                    else
                        make_target(boxes, anchors, grid_size, boxes_per_cell,
                                    classes_number, target_ptr);
                }
            }
            else
            {
                const uint64_t offset = target_data_offset
                    + uint64_t(sample_index) * uint64_t(cache_target_record_floats) * sizeof(float);

                if (matrix_storage)
                {
                    copy_n(targets_ram.data() + size_t(sample_index) * size_t(cache_target_record_floats),
                           cache_target_record_floats,
                           target_data + i * target_record_floats);
                }
                else
                {
                    target_cache_reader.read_at(target_data + i * target_record_floats,
                                                size_t(cache_target_record_floats) * sizeof(float),
                                                offset);
                }
            }
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    throw_if(!omp_error.empty(),
             omp_error);
}

void YoloDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("YoloDataset");
    printer.open_element("DataSource");
    write_json(printer, {
        {"ImagesPath", images_directory.string()},
        {"LabelsPath", labels_directory.string()},
        {"StorageMode", get_storage_mode_string()},
        {"Height", to_string(input_shape[0])},
        {"Width", to_string(input_shape[1])},
        {"Channels", to_string(input_shape[2])},
        {"GridSize", to_string(grid_size)},
        {"BoxesPerCell", to_string(boxes_per_cell)}
    });
    printer.close_element();
    variables_to_JSON(printer);
    samples_to_JSON(printer);
    printer.close_element();
}

void YoloDataset::from_JSON(const JsonDocument& document)
{
    const Json* yolo_element = get_json_root(document, "YoloDataset");
    const Json* source = require_json_field(yolo_element, "DataSource");

    set(read_json_string(source, "ImagesPath"),
        read_json_string(source, "LabelsPath"),
        {read_json_index(source, "Height"),
         read_json_index(source, "Width"),
         read_json_index(source, "Channels")},
        read_json_index(source, "GridSize"),
        read_json_index(source, "BoxesPerCell"));

    set_storage_mode(source->has("StorageMode")
                   ? read_json_string(source, "StorageMode")
                   : "BinaryFile");
}

Index YoloDataset::load_darknet_backbone(NeuralNetwork& network,
                                          const filesystem::path& weights_path,
                                          Index n_backbone_convs)
{
    FILE* f = fopen(weights_path.string().c_str(), "rb");
    throw_if(!f, "load_darknet_backbone: cannot open file: " + weights_path.string());

    // Darknet header: 3 x int32 (major, minor, revision) + 1 x int64 (seen)
    int32_t header[3];
    int64_t seen;
    throw_if(fread(header, sizeof(int32_t), 3, f) != 3,
             "load_darknet_backbone: failed to read header int32s.");
    throw_if(fread(&seen, sizeof(int64_t), 1, f) != 1,
             "load_darknet_backbone: failed to read header seen.");

    cout << "Darknet weights header: major=" << header[0]
              << " minor=" << header[1]
              << " revision=" << header[2]
              << " seen=" << seen << "\n";

    Index loaded = 0;
    const auto& layers = network.get_layers();
    for (size_t li = 0; li < layers.size() && loaded < n_backbone_convs; ++li)
    {
        auto* conv = dynamic_cast<Convolutional*>(layers[li].get());
        if (!conv) continue;

        conv->load_darknet_weights(f);
        ++loaded;
        cout << format("Loaded backbone conv {}/{} from {}\n", loaded, n_backbone_convs, weights_path.string());
    }

    fclose(f);
    return loaded;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
