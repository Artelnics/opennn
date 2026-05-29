//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "yolo_dataset.h"
#include "image_utilities.h"
#include "io_utilities.h"
#include "json.h"

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

constexpr uint32_t YOLO_CACHE_VERSION = 2;
constexpr char YOLO_IMAGE_MAGIC[8] = {'O','P','E','N','N','Y','I','M'};
constexpr char YOLO_TARGET_MAGIC[8] = {'O','P','E','N','N','Y','T','G'};
constexpr char YOLO_BOXES_MAGIC[8] = {'O','P','E','N','N','Y','B','X'};

bool has_image_extension(const filesystem::path& path)
{
    return is_supported_image_file(path);
}

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
    for (const auto& entry : filesystem::directory_iterator(labels_directory))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".names")
            continue;

        ifstream file(entry.path());
        if (!file)
            throw runtime_error(format("Cannot open YOLO classes file: {}", entry.path().string()));

        vector<string> classes;
        string line;
        while (getline(file, line))
            if (!line.empty())
                classes.push_back(line);

        return classes;
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
        if (!(stream >> box.class_id >> box.x >> box.y >> box.w >> box.h))
            throw runtime_error(format("Invalid YOLO label line in {}: {}", label_path.string(), line));

        if (box.x < 0.0f || box.x > 1.0f || box.y < 0.0f || box.y > 1.0f
        ||  box.w < 0.0f || box.w > 1.0f || box.h < 0.0f || box.h > 1.0f)
            throw runtime_error(format("YOLO label values out of range in {}: {}", label_path.string(), line));

        boxes.push_back(box);
    }

    return boxes;
}

uint64_t hash_anchors(const vector<array<float, 2>>& anchors)
{
    uint64_t h = 1469598103934665603ull;

    auto mix = [&](uint32_t value)
    {
        h ^= value;
        h *= 1099511628211ull;
    };

    mix(uint32_t(anchors.size()));

    for (const auto& anchor : anchors)
    {
        uint32_t bits = 0;
        memcpy(&bits, &anchor[0], sizeof(uint32_t));
        mix(bits);
        memcpy(&bits, &anchor[1], sizeof(uint32_t));
        mix(bits);
    }

    return h;
}

uint64_t hash_sources(const filesystem::path& images_dir,
                      const filesystem::path& labels_dir)
{
    uint64_t h = 1469598103934665603ull;

    auto mix_u64 = [&](uint64_t value)
    {
        for (int b = 0; b < 8; ++b)
        {
            h ^= (value >> (b * 8)) & 0xff;
            h *= 1099511628211ull;
        }
    };

    auto mix_bytes = [&](const void* data, size_t n)
    {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
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

    vector<filesystem::path> image_paths = list_files(images_dir, has_image_extension);
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

    return h;
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
    {
        vector<array<float, 2>> fallback(size_t(boxes_per_cell), {0.1f, 0.1f});
        return fallback;
    }

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

            for (Index a = 0; a < boxes_per_cell; ++a)
            {
                const float iou = yolo_iou_wh(boxes[i], anchors[size_t(a)]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_anchor = a;
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

        for (Index a = 0; a < boxes_per_cell; ++a)
            if (counts[size_t(a)] > 0)
                anchors[size_t(a)] = {
                    sums[size_t(a)][0] / float(counts[size_t(a)]),
                    sums[size_t(a)][1] / float(counts[size_t(a)])
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

    for (Index y = 0; y < scaled_height; ++y)
        for (Index x = 0; x < scaled_width; ++x)
            for (Index c = 0; c < channels; ++c)
                output(y + offset_y, x + offset_x, c) = resized(y, x, c);

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

    for (Index a = 0; a < ssize(anchors); ++a)
    {
        const float iou = yolo_iou_wh({box.w, box.h}, anchors[size_t(a)]);
        if (iou > best_iou)
        {
            best_iou = iou;
            best = a;
        }
    }

    return best;
}

struct AugmentationParams
{
    float crop_left;     // pleft / orig_w   (signed, can be negative for padding)
    float crop_top;      // ptop  / orig_h
    float crop_right;    // pright / orig_w
    float crop_bottom;   // pbot  / orig_h
    bool flip;
    float exposure_mul;  // multiply V channel
    float saturation_mul;// multiply S channel
    float hue_shift;     // additive on H in [0, 1] (will be wrapped)
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
    auto rand_unit = [](uint64_t& s) -> float
    {
        s = splitmix64(s);
        return float(s >> 40) / float(1u << 24); // [0, 1)
    };

    auto rand_signed = [&](uint64_t& s, float range) -> float
    {
        return (rand_unit(s) * 2.0f - 1.0f) * range;
    };

    auto rand_scale = [&](uint64_t& s, float max_scale) -> float
    {
        // multiplier uniform in [1/max_scale, max_scale]
        const float r = rand_unit(s);
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
    const float orig_w = float(width);
    const float orig_h = float(height);
    const float src_x0 = p.crop_left * orig_w;
    const float src_y0 = p.crop_top * orig_h;
    const float src_x1 = (1.0f - p.crop_right) * orig_w;
    const float src_y1 = (1.0f - p.crop_bottom) * orig_h;
    const float crop_w = max(1.0f, src_x1 - src_x0);
    const float crop_h = max(1.0f, src_y1 - src_y0);

    for (Index dy = 0; dy < height; ++dy)
    {
        for (Index dx = 0; dx < width; ++dx)
        {
            const Index out_x = p.flip ? (width - 1 - dx) : dx;
            const float fx = src_x0 + (float(dx) + 0.5f) * crop_w / orig_w - 0.5f;
            const float fy = src_y0 + (float(dy) + 0.5f) * crop_h / orig_h - 0.5f;

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
        std::memcpy(dst, src, size_t(src_h) * size_t(src_w) * size_t(channels));
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
    const float crop_w = max(1e-6f, 1.0f - p.crop_left - p.crop_right);
    const float crop_h = max(1e-6f, 1.0f - p.crop_top - p.crop_bottom);

    vector<YoloDataset::Box> out;
    out.reserve(boxes.size());

    for (auto box : boxes)
    {
        const float bx0 = box.x - 0.5f * box.w;
        const float by0 = box.y - 0.5f * box.h;
        const float bx1 = box.x + 0.5f * box.w;
        const float by1 = box.y + 0.5f * box.h;

        float nx0 = (bx0 - p.crop_left) / crop_w;
        float ny0 = (by0 - p.crop_top)  / crop_h;
        float nx1 = (bx1 - p.crop_left) / crop_w;
        float ny1 = (by1 - p.crop_top)  / crop_h;

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

    boxes = std::move(out);
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
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        target[base + 0] = box.x * grid_size - float(col);
        target[base + 1] = box.y * grid_size - float(row);
        target[base + 2] = box.w;
        target[base + 3] = box.h;
        target[base + 4] = 1.0f;
        target[base + 5 + box.class_id] = 1.0f;
    }
}

// Multi-scale (FPN): writes one flat target per sample concatenating the
// per-head chunks in head order. Each ground-truth box is assigned to the
// single best-IoU (head, anchor) pair across all heads — only that one head's
// chunk receives a positive sample for that box; the other two heads have
// objectness 0 at the matching cell. This matches YOLO v3's "best anchor wins"
// assignment rule.
void make_target_multi_scale(const vector<YoloDataset::Box>& boxes,
                             const vector<vector<array<float, 2>>>& head_anchors,
                             const vector<Index>& head_grid_sizes,
                             Index boxes_per_head,
                             Index classes_number,
                             float* target)
{
    const Index values_per_box = 5 + classes_number;
    const Index head_channels = boxes_per_head * values_per_box;

    // Zero the whole multi-head buffer first.
    Index total_floats = 0;
    vector<Index> head_offsets(head_grid_sizes.size() + 1, 0);
    for (size_t h = 0; h < head_grid_sizes.size(); ++h)
    {
        const Index head_floats = head_grid_sizes[h] * head_grid_sizes[h] * head_channels;
        head_offsets[h + 1] = head_offsets[h] + head_floats;
        total_floats += head_floats;
    }
    fill_n(target, total_floats, 0.0f);

    for (const auto& box : boxes)
    {
        if (box.class_id < 0 || box.class_id >= classes_number)
            continue;

        // Best (head, anchor) by w/h IoU across all heads' anchors.
        float best_iou = -1.0f;
        size_t best_head = 0;
        Index best_anchor_in_head = 0;
        for (size_t h = 0; h < head_anchors.size(); ++h)
        {
            const vector<array<float, 2>>& anchors_h = head_anchors[h];
            for (Index a = 0; a < ssize(anchors_h); ++a)
            {
                const float iou = yolo_iou_wh({box.w, box.h}, anchors_h[size_t(a)]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_head = h;
                    best_anchor_in_head = a;
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
        target[base + 4] = 1.0f;
        target[base + 5 + box.class_id] = 1.0f;
    }
}

// VOC XML annotation parsing. Tag-based, not a full XML parser: relies on the
// well-known structure of PASCAL VOC annotations and is robust to whitespace
// and ordering of child tags. Throws on malformed input.

string read_voc_tag(const string& xml, const string& tag, size_t from = 0)
{
    const string open = "<" + tag + ">";
    const string close = "</" + tag + ">";
    const size_t a = xml.find(open, from);
    if (a == string::npos) return {};
    const size_t b = xml.find(close, a + open.size());
    if (b == string::npos) return {};
    string value = xml.substr(a + open.size(), b - a - open.size());
    // Trim ASCII whitespace.
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
    if (!file)
        throw runtime_error(format("Cannot open VOC annotation: {}", xml_path.string()));

    stringstream buffer;
    buffer << file.rdbuf();
    const string xml = buffer.str();

    VocAnnotation annotation;
    const string size_block = read_voc_tag(xml, "size");
    annotation.width  = stof(read_voc_tag(size_block, "width"));
    annotation.height = stof(read_voc_tag(size_block, "height"));

    if (annotation.width <= 0.0f || annotation.height <= 0.0f)
        throw runtime_error(format("VOC annotation has invalid size: {}", xml_path.string()));

    // Each <object> block contains a <name> and a <bndbox>.
    size_t cursor = 0;
    const string obj_open = "<object>";
    while ((cursor = xml.find(obj_open, cursor)) != string::npos)
    {
        const size_t obj_end = xml.find("</object>", cursor);
        if (obj_end == string::npos)
            throw runtime_error(format("Unterminated <object> in {}", xml_path.string()));

        const string obj = xml.substr(cursor, obj_end - cursor);
        const string bbox = read_voc_tag(obj, "bndbox");
        VocBox box;
        box.class_name = read_voc_tag(obj, "name");
        box.xmin = stof(read_voc_tag(bbox, "xmin"));
        box.ymin = stof(read_voc_tag(bbox, "ymin"));
        box.xmax = stof(read_voc_tag(bbox, "xmax"));
        box.ymax = stof(read_voc_tag(bbox, "ymax"));
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
                                       const filesystem::path& output_labels_dir)
{
    if (!filesystem::is_directory(voc_root))
        throw runtime_error(format("VOC root is not a directory: {}", voc_root.string()));

    const filesystem::path image_set_path =
        voc_root / "ImageSets" / "Main" / (image_set + ".txt");
    if (!filesystem::is_regular_file(image_set_path))
        throw runtime_error(format("VOC image-set file not found: {}",
                                   image_set_path.string()));

    const filesystem::path annotations_dir = voc_root / "Annotations";
    if (!filesystem::is_directory(annotations_dir))
        throw runtime_error(format("VOC Annotations dir not found: {}",
                                   annotations_dir.string()));

    filesystem::create_directories(output_labels_dir);

    // Class name → 0..19 lookup so we don't repeat linear scans per box.
    const vector<string>& classes = voc_class_names();
    unordered_map<string, Index> class_index;
    for (Index i = 0; i < ssize(classes); ++i)
        class_index[classes[size_t(i)]] = i;

    // Write the classes file (matches read_yolo_classes' .names lookup).
    ofstream names_file(output_labels_dir / "voc.names");
    if (!names_file)
        throw runtime_error(format("Cannot write VOC names file in {}",
                                   output_labels_dir.string()));
    for (const string& name : classes)
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
            continue;  // some image-set entries (e.g. test splits) have no annotation

        const VocAnnotation ann = parse_voc_xml(xml_path);

        const filesystem::path out_path = output_labels_dir / (image_id + ".txt");
        ofstream out(out_path);
        if (!out)
            throw runtime_error(format("Cannot write YOLO label: {}", out_path.string()));

        for (const VocBox& box : ann.boxes)
        {
            const auto it = class_index.find(box.class_name);
            if (it == class_index.end())
                continue;  // skip non-VOC-canonical classes (extensions, typos)

            const float cx = 0.5f * (box.xmin + box.xmax) / ann.width;
            const float cy = 0.5f * (box.ymin + box.ymax) / ann.height;
            const float bw = (box.xmax - box.xmin) / ann.width;
            const float bh = (box.ymax - box.ymin) / ann.height;

            // Clamp slightly out-of-range coords that occur in some VOC files
            // due to 1-indexed boundary conventions.
            auto clamp01 = [](float v) { return min(1.0f, max(0.0f, v)); };
            out << it->second << ' '
                << clamp01(cx) << ' ' << clamp01(cy) << ' '
                << clamp01(bw) << ' ' << clamp01(bh) << '\n';
        }
        ++converted;
    }

    return converted;
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

    for (Index k = 0; k < max_boxes; ++k)
    {
        const float* row = nms_output + k * 6;
        const float score = row[4];

        if (score <= 0.0f) break;  // NMS writes kept boxes first; first zero terminates

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
    if (new_input_shape.rank != 3)
        throw runtime_error("YoloDataset: input_shape must be rank 3.");
    if (new_grid_size <= 0 || new_boxes_per_cell <= 0)
        throw runtime_error("YoloDataset: grid_size and boxes_per_cell must be positive.");

    images_directory = new_images_dir;
    labels_directory = new_labels_dir;
    data_path = images_directory;
    input_shape = new_input_shape;
    grid_size = new_grid_size;
    boxes_per_cell = new_boxes_per_cell;
    image_record_bytes = input_shape.size();
    class_names = read_yolo_classes(labels_directory);

    image_cache_path = images_directory / ".cache" / "yolo_images.bin";
    target_cache_path = images_directory / ".cache" / "yolo_targets.bin";
    boxes_cache_path = images_directory / ".cache" / "yolo_boxes.bin";

    image_filenames = list_files(images_directory, has_image_extension);

    open_or_build_cache(new_anchors);
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

        anchors = std::move(cached_anchors);
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
    vector<filesystem::path> image_paths = list_files(images_directory, has_image_extension);
    if (image_paths.empty())
        throw runtime_error(format("YoloDataset: no images found in {}", images_directory.string()));

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
        if (image.dimension(2) != input_shape[2])
            throw runtime_error(format("YoloDataset: channel mismatch in {}", image_paths[i].string()));

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

        for (Index p = 0; p < image_record_bytes; ++p)
        {
            const float v = prepared.data()[p];
            const int iv = int(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v) + 0.5f);
            pixels[size_t(p)] = uint8_t(iv);
        }

        image_writer.write(pixels.data(), pixels.size());

        if (display && (i % 1000 == 0 || i + 1 == image_paths.size()))
            display_progress_bar(Index(i + 1), Index(image_paths.size()));
    }

    image_writer.finish_with_rename(image_cache_path);

    classes_number = class_names.empty() ? max_class_id + 1 : ssize(class_names);
    if (classes_number <= 0)
        throw runtime_error("YoloDataset: cannot infer classes_number.");

    if (class_names.empty())
    {
        class_names.resize(size_t(classes_number));
        for (Index i = 0; i < classes_number; ++i)
            class_names[size_t(i)] = to_string(i);
    }

    anchors = requested_anchors.empty()
        ? calculate_yolo_anchors(labels, boxes_per_cell)
        : requested_anchors;

    if (ssize(anchors) != boxes_per_cell)
        throw runtime_error("YoloDataset: anchors size must equal boxes_per_cell.");

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

    // Boxes side-cache: raw per-sample box lists for on-the-fly augmentation.
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
            YoloBoxRecord rec{};
            rec.class_id = int32_t(box.class_id);
            rec.x = box.x; rec.y = box.y; rec.w = box.w; rec.h = box.h;
            boxes_writer.write(&rec, sizeof(rec));
        }

    boxes_writer.finish_with_rename(boxes_cache_path);

    image_cache_reader.open(image_cache_path);
    target_cache_reader.open(target_cache_path);
    boxes_cache_reader.open(boxes_cache_path);
    target_data_offset = target_header.targets_offset;
    boxes_data_offset = boxes_header.boxes_byte_offset;
    boxes_offsets = std::move(offsets);

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

    variables.clear();
    variables.resize(size_t(input_shape.size() + target_shape.size()));

    for (Index i = 0; i < input_shape.size(); ++i)
    {
        Variable& variable = variables[size_t(i)];
        variable.name = format("pixel_{}", i + 1);
        variable.role = VariableRole::Input;
        variable.type = VariableType::Numeric;
        variable.scaler = ScalerMethod::None;
    }

    for (Index i = 0; i < target_shape.size(); ++i)
    {
        Variable& variable = variables[size_t(input_shape.size() + i)];
        variable.name = format("yolo_target_{}", i + 1);
        variable.role = VariableRole::Target;
        variable.type = VariableType::Numeric;
        variable.scaler = ScalerMethod::None;
    }

    sample_roles.assign(size_t(samples_number), SampleRole::Training);
    split_samples_random();
}

void YoloDataset::set_runtime_input_shape(const Shape& new_shape)
{
    if (new_shape.rank != 3)
        throw runtime_error("YoloDataset::set_runtime_input_shape: rank must be 3.");
    if (new_shape[2] != cache_input_shape[2])
        throw runtime_error("YoloDataset::set_runtime_input_shape: channel count must match the cache.");
    if (new_shape[0] <= 0 || new_shape[1] <= 0)
        throw runtime_error("YoloDataset::set_runtime_input_shape: H/W must be positive.");
    if (new_shape[0] > cache_input_shape[0] || new_shape[1] > cache_input_shape[1])
        throw runtime_error("YoloDataset::set_runtime_input_shape: H/W cannot exceed the cache letterbox size.");
    const Index stride = 32;
    if (new_shape[0] % stride != 0 || new_shape[1] % stride != 0)
        throw runtime_error("YoloDataset::set_runtime_input_shape: H/W must be multiples of 32.");

    input_shape = new_shape;
    grid_size = new_shape[0] / stride;
    image_record_bytes = input_shape.size();
    target_record_floats = grid_size * grid_size * boxes_per_cell * (5 + classes_number);
    target_shape = {grid_size, grid_size, boxes_per_cell * (5 + classes_number)};
}

void YoloDataset::set_multi_scale_heads(const vector<Index>& grid_sizes,
                                        const vector<vector<array<float, 2>>>& per_head_anchors)
{
    if (grid_sizes.empty() || grid_sizes.size() != per_head_anchors.size())
        throw runtime_error("YoloDataset::set_multi_scale_heads: head counts must match and be non-zero.");

    const Index per_head = ssize(per_head_anchors[0]);
    if (per_head <= 0)
        throw runtime_error("YoloDataset::set_multi_scale_heads: each head needs at least one anchor.");
    for (const auto& a : per_head_anchors)
        if (ssize(a) != per_head)
            throw runtime_error("YoloDataset::set_multi_scale_heads: all heads must have the same boxes_per_head.");
    for (Index g : grid_sizes)
        if (g <= 0)
            throw runtime_error("YoloDataset::set_multi_scale_heads: grid sizes must be positive.");

    head_grid_sizes = grid_sizes;
    head_anchors = per_head_anchors;
    boxes_per_head = per_head;

    // Flat target buffer = sum of per-head buffer sizes. The Loss decodes by
    // querying the network's Detection layers for their grid sizes.
    const Index values_per_box = 5 + classes_number;
    Index total_floats = 0;
    for (Index g : grid_sizes)
        total_floats += g * g * boxes_per_head * values_per_box;
    target_record_floats = total_floats;
    target_shape = {target_record_floats};
}

void YoloDataset::read_sample_boxes(Index sample_index, vector<Box>& out) const
{
    const uint64_t begin = boxes_offsets[size_t(sample_index)];
    const uint64_t end   = boxes_offsets[size_t(sample_index) + 1];
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

void YoloDataset::fill_inputs(const vector<Index>& sample_indices,
                              const vector<Index>&,
                              float* input_data,
                              bool is_training,
                              int) const
{
    const Index batch_size = ssize(sample_indices);
    const float scale = 1.0f / 255.0f;

    const bool augment = is_training && augmentation.enabled;
    const uint64_t epoch_seed = augment
        ? augmentation_counter.fetch_add(1, std::memory_order_relaxed) + 1
        : 0;

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

            const uint64_t offset = sizeof(YoloImageCacheHeader)
                + uint64_t(sample_indices[size_t(i)]) * uint64_t(cache_image_record_bytes);

            image_cache_reader.read_at(pixels.data(), pixels.size(), offset);

            const uint8_t* image_bytes = pixels.data();

            if (augment)
            {
                const ::opennn::AugmentationConfig free_cfg{
                    cfg.jitter, cfg.exposure, cfg.saturation, cfg.hue, cfg.flip
                };
                const AugmentationParams p = derive_augmentation_params(
                    epoch_seed, uint64_t(sample_indices[size_t(i)]), free_cfg);

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

            float* dst = input_data + i * image_record_bytes;
            for (Index p = 0; p < image_record_bytes; ++p)
                dst[p] = float(image_bytes[size_t(p)]) * scale;
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    if (!omp_error.empty())
        throw runtime_error(omp_error);
}

void YoloDataset::fill_targets(const vector<Index>& sample_indices,
                               const vector<Index>&,
                               float* target_data,
                               bool is_training,
                               int) const
{
    const Index batch_size = ssize(sample_indices);

    const bool augment = is_training && augmentation.enabled;
    const uint64_t epoch_seed = augment
        ? augmentation_counter.load(std::memory_order_relaxed)
        : 0;

    AugmentationConfig cfg = augmentation;

    const bool grid_changed = (grid_size != cache_grid_size);
    // Multi-scale: the target_cache only stores single-scale targets — always
    // re-encode from the boxes_cache, which is layout-independent.
    const bool reencode = augment || grid_changed || is_multi_scale();

    string omp_error;

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            if (reencode)
            {
                vector<Box> boxes;
                read_sample_boxes(sample_indices[size_t(i)], boxes);

                if (augment)
                {
                    const ::opennn::AugmentationConfig free_cfg{
                        cfg.jitter, cfg.exposure, cfg.saturation, cfg.hue, cfg.flip
                    };
                    const AugmentationParams p = derive_augmentation_params(
                        epoch_seed, uint64_t(sample_indices[size_t(i)]), free_cfg);
                    apply_geometric_to_boxes(boxes, p);
                }

                if (is_multi_scale())
                    make_target_multi_scale(boxes, head_anchors, head_grid_sizes,
                                            boxes_per_head, classes_number,
                                            target_data + i * target_record_floats);
                else
                    make_target(boxes, anchors, grid_size, boxes_per_cell, classes_number,
                                target_data + i * target_record_floats);
            }
            else
            {
                const uint64_t offset = target_data_offset
                    + uint64_t(sample_indices[size_t(i)]) * uint64_t(cache_target_record_floats) * sizeof(float);

                target_cache_reader.read_at(target_data + i * target_record_floats,
                                            size_t(cache_target_record_floats) * sizeof(float),
                                            offset);
            }
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    if (!omp_error.empty())
        throw runtime_error(omp_error);
}

void YoloDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("YoloDataset");
    printer.open_element("DataSource");
    write_json(printer, {
        {"ImagesPath", images_directory.string()},
        {"LabelsPath", labels_directory.string()},
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
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
