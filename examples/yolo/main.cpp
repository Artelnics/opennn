//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#  include <cuda_runtime.h>
#endif
#include "opennn/image_processing.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/configuration.h"
#include "opennn/forward_propagation.h"
#include "opennn/layer.h"
#include "opennn/non_max_suppression_layer.h"
#include "opennn/random_utilities.h"
#include "opennn/standard_networks.h"
#include "opennn/loss.h"
#include "opennn/training_strategy.h"
#include "opennn/yolo_dataset.h"

using namespace opennn;

namespace {

// Top-down 24-bit RGB image buffer. rgb[(y*W+x)*3 + {0,1,2}] = {R,G,B}.
struct Image24
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

void write_bmp24_top_down(const std::filesystem::path& path, const Image24& img)
{
    const int row_bytes_unpadded = img.width * 3;
    const int row_pad = (4 - row_bytes_unpadded % 4) % 4;
    const int row_stride = row_bytes_unpadded + row_pad;
    const int pixel_data_size = row_stride * img.height;
    const int file_size = 54 + pixel_data_size;

    std::vector<uint8_t> file(size_t(file_size), 0);

    file[0] = 'B'; file[1] = 'M';
    file[2] = uint8_t(file_size & 0xff);
    file[3] = uint8_t((file_size >> 8) & 0xff);
    file[4] = uint8_t((file_size >> 16) & 0xff);
    file[5] = uint8_t((file_size >> 24) & 0xff);
    file[10] = 54;
    file[14] = 40;
    file[18] = uint8_t(img.width & 0xff);
    file[19] = uint8_t((img.width >> 8) & 0xff);
    // Negative height = top-down BMP. Row 0 in pixel data is the top row.
    const int32_t h_signed = -img.height;
    file[22] = uint8_t(h_signed & 0xff);
    file[23] = uint8_t((h_signed >> 8) & 0xff);
    file[24] = uint8_t((h_signed >> 16) & 0xff);
    file[25] = uint8_t((h_signed >> 24) & 0xff);
    file[26] = 1;
    file[28] = 24;

    for (int y = 0; y < img.height; ++y)
    {
        const size_t src = size_t(y) * size_t(img.width) * 3;
        const size_t dst = 54 + size_t(y) * size_t(row_stride);
        for (int x = 0; x < img.width; ++x)
        {
            file[dst + size_t(x) * 3 + 0] = img.rgb[src + size_t(x) * 3 + 2]; // B
            file[dst + size_t(x) * 3 + 1] = img.rgb[src + size_t(x) * 3 + 1]; // G
            file[dst + size_t(x) * 3 + 2] = img.rgb[src + size_t(x) * 3 + 0]; // R
        }
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), file.size());
}

Image24 read_bmp24(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    if (buf.size() < 54 || buf[0] != 'B' || buf[1] != 'M')
        throw std::runtime_error("not a 24-bit BMP: " + path.string());

    auto u16 = [&](int off) { return uint16_t(uint8_t(buf[off]) | (uint8_t(buf[off+1]) << 8)); };
    auto u32 = [&](int off) {
        return uint32_t(uint8_t(buf[off])
                        | (uint8_t(buf[off+1]) << 8)
                        | (uint8_t(buf[off+2]) << 16)
                        | (uint8_t(buf[off+3]) << 24));
    };
    auto s32 = [&](int off) { return int32_t(u32(off)); };

    const uint32_t data_offset = u32(10);
    const int32_t width = s32(18);
    const int32_t height_signed = s32(22);
    if (u16(28) != 24)
        throw std::runtime_error("only 24-bit BMP supported: " + path.string());

    const bool top_down = height_signed < 0;
    const int height = std::abs(height_signed);
    const int row_stride = ((width * 3 + 3) / 4) * 4;

    Image24 img;
    img.width = width;
    img.height = height;
    img.rgb.assign(size_t(width) * size_t(height) * 3, 0);

    for (int y_out = 0; y_out < height; ++y_out)
    {
        const int y_src = top_down ? y_out : (height - 1 - y_out);
        const size_t src = data_offset + size_t(y_src) * size_t(row_stride);
        const size_t dst = size_t(y_out) * size_t(width) * 3;
        for (int x = 0; x < width; ++x)
        {
            img.rgb[dst + size_t(x) * 3 + 0] = uint8_t(buf[src + size_t(x) * 3 + 2]); // R
            img.rgb[dst + size_t(x) * 3 + 1] = uint8_t(buf[src + size_t(x) * 3 + 1]); // G
            img.rgb[dst + size_t(x) * 3 + 2] = uint8_t(buf[src + size_t(x) * 3 + 0]); // B
        }
    }
    return img;
}

void draw_rect_outline(Image24& img,
                       int x0, int y0, int x1, int y1,
                       std::array<uint8_t, 3> color, int thickness = 2)
{
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);

    auto put = [&](int x, int y)
    {
        if (x < 0 || x >= img.width || y < 0 || y >= img.height) return;
        const size_t i = (size_t(y) * size_t(img.width) + size_t(x)) * 3;
        img.rgb[i + 0] = color[0];
        img.rgb[i + 1] = color[1];
        img.rgb[i + 2] = color[2];
    };

    for (int t = 0; t < thickness; ++t)
    {
        for (int x = x0; x <= x1; ++x) { put(x, y0 + t); put(x, y1 - t); }
        for (int y = y0; y <= y1; ++y) { put(x0 + t, y); put(x1 - t, y); }
    }
}

// Write a 24-bit top-down BMP filled with `background`, with a solid
// `foreground` block of size `block`x`block` whose top-left corner is at
// (top_x, top_y) in top-down image coordinates.
void write_synthetic_bmp(const std::filesystem::path& path,
                         int width, int height,
                         std::array<uint8_t, 3> background,
                         std::array<uint8_t, 3> foreground,
                         int top_x, int top_y, int block)
{
    Image24 img;
    img.width = width;
    img.height = height;
    img.rgb.assign(size_t(width) * size_t(height) * 3, 0);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t i = (size_t(y) * size_t(width) + size_t(x)) * 3;
            img.rgb[i + 0] = background[0];
            img.rgb[i + 1] = background[1];
            img.rgb[i + 2] = background[2];
        }

    for (int dy = 0; dy < block; ++dy)
        for (int dx = 0; dx < block; ++dx)
        {
            const int x = top_x + dx;
            const int y = top_y + dy;
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            const size_t i = (size_t(y) * size_t(width) + size_t(x)) * 3;
            img.rgb[i + 0] = foreground[0];
            img.rgb[i + 1] = foreground[1];
            img.rgb[i + 2] = foreground[2];
        }

    write_bmp24_top_down(path, img);
}

void write_label(const std::filesystem::path& path,
                 int class_id, float cx, float cy, float w, float h)
{
    std::ofstream out(path);
    out << class_id << ' ' << cx << ' ' << cy << ' ' << w << ' ' << h << '\n';
}

void generate_synthetic_dataset(const std::filesystem::path& images_dir,
                                const std::filesystem::path& labels_dir,
                                int samples_per_class)
{
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);

    constexpr int image_size = 128;
    constexpr int block = 32;
    constexpr std::array<uint8_t, 3> background{255, 255, 255};
    const std::array<std::array<uint8_t, 3>, 3> colors{{
        {220,  40,  40},   // class 0 = "red"
        { 40,  60, 220},   // class 1 = "blue"
        { 40, 200,  40},   // class 2 = "green"
    }};

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> pos_dist(0, image_size - block);

    int index = 0;
    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < samples_per_class; ++i)
        {
            const int top_x = pos_dist(rng);
            const int top_y = pos_dist(rng);

            const std::string name = "sample_" + std::to_string(index);
            const std::filesystem::path image_path = images_dir / (name + ".bmp");
            const std::filesystem::path label_path = labels_dir / (name + ".txt");

            write_synthetic_bmp(image_path, image_size, image_size,
                                background, colors[size_t(c)],
                                top_x, top_y, block);

            const float cx = (float(top_x) + float(block) / 2.0f) / float(image_size);
            const float cy = (float(top_y) + float(block) / 2.0f) / float(image_size);
            const float w  = float(block) / float(image_size);
            const float h  = float(block) / float(image_size);

            write_label(label_path, c, cx, cy, w, h);
            ++index;
        }
    }

    std::ofstream classes(labels_dir / "classes.names");
    classes << "red\n" << "blue\n" << "green\n";
}

}

int main()
{
    try
    {
        std::cout << "OpenNN. YOLO Example." << std::endl;

        // Check 1: CPU self-consistency — does backward match forward's finite differences?
        {
            const float grad_err = opennn::yolo_loss_gradient_check_cpu();
            std::cout << "YOLO loss check 1 (CPU gradient self-consistency): max rel err = "
                      << std::scientific << std::setprecision(3) << grad_err;
            // Threshold 1e-2: float32 BCE uses log(p+ε) in forward but ∂/∂p of log(p) in backward.
            // The ε-mismatch gives ~3e-3 relative error — acceptable for single-precision.
            std::cout << (grad_err < 1e-2f ? "  [PASS]\n" : "  [FAIL — forward/backward inconsistent!]\n");
        }

        // Check 2: independent expected-value verification — does the forward compute
        // the RIGHT mathematical objective? (sign, lambda, GIoU formula, gradient direction)
        {
            std::cout << "YOLO loss check 2 (expected values & gradient directions):\n";
            const float ev_err = opennn::yolo_loss_expected_value_check_cpu();
            std::cout << "  max absolute error vs. hand-computed: "
                      << std::scientific << std::setprecision(3) << ev_err;
            std::cout << (ev_err < 1e-4f ? "  [PASS]\n" : "  [FAIL — math objective is wrong!]\n");
        }

        set_seed(42);
        Configuration::instance().set(Device::CUDA, Type::FP32);

        // ===== Configuration toggles =====
        //
        // Two independent switches: backbone architecture and dataset source.
        // Keep both at their defaults to reproduce the Phase 2 baseline; flip
        // them to exercise the Phase 3 additions.
        //
        // PASCAL VOC: download VOCdevkit/VOC2007 and point voc_root at it.
        // The converter writes YOLO-format labels into <data_dir>/voc_labels
        // (only the first time; subsequent runs reuse them).
        // Softmax = mutually-exclusive classes. Sigmoid = independent per-class
        // labels (YOLOv3-style). Use Sigmoid for all datasets here.
        const auto class_activation = YoloNetwork::ClassActivation::Sigmoid;

        const bool use_voc     = true;
        const bool use_raccoon = false;
        const bool use_coco    = false;  // COCO val2017, 3 classes: person/car/cat

        // VOC/COCO: Darknet53 (or CSPDarknet53) + 3-head FPN (loads pretrained darknet53.conv.74
        // for plain Darknet53; CSPDarknet53 trains from scratch — no compatible pretrained weights).
        // Raccoon/synthetic: VGG + single head (fast, good for small datasets).
        // Set use_panet=true to switch from FPN to PANet (adds a bottom-up path from small to large
        // head; weights file gets _panet suffix automatically).
        const bool use_panet   = false;
        // CSP backbone (§17): Cross-Stage Partial Darknet53. Each stage splits channels into two
        // branches — one goes through N residual blocks, the other is a linear skip — then concat
        // + merge. Halves gradient duplication vs plain Darknet53 at similar param count.
        // Trains from scratch (CSP layer names differ from darknet53.conv.74 weights).
        const bool use_csp     = true;
        // SPPF: three cascaded 5×5 max-pools + concat before the FPN neck. Tested on VOC 2007:
        // 48.9% mAP vs 54.9% plain FPN (-6pp). Disabled; left for reference.
        const bool use_sppf    = false;

        const auto backbone = (use_coco || use_voc)
            ? (use_csp ? YoloNetwork::Backbone::CSPDarknet53 : YoloNetwork::Backbone::Darknet53)
            : YoloNetwork::Backbone::Vgg;

        const auto head_style = (use_coco || use_voc)
            ? (use_panet ? YoloNetwork::HeadStyle::PANet : YoloNetwork::HeadStyle::FPN)
            : YoloNetwork::HeadStyle::Single;

        const auto body_activation = YoloNetwork::BodyActivation::LeakyReLU;

        const std::filesystem::path voc_root = "/home/alvaromartin/VOCdevkit/VOC2007";
        const std::string voc_image_set = "trainval";

        // Leave empty for all 20 VOC classes.
        // Set to a small list (e.g. {"dog","cat","car"}) for a simpler experiment.
        const std::vector<std::string> voc_class_filter = {}; //{"dog", "cat", "car"};

        // ===== Dataset =====

        const std::filesystem::path data_dir = use_voc     ? "yolo_voc_data"
                                             : use_raccoon ? "yolo_raccoon_data"
                                             : use_coco   ? "yolo_coco_data"
                                             :               "yolo_data";
        std::filesystem::create_directories(data_dir);

        std::filesystem::path images_dir;
        std::filesystem::path labels_dir;
        Index grid_size;
        Index boxes_per_cell;
        Shape input_shape;
        std::vector<std::array<float, 2>> anchors;

        if (use_voc)
        {
            images_dir = voc_root / "JPEGImages";
            // Use a separate labels dir when filtering so the 20-class cache is untouched
            labels_dir = data_dir / (voc_class_filter.empty() ? "voc_labels" : "voc_labels_filtered");
            const Index converted =
                YoloDataset::convert_voc_to_yolo(voc_root, voc_image_set, labels_dir, voc_class_filter);
            std::cout << "Converted " << converted
                      << " VOC samples to YOLO format in " << labels_dir << "\n";

            // When filtering, only use images that actually have filtered-class objects.
            // Without this, the ~3000 images with no labels are treated as all-background
            // and teach the model to suppress everything, collapsing mAP.
            if (!voc_class_filter.empty())
            {
                const std::filesystem::path filtered_images_dir = data_dir / "voc_images_filtered";
                std::filesystem::create_directories(filtered_images_dir);
                Index linked = 0;
                for (const auto& entry : std::filesystem::directory_iterator(labels_dir))
                {
                    if (entry.path().extension() != ".txt") continue;
                    if (entry.path().stem().string() == "voc") continue;
                    const std::string stem = entry.path().stem().string();
                    for (const char* ext : {".jpg", ".jpeg", ".JPG", ".JPEG"})
                    {
                        const auto src = images_dir / (stem + ext);
                        if (std::filesystem::exists(src))
                        {
                            const auto dst = filtered_images_dir / (stem + ext);
                            if (!std::filesystem::exists(dst))
                                std::filesystem::create_symlink(src, dst);
                            ++linked;
                            break;
                        }
                    }
                }
                images_dir = filtered_images_dir;
                std::cout << "Filtered to " << linked << " images containing the requested classes.\n";
            }

            grid_size = 13;
            boxes_per_cell = 3;          // 3 anchors per head × 2 FPN heads = 6 total
            input_shape = Shape{416, 416, 3};
            // Standard YOLOv3-tiny anchors (in image-fraction units).
            // Matches yolov3-tiny.weights training configuration so pretrained
            // backbone features align with anchor scale expectations.
            anchors = {
                {0.024f, 0.034f}, {0.055f, 0.065f}, {0.089f, 0.139f},   // small head 26×26
                {0.195f, 0.197f}, {0.325f, 0.406f}, {0.827f, 0.767f}};  // large head 13×13
        }
        else if (use_coco)
        {
            // COCO val2017 filtered to person/car/cat (~3022 labeled images).
            // Only images that have at least one label are symlinked to
            // coco_mini_data/labeled_images/ to avoid training on 1978
            // background-only images (which doubles epoch time for no gain).
            labels_dir = "/home/alvaromartin/coco_mini_data/train2017_labels";
            images_dir = data_dir / "labeled_images";
            std::filesystem::create_directories(images_dir);
            const std::filesystem::path src_images =
                "/home/alvaromartin/coco_mini_data/train2017";
            for (const auto& entry : std::filesystem::directory_iterator(labels_dir))
            {
                if (entry.path().extension() != ".txt") continue;
                if (entry.path().stem().string().find("coco_mini") != std::string::npos) continue;
                for (const char* ext : {".jpg", ".jpeg", ".png"})
                {
                    const auto src = src_images / (entry.path().stem().string() + ext);
                    const auto dst = images_dir / src.filename();
                    if (std::filesystem::exists(src) && !std::filesystem::exists(dst))
                        std::filesystem::create_symlink(src, dst);
                }
            }

            // DarknetTinyV3: 2-head FPN (stride-32 large + stride-16 small).
            // 6 anchors, 3 per head. Use the 6 best from k-means (drop medium).
            grid_size      = 13;   // stride-32 head; FPN adds 26×26 small head
            boxes_per_cell = 3;
            input_shape    = Shape{416, 416, 3};
            anchors = {
                {0.0240f, 0.0444f}, {0.0859f, 0.2730f}, {0.1625f, 0.4495f},  // small head (26×26)
                {0.4037f, 0.3609f}, {0.4677f, 0.7767f}, {0.7996f, 0.8253f},  // large head (13×13)
            };
        }
        else if (use_raccoon)
        {
            // Raccoon dataset: ~200 real JPEG photos, 1 class.
            // Annotations were converted from VOC XML to YOLO txt by raccoon_data/convert.py.
            images_dir = "/home/artelnics/Documents/opennn/raccoon_dataset/images";
            labels_dir = "/home/artelnics/Documents/opennn/raccoon_data/labels";

            grid_size      = 13;
            boxes_per_cell = 3;
            input_shape    = Shape{416, 416, 3};
            // k=3 anchors computed from the dataset (image-fraction units).
            anchors = {{0.334f, 0.476f}, {0.519f, 0.818f}, {0.807f, 0.852f}};
        }
        else
        {
            // Synthetic dataset: 128x128 BMPs, one colored block per image, 3 classes.
            images_dir = data_dir / "images";
            labels_dir = data_dir / "labels";
            generate_synthetic_dataset(images_dir, labels_dir, /*samples_per_class=*/256);

            // Grid 4x4 over 128x128 input requires input H/W == grid*32.
            grid_size      = 4;
            boxes_per_cell = 2;
            input_shape    = Shape{128, 128, 3};
            anchors        = {{0.25f, 0.25f}, {0.5f, 0.5f}};
        }

        // FPN anchor setup: DarknetTinyV3 uses 6 anchors (2-head: small 26×26 +
        // large 13×13). DarknetTiny (3-head residual) uses 9 anchors.
        // boxes_per_head ends up = 3 across all heads.
        const bool is_v3std     = (backbone == YoloNetwork::Backbone::DarknetTinyV3);
        const bool is_darknet53 = (backbone == YoloNetwork::Backbone::Darknet53);
        const bool is_csp53     = (backbone == YoloNetwork::Backbone::CSPDarknet53);
        const bool is_large_backbone = is_darknet53 || is_csp53;
        if (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
        {
            if (is_v3std)
            {
                // Standard YOLOv3-tiny anchors for VOC (image-fraction units).
                // Sorted smallest→largest; FPN ctor assigns small→26×26, large→13×13.
                if (use_voc)
                {
                    anchors = {
                        {0.024f, 0.031f}, {0.038f, 0.072f}, {0.079f, 0.055f},  // small head (26×26)
                        {0.279f, 0.216f}, {0.375f, 0.476f}, {0.896f, 0.783f}}; // large head (13×13)
                }
                else if (!use_coco)
                {
                    // Generic fallback for non-VOC, non-COCO datasets with FPN+TinyV3.
                    // COCO keeps its k-means anchors set above.
                    anchors = {
                        {0.08f, 0.08f}, {0.15f, 0.15f}, {0.25f, 0.25f},  // small head
                        {0.40f, 0.40f}, {0.55f, 0.55f}, {0.75f, 0.75f}}; // large head
                }
            }
            else
            {
                // DarknetTiny / Darknet53 (3-head FPN): 9 anchors.
                if (use_voc)
                {
                    // YOLOv3-style anchors k-means'd on VOC 2007 (image-fraction units).
                    // Sorted smallest to largest so the FPN ctor assigns small→stride-8,
                    // large→stride-32 automatically.
                    anchors = {
                        {0.024f, 0.031f}, {0.038f, 0.072f}, {0.079f, 0.055f}, // small
                        {0.072f, 0.147f}, {0.149f, 0.108f}, {0.142f, 0.286f}, // medium
                        {0.279f, 0.216f}, {0.375f, 0.476f}, {0.896f, 0.783f}};// large
                }
                else
                {
                    anchors = {
                        {0.05f, 0.05f}, {0.08f, 0.08f}, {0.12f, 0.12f},  // small head
                        {0.18f, 0.18f}, {0.25f, 0.25f}, {0.32f, 0.32f},  // medium head
                        {0.40f, 0.40f}, {0.55f, 0.55f}, {0.75f, 0.75f}}; // large head
                }
            }
            boxes_per_cell = 3;
        }

        // Dataset constructor validates anchors.size() == boxes_per_cell. In
        // FPN mode the single-scale path is unused after set_multi_scale_heads,
        // so any 3 anchors satisfy the constructor.
        const std::vector<std::array<float, 2>> ctor_anchors =
            (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
                ? std::vector<std::array<float, 2>>{anchors[0], anchors[1], anchors[2]}
                : anchors;

        YoloDataset dataset(images_dir, labels_dir, input_shape,
                            grid_size, boxes_per_cell, ctor_anchors);

        // FPN dataset configuration: grid scales and anchor groups per head.
        // DarknetTinyV3: 2 heads (13×13 large, 26×26 small).
        // DarknetTiny / Darknet53: 3 heads (13×13 large, 26×26 medium, 52×52 small).
        if (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
        {
            if (is_v3std)
            {
                // 2-head: stride-32 (13×13) and stride-16 (26×26)
                const std::vector<Index> head_grids = {
                    grid_size,         // 13×13 large head
                    grid_size * 2,     // 26×26 small head
                };
                const std::vector<std::vector<std::array<float, 2>>> head_anchors = {
                    {anchors[3], anchors[4], anchors[5]},  // large head (largest 3)
                    {anchors[0], anchors[1], anchors[2]},  // small head
                };
                dataset.set_multi_scale_heads(head_grids, head_anchors);
            }
            else
            {
                // 3-head: stride-32 / stride-16 / stride-8
                const std::vector<Index> head_grids = {
                    grid_size,         // stride 32 head: matches "grid_size" param
                    grid_size * 2,     // stride 16
                    grid_size * 4      // stride 8
                };
                const std::vector<std::vector<std::array<float, 2>>> head_anchors = {
                    {anchors[6], anchors[7], anchors[8]},  // large head (largest 3)
                    {anchors[3], anchors[4], anchors[5]},  // medium head
                    {anchors[0], anchors[1], anchors[2]},  // small head
                };
                dataset.set_multi_scale_heads(head_grids, head_anchors);
            }
        }

        YoloDataset::AugmentationConfig aug;
        if (use_voc || use_raccoon || use_coco)
        {
            // Standard YOLO augmentation for real detection data.
            aug.jitter      = 0.2f;
            aug.flip        = true;
            aug.exposure    = 1.5f;
            aug.saturation  = 1.5f;
            aug.hue         = 0.1f;
            aug.enabled     = true;
            // Raccoon objects fill ~70% of frame — mosaic shrinks them below anchor range.
            // COCO has varied sizes; mosaic composites are fine with multi-scale anchors.
            aug.mosaic      = use_voc || use_coco;
        }
        else
        {
            // Synthetic demo: class signal is color — disable hue/sat jitter so
            // we don't erase it. Geometric augmentation (flip + crop) is still on.
            aug.jitter      = 0.2f;
            aug.flip        = true;
            aug.exposure    = 1.2f;
            aug.saturation  = 1.0f;
            aug.hue         = 0.0f;
            aug.enabled     = false;
        }
        dataset.set_augmentation(aug);

        // 70% training, 30% selection (held-out) — gives a generalization signal:
        // TrainingStrategy reports both training_error and selection_error per epoch,
        // and the visualization at the end runs inference on samples the model has
        // never seen during training.
        const double train_frac = use_raccoon ? 0.8 : 0.7;
        const double val_frac   = use_raccoon ? 0.2 : 0.3;
        dataset.split_samples_random(train_frac, val_frac, 0.0);

        // Network.

        // Network anchors: FPN needs all 6 (DarknetTinyV3) or 9 (DarknetTiny)
        // anchors (the dataset only stores its 3 single-scale ctor anchors after
        // multi-scale config). Single-head continues to use the dataset's anchor list.
        const std::vector<std::array<float, 2>>& network_anchors =
            (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
                ? anchors : dataset.get_anchors();

        YoloNetwork yolo_network(input_shape,
                                 dataset.get_classes_number(),
                                 network_anchors,
                                 grid_size,
                                 backbone,
                                 class_activation,
                                 head_style,
                                 body_activation,
                                 use_sppf && is_large_backbone);

        std::cout << "Device: " << (yolo_network.is_gpu() ? "GPU" : "CPU")
                  << "  " << device::gpu_info_string() << "\n";
        std::cout << "Network: backbone="
                  << (backbone == YoloNetwork::Backbone::Vgg           ? "Vgg"
                    : backbone == YoloNetwork::Backbone::DarknetTinyV3  ? "DarknetTinyV3"
                    : backbone == YoloNetwork::Backbone::Darknet53      ? "Darknet53"
                    : backbone == YoloNetwork::Backbone::CSPDarknet53   ? "CSPDarknet53"
                    :                                                      "DarknetTiny")
                  << ", class_activation="
                  << (class_activation == YoloNetwork::ClassActivation::Sigmoid ? "Sigmoid" : "Softmax")
                  << ", head_style="
                  << (head_style == YoloNetwork::HeadStyle::FPN   ? "FPN"   :
                      head_style == YoloNetwork::HeadStyle::PANet  ? "PANet" : "Single")
                  << ", body_activation="
                  << (body_activation == YoloNetwork::BodyActivation::LeakyReLU ? "LeakyReLU" : "ReLU")
                  << (use_csp ? ", csp=on" : "")
                  << (use_sppf && is_large_backbone ? ", sppf=on" : "")
                  << ", layers=" << yolo_network.get_layers_number()
                  << ", parameters=" << yolo_network.get_parameters_number() << "\n";

        // Loosen the NMS confidence threshold so the demo prints something visible
        // even on an under-trained tiny model. FPN mode has no NMS layer —
        // cross-scale NMS happens externally in decode_yolo_fpn_detections, so
        // skip the NMS configuration in that case.
        if (head_style != YoloNetwork::HeadStyle::FPN && head_style != YoloNetwork::HeadStyle::PANet)
        {
            auto* nms_layer = dynamic_cast<NonMaxSuppression*>(
                yolo_network.get_layer("non_max_suppression_layer").get());
            if (nms_layer)
                nms_layer->set(yolo_network.get_layer(yolo_network.get_layers_number() - 2)->get_output_shape(),
                               boxes_per_cell, /*confidence=*/0.0f, /*iou=*/0.4f,
                               "non_max_suppression_layer");
        }

        // Training.

        TrainingStrategy training_strategy(&yolo_network, &dataset);
        training_strategy.set_loss("Yolo");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        // L2 weight decay is the brake on directional weight drift from the
        // geometry-coupled YOLO loss (especially when GIoU/DIoU is enabled).
        // Default weight 0.001 — matches every other example in the repo.
        training_strategy.get_loss()->set_regularization("L2");
        training_strategy.get_loss()->set_yolo_lambda_noobj(1.0f);      // raised from 0.5: suppress the massive false-positive rate
        training_strategy.get_loss()->set_yolo_lambda_class(2.0f);
        training_strategy.get_loss()->set_yolo_focal_gamma(2.0f);      // focal loss on class BCE
        training_strategy.get_loss()->set_yolo_obj_focal_gamma(2.0f);  // focal loss on objectness BCE

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        // Raccoon/VOC: real photos need a smaller batch (GPU memory) and more
        // patience before early stop fires (loss is noisier on small real datasets).
        // Darknet53 is 7x larger than TinyV3 — batch 4 keeps it within 7.7 GB VRAM.
        const int batch_size = is_large_backbone ? 4 : 16;
        adam->set_batch_size(batch_size);
        adam->set_display_period(1);
        adam->set_gradient_clip_norm(0.1f);
        adam->set_maximum_validation_failures(use_coco ? 50 : (use_voc && is_large_backbone) ? 40 : use_voc ? 25 : use_raccoon ? 25 : 15);

        // Training control:
        //   resume_training = true  → load weights if they exist, then continue
        //                             training from the saved epoch count.
        //   resume_training = false → if weights exist, skip training entirely
        //                             (visualization / inference only).
        // Delete the weights file (and epochs_done.txt) to train from scratch.
        const bool resume_training = true;

        const std::string dataset_tag  = use_voc ? "voc" : use_raccoon ? "raccoon" : use_coco ? "coco" : "synth";
        const std::string filter_tag   = voc_class_filter.empty() ? "" :
            "_" + std::to_string(voc_class_filter.size()) + "cls";
        const std::string weights_filename = std::string("yolo_weights_") + dataset_tag + "_" +
            (backbone == YoloNetwork::Backbone::Vgg           ? "vgg"
           : backbone == YoloNetwork::Backbone::DarknetTinyV3 ? "darknet_v3std"
           : backbone == YoloNetwork::Backbone::CSPDarknet53  ? "csp53"
           : backbone == YoloNetwork::Backbone::Darknet53     ? "darknet53"
           :                                                     "darknet") +
            (head_style == YoloNetwork::HeadStyle::FPN    ? "_fpn"    :
             head_style == YoloNetwork::HeadStyle::PANet  ? "_panet"  : "") +
            (use_sppf && is_large_backbone ? "_sppf" : "") +
            (body_activation == YoloNetwork::BodyActivation::LeakyReLU ? "_leaky" : "") +
            (class_activation == YoloNetwork::ClassActivation::Sigmoid ? "_sigmoid" : "") +
            filter_tag +
            std::string("_bce_ig_bgfocal.bin");  // bgfocal = focal on bg objectness only (asymmetric)
        std::filesystem::path weights_path = data_dir / weights_filename;
        // Backward-compat: load the Phase 2 committed weights file if present.
        const std::filesystem::path legacy_weights = data_dir / "yolo_weights.bin";
        if (backbone == YoloNetwork::Backbone::Vgg
        &&  !std::filesystem::exists(weights_path)
        &&   std::filesystem::exists(legacy_weights))
            weights_path = legacy_weights;

        // States (e.g. BatchNorm running_mean / running_variance) live in a
        // separate file so they are saved and restored alongside parameters.
        std::filesystem::path states_path = weights_path;
        states_path.replace_extension(".states.bin");

        const bool weights_exist = std::filesystem::exists(weights_path);
        if (weights_exist)
        {
            yolo_network.load_parameters_binary(weights_path);
            if (std::filesystem::exists(states_path))
                yolo_network.load_states_binary(states_path);
            std::cout << "\nLoaded weights from \"" << weights_path.string() << "\".\n";
        }

        // Load Darknet pretrained backbone weights (only on first run — skip when
        // resuming from a fine-tuned checkpoint, which already has better weights).
        const bool needs_darknet_backbone =
            (backbone == YoloNetwork::Backbone::DarknetTinyV3 ||
             backbone == YoloNetwork::Backbone::Darknet53 ||
             backbone == YoloNetwork::Backbone::CSPDarknet53) && !weights_exist;
        bool backbone_pretrained_loaded = false;
        if (needs_darknet_backbone)
        {
            const bool is53  = (backbone == YoloNetwork::Backbone::Darknet53);
            const bool iscsp = (backbone == YoloNetwork::Backbone::CSPDarknet53);
            // yolov4.conv.137 = first 137 conv layers of YOLOv4; backbone is first 72.
            const std::string darknet_filename = is53 ? "darknet53.conv.74"
                                               : iscsp ? "yolov4.conv.137"
                                               : "yolov3-tiny.weights";
            // Look in data_dir first, then in yolo_voc_data (where it was originally downloaded).
            std::filesystem::path darknet_weights = data_dir / darknet_filename;
            if (!std::filesystem::exists(darknet_weights))
                darknet_weights = std::filesystem::path("yolo_voc_data") / darknet_filename;
            const Index n_backbone_convs = is53 ? 52 : iscsp ? 72 : 8;
            if (std::filesystem::exists(darknet_weights))
            {
                const Index loaded = YoloDataset::load_darknet_backbone(
                    yolo_network, darknet_weights, n_backbone_convs);
                std::cout << "Loaded " << loaded
                          << " backbone layers from " << darknet_weights << "\n";
                backbone_pretrained_loaded = true;
            }
            else
            {
                std::cout << "Darknet pretrained weights not found at " << darknet_weights << ".\n";
                if (is53)
                    std::cout << "Download darknet53.conv.74 from "
                              << "https://pjreddie.com/media/files/darknet53.conv.74 "
                              << "and place it in " << data_dir << "\n";
                else if (iscsp)
                    std::cout << "Download yolov4.conv.137 from "
                              << "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 "
                              << "and place it in " << data_dir << "\n";
                else
                    std::cout << "Download yolov3-tiny.weights from "
                              << "https://pjreddie.com/media/files/yolov3-tiny.weights "
                              << "and place it in " << data_dir << "\n";
                std::cout << "Training from scratch instead.\n";
            }
        }

        // Backbone freezing: freeze pretrained backbone layers during warmup so the
        // randomly-initialised neck+head can stabilise before the backbone adapts.
        // Only applies on a fresh run where backbone weights were just loaded.
        bool backbone_frozen = false;
        auto set_backbone_trainable = [&](bool trainable) {
            const std::string prefix = (backbone == YoloNetwork::Backbone::Darknet53)    ? "dn53_"  :
                                       (backbone == YoloNetwork::Backbone::CSPDarknet53) ? "csp53_" :
                                       (backbone == YoloNetwork::Backbone::DarknetTinyV3) ? "dntv3_" : "";
            if (prefix.empty()) return;
            for (auto& layer : yolo_network.get_layers())
                if (layer && layer->get_label().rfind(prefix, 0) == 0)
                    layer->set_is_trainable(trainable);
            std::cout << (trainable ? "Unfreezing" : "Freezing") << " backbone layers (" << prefix << "*).\n";
        };
        // Backbone freezing disabled: Adam initialises gradient buffers for the
        // frozen (head-only) state, so unfreezing mid-training leaves backbone
        // layers without allocated gradient buffers → GPU kernel hang on first
        // backward pass. Train full network from epoch 0 instead.
        // Backbone freezing disabled — causes GPU hang on unfreeze.
        // set_backbone_trainable(false);
        // backbone_frozen = true;
        // }

        // LR step-decay schedule (YOLO convention).
        // epochs_done.txt tracks progress so resume always picks the right stage.
        struct TrainingRound { float lr; int epochs; };
        // Large backbones (Darknet53/CSPDarknet53) run batch=4 (VRAM constraint).
        // Scale LR ∝ batch_size vs the batch=16 baseline: ×(4/16) = ×0.25.
        // CSPDarknet53 trains from scratch (no pretrained weights) — runs 200 epochs
        // at the main LR to compensate.
        const std::vector<TrainingRound> lr_schedule =
            use_coco         ? std::vector<TrainingRound>{{1e-4f, 300}, {3e-5f, 200}}                     :
            (use_voc && is_csp53)
                             ? std::vector<TrainingRound>{{1.25e-4f, 200}, {2.5e-5f, 150}, {1e-5f, 100}} :
            (use_voc && is_darknet53)
                             ? std::vector<TrainingRound>{{1.25e-4f, 150}, {2.5e-5f, 150}, {1e-5f, 100}} :
            use_voc          ? std::vector<TrainingRound>{{5e-4f, 150}, {1e-4f, 150}, {3e-5f, 100}}      :
            use_raccoon      ? std::vector<TrainingRound>{{5e-4f, 400}, {1e-4f, 300}}                     :
                               std::vector<TrainingRound>{{1e-3f, 200}};

        // Epochs file is scoped to the weights filename so switching variants
        // (backbone, class activation, head style) always starts from 0.
        const std::filesystem::path epochs_file =
            data_dir / (weights_filename.substr(0, weights_filename.size() - 4) + "_epochs.txt");
        int epochs_done = 0;
        if (std::filesystem::exists(epochs_file))
        {
            std::ifstream ef(epochs_file);
            ef >> epochs_done;
        }
        std::cout << "Epochs completed so far: " << epochs_done << "\n";

        if (resume_training || !std::filesystem::exists(weights_path))
        {
            int cumulative = 0;
            for (const TrainingRound& rnd : lr_schedule)
            {
                const int round_end = cumulative + rnd.epochs;
                if (epochs_done >= round_end) { cumulative = round_end; continue; }

                // Unfreeze backbone before first post-warmup phase
                if (backbone_frozen && cumulative >= 5) {
                    set_backbone_trainable(true);
                    backbone_frozen = false;
                }

                const int to_run = round_end - epochs_done;
                adam->set_learning_rate(rnd.lr);
                adam->set_maximum_epochs(to_run);
                std::cout << "\nTraining: lr=" << rnd.lr
                          << " for " << to_run << " epochs"
                          << " (target epoch " << round_end << ").\n";
                const auto train_result = training_strategy.train();
                epochs_done += static_cast<int>(train_result.get_epochs_number());
                cumulative   = round_end;

                yolo_network.save_parameters_binary(weights_path);
                yolo_network.save_states_binary(states_path);
                { std::ofstream ef(epochs_file); ef << epochs_done; }
                std::cout << "Checkpoint saved: " << epochs_done << " total epochs.\n";
            }
            std::cout << "Training complete (" << epochs_done << " total epochs).\n";
        }

        // FPN mode: gather per-head Detection layer outputs from a manual
        // forward pass (no NMS layer is appended in FPN networks), then run
        // cross-scale NMS in decode_yolo_fpn_detections. Single-head mode
        // continues to read from the appended NMS layer below.
        const bool is_fpn = (head_style == YoloNetwork::HeadStyle::FPN ||
                              head_style == YoloNetwork::HeadStyle::PANet);

        // Inference on 5 SELECTION (held-out) samples — the model has never
        // seen these during training, so this is the real generalization test.
        // Each image is saved as an annotated BMP showing the input, the
        // ground-truth box (green), and the top-1 predicted box (red).

        const std::filesystem::path output_dir = data_dir / "annotated";
        // Clear stale files from previous runs so the folder only ever shows
        // the current run's output.
        if (std::filesystem::exists(output_dir))
            for (const auto& entry : std::filesystem::directory_iterator(output_dir))
                std::filesystem::remove(entry.path());
        std::filesystem::create_directories(output_dir);

        const Index image_floats = input_shape[0] * input_shape[1] * input_shape[2];
        std::vector<float> input_buffer(size_t(image_floats), 0.0f);
        Tensor4 input(1, input_shape[0], input_shape[1], input_shape[2]);

        const Index max_boxes = grid_size * grid_size * boxes_per_cell;
        const std::vector<std::string>& class_names = dataset.get_class_names();
        const std::array<std::array<uint8_t, 3>, 2> box_color_by_role{{
            { 30, 200,  30}, // ground truth = green
            {220,  40,  40}, // prediction   = red
        }};

        // Pick 3 validation samples, one per class, so all colours appear in the output.
        // For the synthetic dataset, class is encoded in the filename: sample_N where
        //   N < samples_per_class          → class 0 (red)
        //   N < 2 * samples_per_class      → class 1 (blue)
        //   N < 3 * samples_per_class      → class 2 (green)
        // For VOC, fall back to the first 3 validation samples.
        const std::vector<Index> selection_indices = dataset.get_sample_indices("Validation");
        const Index num_classes = dataset.get_classes_number();
        std::vector<Index> vis_indices;
        const bool is_synthetic = !use_voc && !use_raccoon && !use_coco;
        const Index max_vis = is_synthetic ? num_classes : Index(9);
        if (is_synthetic)
        {
            // One sample per class, identified by filename index.
            std::vector<bool> class_found(size_t(num_classes), false);
            constexpr int spc = 256;
            for (Index idx : selection_indices)
            {
                const std::string stem = dataset.get_image_path(idx).stem().string();
                const auto under = stem.rfind('_');
                if (under == std::string::npos) continue;
                const int n = std::stoi(stem.substr(under + 1));
                const int c = n / spc;
                if (c >= 0 && c < int(num_classes) && !class_found[size_t(c)])
                {
                    vis_indices.push_back(idx);
                    class_found[size_t(c)] = true;
                }
                if (Index(vis_indices.size()) == max_vis) break;
            }
        }
        else
        {
            // For real datasets pick 9 samples evenly spaced across the val set
            // so a variety of scenes appears in the annotated folder.
            for (Index k = 0; k < max_vis; ++k)
            {
                const Index pos = k * Index(selection_indices.size()) / max_vis;
                vis_indices.push_back(selection_indices[size_t(pos)]);
            }
        }
        const Index samples_to_visualize = Index(vis_indices.size());
        std::cout << "\nVisualizing " << samples_to_visualize
                  << " held-out (validation) samples — model has never seen these. "
                  << "Annotated BMPs in " << output_dir << ":\n";

        for (Index k = 0; k < samples_to_visualize; ++k)
        {
            const Index s = vis_indices[size_t(k)];

            dataset.fill_inputs({s}, {}, input_buffer.data(),
                                /*is_training=*/false, /*parallelize=*/false);
            std::copy(input_buffer.begin(), input_buffer.end(), input.data());

            std::vector<YoloDetection> detections;

            if (is_fpn)
            {
                // Manual forward pass — we need every Detection layer's output,
                // not just the network terminal. Walk layers, build YoloFpnHead
                // entries for each Detection layer, then cross-scale NMS.
                ForwardPropagation forward_propagation(/*batch_size=*/1, &yolo_network);
                const std::vector<TensorView> input_views = {
                    TensorView(input.data(),
                               {1, input.dimension(1), input.dimension(2), input.dimension(3)},
                               Type::FP32)
                };
                yolo_network.forward_propagate(input_views, forward_propagation, /*is_training=*/false);

                std::vector<YoloFpnHead> fpn_heads;
                std::vector<std::vector<float>> fpn_cpu_buffers; // keep CPU copies alive
                const auto& layers = yolo_network.get_layers();
                for (size_t li = 0; li < layers.size(); ++li)
                {
                    if (!layers[li] || layers[li]->get_type() != LayerType::Detection) continue;
                    const Shape head_shape = layers[li]->get_output_shape();
                    // Shape per Detection layer: [grid, grid, 3*(5+classes)] — drop batch.
                    const TensorView view = forward_propagation.forward_slots[li].back();
                    const Index channels = head_shape[2];
                    const Index classes_n = Index(dataset.get_classes_number());
                    const Index boxes_per_head = channels / (5 + classes_n);

                    const float* data_ptr = view.as<float>();
#ifdef OPENNN_HAS_CUDA
                    if (view.is_cuda())
                    {
                        fpn_cpu_buffers.emplace_back(size_t(view.size()));
                        cudaMemcpy(fpn_cpu_buffers.back().data(), data_ptr,
                                   size_t(view.size()) * sizeof(float), cudaMemcpyDeviceToHost);
                        data_ptr = fpn_cpu_buffers.back().data();
                    }
#endif
                    fpn_heads.push_back({ data_ptr, head_shape[0], boxes_per_head, classes_n });
                }

                // Raw-score diagnostics — printed before NMS so we can see what
                // the model is actually predicting even when nothing survives.
                {
                    float max_obj = 0.f, max_score = 0.f;
                    int above_001 = 0, above_01 = 0, above_025 = 0;
                    for (const YoloFpnHead& h : fpn_heads)
                    {
                        const Index vpb = 5 + h.classes_number;
                        const Index chan = h.boxes_per_cell * vpb;
                        for (Index r = 0; r < h.grid_size; ++r)
                        for (Index c = 0; c < h.grid_size; ++c)
                        for (Index b = 0; b < h.boxes_per_cell; ++b)
                        {
                            const Index base = (r * h.grid_size + c) * chan + b * vpb;
                            const float obj = h.data[base + 4];
                            float best_p = 0.f;
                            for (Index cl = 0; cl < h.classes_number; ++cl)
                                best_p = std::max(best_p, h.data[base + 5 + cl]);
                            const float score = obj * best_p;
                            max_obj   = std::max(max_obj,   obj);
                            max_score = std::max(max_score, score);
                            if (score >= 0.01f)  ++above_001;
                            if (score >= 0.1f)   ++above_01;
                            if (score >= 0.25f)  ++above_025;
                        }
                    }
                    std::cout << "  Raw : max_obj=" << max_obj
                              << " max_score=" << max_score
                              << " boxes≥0.01:" << above_001
                              << " ≥0.1:" << above_01
                              << " ≥0.25:" << above_025 << "\n";
                }

                detections = decode_yolo_fpn_detections(
                    fpn_heads,
                    /*original_height=*/input_shape[0],
                    /*original_width=*/input_shape[1],
                    /*network_height=*/input_shape[0],
                    /*network_width=*/input_shape[1],
                    /*confidence_threshold=*/0.001f,
                    /*iou_threshold=*/0.45f);
            }
            else
            {
                const MatrixR outputs = yolo_network.calculate_outputs(input);
                detections = decode_yolo_detections(
                    outputs.data(), max_boxes,
                    /*original_height=*/input_shape[0],
                    /*original_width=*/input_shape[1],
                    /*network_height=*/input_shape[0],
                    /*network_width=*/input_shape[1]);
            }

            const std::filesystem::path image_path = dataset.get_image_path(s);
            const std::string name = image_path.stem().string();
            std::filesystem::path label_path = labels_dir / image_path.filename();
            label_path.replace_extension(".txt");

            // Read ALL GT boxes from the label file.
            struct GtBox { int cls; float cx, cy, w, h; };
            std::vector<GtBox> gt_boxes;
            {
                std::ifstream lf(label_path);
                int c; float cx, cy, w, h;
                while (lf >> c >> cx >> cy >> w >> h)
                    gt_boxes.push_back({c, cx, cy, w, h});
            }
            // Primary GT (first box) — drives IoU ranking and cell diagnostics.
            const int   gt_class = gt_boxes.empty() ? 0     : gt_boxes[0].cls;
            const float gt_cx    = gt_boxes.empty() ? 0.5f  : gt_boxes[0].cx;
            const float gt_cy    = gt_boxes.empty() ? 0.5f  : gt_boxes[0].cy;
            const float gt_w     = gt_boxes.empty() ? 0.0f  : gt_boxes[0].w;
            const float gt_h     = gt_boxes.empty() ? 0.0f  : gt_boxes[0].h;

            // Load image and place it on a network-input-sized (416×416) letterbox canvas.
            // All coordinates — GT and predictions — live in this 416×416 pixel space.
            const int canvas_W = int(input_shape[1]);
            const int canvas_H = int(input_shape[0]);
            const int W = canvas_W, H = canvas_H;  // keep W/H for downstream cell math

            Image24 img;
            float lb_scale = 1.0f, lb_pad_x = 0.0f, lb_pad_y = 0.0f;
            int orig_W = canvas_W, orig_H = canvas_H;

            if (image_path.extension() == ".bmp" || image_path.extension() == ".BMP")
            {
                // Synthetic BMP is already 128×128 = network size; no letterbox needed.
                img    = read_bmp24(image_path);
                orig_W = img.width;
                orig_H = img.height;
            }
            else
            {
                const auto raw = opennn::load_image(image_path);
                orig_H  = int(raw.dimension(0));
                orig_W  = int(raw.dimension(1));
                lb_scale = std::min(float(canvas_W) / float(orig_W),
                                    float(canvas_H) / float(orig_H));
                lb_pad_x = (float(canvas_W) - float(orig_W) * lb_scale) * 0.5f;
                lb_pad_y = (float(canvas_H) - float(orig_H) * lb_scale) * 0.5f;
                const int scaled_W = int(std::round(float(orig_W) * lb_scale));
                const int scaled_H = int(std::round(float(orig_H) * lb_scale));
                const int off_x    = int(std::round(lb_pad_x));
                const int off_y    = int(std::round(lb_pad_y));

                img.width = canvas_W; img.height = canvas_H;
                img.rgb.assign(size_t(canvas_W * canvas_H * 3), 128);  // gray letterbox fill

                const int ch = int(raw.dimension(2));
                for (int py = 0; py < scaled_H; ++py)
                {
                    const int src_y = std::min(int(float(py) / lb_scale + 0.5f), orig_H - 1);
                    for (int px = 0; px < scaled_W; ++px)
                    {
                        const int src_x = std::min(int(float(px) / lb_scale + 0.5f), orig_W - 1);
                        const int dst_y = py + off_y, dst_x = px + off_x;
                        if (dst_y < 0 || dst_y >= canvas_H || dst_x < 0 || dst_x >= canvas_W) continue;
                        const float r = raw(src_y, src_x, 0);
                        const float g = ch > 1 ? raw(src_y, src_x, 1) : r;
                        const float b = ch > 2 ? raw(src_y, src_x, 2) : r;
                        const size_t idx = size_t((dst_y * canvas_W + dst_x) * 3);
                        img.rgb[idx]     = uint8_t(std::clamp(int(r), 0, 255));
                        img.rgb[idx + 1] = uint8_t(std::clamp(int(g), 0, 255));
                        img.rgb[idx + 2] = uint8_t(std::clamp(int(b), 0, 255));
                    }
                }
            }

            // Convert normalized label coords → letterbox pixel coords.
            auto lb_px = [&](float cx, float cy, float w, float h,
                              int& x0, int& y0, int& x1, int& y1)
            {
                const float cx_px = cx * float(orig_W) * lb_scale + lb_pad_x;
                const float cy_px = cy * float(orig_H) * lb_scale + lb_pad_y;
                const float hw    = w * float(orig_W) * lb_scale * 0.5f;
                const float hh    = h * float(orig_H) * lb_scale * 0.5f;
                x0 = int(std::round(cx_px - hw));
                y0 = int(std::round(cy_px - hh));
                x1 = int(std::round(cx_px + hw)) - 1;
                y1 = int(std::round(cy_px + hh)) - 1;
            };

            // Primary GT pixel coords used for IoU calculations below.
            int gt_x0, gt_y0, gt_x1, gt_y1;
            lb_px(gt_cx, gt_cy, gt_w, gt_h, gt_x0, gt_y0, gt_x1, gt_y1);

            // Draw all GT boxes in green.
            for (const auto& gb : gt_boxes)
            {
                int x0, y0, x1, y1;
                lb_px(gb.cx, gb.cy, gb.w, gb.h, x0, y0, x1, y1);
                draw_rect_outline(img, x0, y0, x1, y1, box_color_by_role[0], 2);
            }

            std::sort(detections.begin(), detections.end(),
                      [](const YoloDetection& a, const YoloDetection& b)
                      { return a.score > b.score; });

            const std::string gt_class_name =
                (gt_class >= 0 && size_t(gt_class) < class_names.size())
                    ? class_names[size_t(gt_class)] : std::to_string(gt_class);

            // Diagnostic: find detections that landed inside the GT's responsible cell.
            // If the model trained correctly, a detection here should have non-trivial
            // score and IoU > 0.5 with GT.
            const int gt_col_cell = std::min(int(grid_size) - 1,
                                              std::max(0, int(gt_cx * float(grid_size))));
            const int gt_row_cell = std::min(int(grid_size) - 1,
                                              std::max(0, int(gt_cy * float(grid_size))));
            const float cell_w_px = float(W) / float(grid_size);
            const float cell_h_px = float(H) / float(grid_size);

            std::cout << "\nSample " << s << " (" << name << ".bmp)\n"
                      << "  GT   : class=" << gt_class_name
                      << " box=(" << gt_x0 << "," << gt_y0 << ")-("
                      << gt_x1 << "," << gt_y1 << ")\n";

            // Axis-aligned-rect IoU on integer pixel coords.
            auto iou_with_gt = [&](int x0, int y0, int x1, int y1) -> float
            {
                const int ix0 = std::max(x0, gt_x0);
                const int iy0 = std::max(y0, gt_y0);
                const int ix1 = std::min(x1, gt_x1);
                const int iy1 = std::min(y1, gt_y1);
                if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
                const float inter = float(ix1 - ix0) * float(iy1 - iy0);
                const float a = float(x1 - x0) * float(y1 - y0);
                const float b = float(gt_x1 - gt_x0) * float(gt_y1 - gt_y0);
                return inter / std::max(a + b - inter, 1e-6f);
            };

            // Top-K visualization: red (top-1), orange (top-2), yellow (top-3),
            // cyan (best IoU vs GT — only if it isn't already in the top-3).
            const std::array<std::array<uint8_t, 3>, 3> rank_colors{{
                {220,  40,  40},  // top-1 red
                {255, 140,   0},  // top-2 orange
                {235, 200,   0},  // top-3 yellow
            }};
            const std::array<uint8_t, 3> best_iou_color{ 40, 200, 220};  // cyan

            const Index shown = std::min(Index(detections.size()), Index(5));
            if (shown == 0)
            {
                std::cout << "  Pred : (no boxes survived NMS)\n";
            }

            Index best_iou_rank = -1;
            float best_iou_value = 0.0f;
            for (Index r = 0; r < Index(detections.size()); ++r)
            {
                const auto& d = detections[size_t(r)];
                const int x0 = int(std::round(d.center_x - d.width  * 0.5f));
                const int y0 = int(std::round(d.center_y - d.height * 0.5f));
                const int x1 = int(std::round(d.center_x + d.width  * 0.5f)) - 1;
                const int y1 = int(std::round(d.center_y + d.height * 0.5f)) - 1;
                const float iou = iou_with_gt(x0, y0, x1, y1);
                if (iou > best_iou_value && d.class_id == gt_class)
                {
                    best_iou_value = iou;
                    best_iou_rank = r;
                }
            }

            for (Index r = 0; r < shown; ++r)
            {
                const auto& d = detections[size_t(r)];
                const int x0 = int(std::round(d.center_x - d.width  * 0.5f));
                const int y0 = int(std::round(d.center_y - d.height * 0.5f));
                const int x1 = int(std::round(d.center_x + d.width  * 0.5f)) - 1;
                const int y1 = int(std::round(d.center_y + d.height * 0.5f)) - 1;
                const float iou = iou_with_gt(x0, y0, x1, y1);

                const std::string pr_class_name =
                    (d.class_id >= 0 && size_t(d.class_id) < class_names.size())
                        ? class_names[size_t(d.class_id)] : std::to_string(d.class_id);

                if (r < Index(rank_colors.size()))
                    draw_rect_outline(img, x0, y0, x1, y1, rank_colors[size_t(r)], 2);

                std::cout << "  Top-" << (r + 1) << ": class=" << pr_class_name
                          << " score=" << d.score
                          << " iou=" << iou
                          << " box=(" << x0 << "," << y0 << ")-("
                          << x1 << "," << y1 << ")\n";
            }

            // Detections landing inside the GT-responsible cell.
            std::cout << "  GT cell: (col=" << gt_col_cell << ", row=" << gt_row_cell << ")\n";
            int gt_cell_hits = 0;
            for (Index r = 0; r < Index(detections.size()); ++r)
            {
                const auto& d = detections[size_t(r)];
                const int det_col = std::min(int(grid_size) - 1,
                                              std::max(0, int(d.center_x / cell_w_px)));
                const int det_row = std::min(int(grid_size) - 1,
                                              std::max(0, int(d.center_y / cell_h_px)));
                if (det_col == gt_col_cell && det_row == gt_row_cell)
                {
                    const std::string nm =
                        (d.class_id >= 0 && size_t(d.class_id) < class_names.size())
                            ? class_names[size_t(d.class_id)] : std::to_string(d.class_id);
                    std::cout << "    rank=" << (r + 1) << " in GT cell: class=" << nm
                              << " score=" << d.score
                              << " center=(" << d.center_x << ", " << d.center_y << ")"
                              << " size=(" << d.width << " x " << d.height << ")\n";
                    if (++gt_cell_hits >= 4) break;
                }
            }
            if (gt_cell_hits == 0)
                std::cout << "    (no detections in GT cell — model never fired the responsible cell)\n";

            if (best_iou_rank >= 0
                && best_iou_rank >= Index(rank_colors.size())
                && best_iou_value > 0.0f)
            {
                const auto& d = detections[size_t(best_iou_rank)];
                const int x0 = int(std::round(d.center_x - d.width  * 0.5f));
                const int y0 = int(std::round(d.center_y - d.height * 0.5f));
                const int x1 = int(std::round(d.center_x + d.width  * 0.5f)) - 1;
                const int y1 = int(std::round(d.center_y + d.height * 0.5f)) - 1;
                draw_rect_outline(img, x0, y0, x1, y1, best_iou_color, 2);

                std::cout << "  Best-IoU box is rank " << (best_iou_rank + 1)
                          << " (iou=" << best_iou_value << ", score=" << d.score
                          << ") — drawn in cyan\n";
            }
            else if (best_iou_rank >= 0)
            {
                std::cout << "  Best-IoU box is also top-" << (best_iou_rank + 1)
                          << " (iou=" << best_iou_value << ")\n";
            }

            // Synthetic: name by class (one file per class, always 3 files).
            // Real datasets: name by image stem so all 9 files are distinct.
            const std::string class_label =
                (gt_class >= 0 && size_t(gt_class) < class_names.size())
                    ? class_names[size_t(gt_class)] : std::to_string(gt_class);
            const std::string out_stem = is_synthetic
                ? ("annotated_" + class_label)
                : ("annotated_" + image_path.stem().string());
            const std::filesystem::path out_path = output_dir / (out_stem + ".bmp");
            write_bmp24_top_down(out_path, img);
            std::cout << "  -> " << out_path << "\n";
        }

        std::cout << "\nLegend: green = GT, red = top-1, orange = top-2, "
                  << "yellow = top-3, cyan = best-IoU-vs-GT (if outside top-3).\n";

        // ===== VOC mAP@0.5 =====
        // Standard 11-point interpolated AP per class, averaged to mAP.
        // GT boxes are taken from the YOLO .txt label files (original-image-normalized).
        // Predictions are decoded in letterbox space (416×416). Both are transformed to
        // letterbox-normalized coords before IoU matching so non-square images compare correctly.
        {
            std::cout << "\nComputing VOC mAP@0.5 on "
                      << selection_indices.size() << " validation images...\n";

            struct GtBox { int cls; float cx, cy, w, h; };
            struct Pred  { int img_k, cls; float score, cx, cy, w, h; };

            const int N_cls = int(dataset.get_classes_number());
            const int N_val = int(selection_indices.size());

            // Read image width/height from BMP/JPEG/PNG header without loading pixels.
            // Returns {0,0} on failure (unknown format or I/O error).
            auto read_image_dims = [](const std::filesystem::path& p) -> std::pair<int,int> {
                std::ifstream f(p, std::ios::binary);
                unsigned char h[30] = {};
                f.read(reinterpret_cast<char*>(h), 30);
                if (!f.gcount()) return {0, 0};
                // BMP
                if (h[0] == 'B' && h[1] == 'M') {
                    int w = 0, ht = 0;
                    std::memcpy(&w,  h + 18, 4);
                    std::memcpy(&ht, h + 22, 4);
                    if (ht < 0) ht = -ht;
                    return {ht, w};
                }
                // PNG
                if (h[0] == 0x89 && h[1] == 'P' && h[2] == 'N' && h[3] == 'G') {
                    int w  = (h[16]<<24)|(h[17]<<16)|(h[18]<<8)|h[19];
                    int ht = (h[20]<<24)|(h[21]<<16)|(h[22]<<8)|h[23];
                    return {ht, w};
                }
                // JPEG: scan for SOF0/SOF2 (FF C0 / FF C2)
                if (h[0] == 0xFF && h[1] == 0xD8) {
                    f.seekg(2);
                    for (int iter = 0; iter < 2000; ++iter) {
                        unsigned char m[2];
                        f.read(reinterpret_cast<char*>(m), 2);
                        if (f.gcount() < 2) break;
                        if (m[0] != 0xFF) { f.seekg(-1, std::ios::cur); continue; }
                        if (m[1] == 0xC0 || m[1] == 0xC2) {
                            unsigned char seg[7];
                            f.read(reinterpret_cast<char*>(seg), 7);
                            if (f.gcount() < 7) break;
                            return {(int(seg[3])<<8)|seg[4], (int(seg[5])<<8)|seg[6]};
                        }
                        unsigned char len[2];
                        f.read(reinterpret_cast<char*>(len), 2);
                        if (f.gcount() < 2) break;
                        int skip = ((int(len[0])<<8)|len[1]) - 2;
                        if (skip > 0) f.seekg(skip, std::ios::cur);
                    }
                }
                return {0, 0};
            };

            // Load all GT boxes transformed into letterbox-normalized space so they
            // can be directly compared with predictions (which are also in letterbox space).
            // GT .txt files use original-image-normalized coords; non-square VOC images
            // have a ~0.832 letterbox scale + padding, causing IoU underestimation otherwise.
            std::vector<std::vector<GtBox>> val_gt(N_val);
            for (int k = 0; k < N_val; ++k)
            {
                const Index s = selection_indices[size_t(k)];
                const std::filesystem::path img_path = dataset.get_image_path(s);
                std::filesystem::path lbl = labels_dir / img_path.filename();
                lbl.replace_extension(".txt");

                // Compute letterbox transform for this image.
                const auto [orig_H, orig_W] = read_image_dims(img_path);
                float lb_scale = 1.0f, lb_off_x = 0.0f, lb_off_y = 0.0f;
                if (orig_H > 0 && orig_W > 0) {
                    lb_scale = std::min(float(input_shape[0]) / float(orig_H),
                                        float(input_shape[1]) / float(orig_W));
                    lb_off_x = (float(input_shape[1]) - float(orig_W) * lb_scale) * 0.5f;
                    lb_off_y = (float(input_shape[0]) - float(orig_H) * lb_scale) * 0.5f;
                }
                const float inv_iW = 1.0f / float(input_shape[1]);
                const float inv_iH = 1.0f / float(input_shape[0]);

                std::ifstream f(lbl);
                int c; float cx, cy, w, h;
                while (f >> c >> cx >> cy >> w >> h) {
                    // Transform from original-image-normalized to letterbox-normalized.
                    const float lb_cx = (cx * float(orig_W) * lb_scale + lb_off_x) * inv_iW;
                    const float lb_cy = (cy * float(orig_H) * lb_scale + lb_off_y) * inv_iH;
                    const float lb_w  = w * float(orig_W) * lb_scale * inv_iW;
                    const float lb_h  = h * float(orig_H) * lb_scale * inv_iH;
                    val_gt[k].push_back({c, lb_cx, lb_cy, lb_w, lb_h});
                }
            }

            // Run inference on every validation image; store predictions per class.
            std::vector<std::vector<Pred>> cls_preds(N_cls);
            for (int k = 0; k < N_val; ++k)
            {
                const Index s = selection_indices[size_t(k)];
                dataset.fill_inputs({s}, {}, input_buffer.data(), false, false);
                std::copy(input_buffer.begin(), input_buffer.end(), input.data());

                std::vector<YoloDetection> dets;
                if (is_fpn)
                {
                    ForwardPropagation fp_m(1, &yolo_network);
                    const std::vector<TensorView> iv = {
                        TensorView(input.data(),
                                   {1, input.dimension(1), input.dimension(2), input.dimension(3)},
                                   Type::FP32)
                    };
                    yolo_network.forward_propagate(iv, fp_m, false);

                    std::vector<YoloFpnHead> heads;
                    std::vector<std::vector<float>> cpu_bufs;
                    const auto& all_layers = yolo_network.get_layers();
                    for (size_t li = 0; li < all_layers.size(); ++li)
                    {
                        if (!all_layers[li] || all_layers[li]->get_type() != LayerType::Detection)
                            continue;
                        const Shape hs = all_layers[li]->get_output_shape();
                        const TensorView view = fp_m.forward_slots[li].back();
                        const float* ptr = view.as<float>();
#ifdef OPENNN_HAS_CUDA
                        if (view.is_cuda())
                        {
                            cpu_bufs.emplace_back(size_t(view.size()));
                            cudaMemcpy(cpu_bufs.back().data(), ptr,
                                       size_t(view.size()) * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            ptr = cpu_bufs.back().data();
                        }
#endif
                        heads.push_back({ptr, hs[0],
                                         hs[2] / (5 + Index(N_cls)),
                                         Index(N_cls)});
                    }
                    dets = decode_yolo_fpn_detections(heads,
                               input_shape[0], input_shape[1],
                               input_shape[0], input_shape[1],
                               0.001f, 0.45f);
                }
                else
                {
                    const MatrixR outputs = yolo_network.calculate_outputs(input);
                    dets = decode_yolo_detections(outputs.data(), max_boxes,
                               input_shape[0], input_shape[1],
                               input_shape[0], input_shape[1]);
                }

                const float IW = float(input_shape[1]), IH = float(input_shape[0]);
                for (const auto& d : dets)
                {
                    if (d.class_id < 0 || d.class_id >= N_cls) continue;
                    cls_preds[d.class_id].push_back({k, int(d.class_id), d.score,
                        d.center_x / IW, d.center_y / IH,
                        d.width    / IW, d.height   / IH});
                }
            }

            // IoU between two boxes in cx/cy/w/h normalized coordinates.
            auto iou_box = [](float cx1, float cy1, float w1, float h1,
                              float cx2, float cy2, float w2, float h2) -> float
            {
                const float lx = std::max(cx1 - w1 * 0.5f, cx2 - w2 * 0.5f);
                const float ly = std::max(cy1 - h1 * 0.5f, cy2 - h2 * 0.5f);
                const float rx = std::min(cx1 + w1 * 0.5f, cx2 + w2 * 0.5f);
                const float ry = std::min(cy1 + h1 * 0.5f, cy2 + h2 * 0.5f);
                const float inter = std::max(0.f, rx - lx) * std::max(0.f, ry - ly);
                return inter / std::max(w1 * h1 + w2 * h2 - inter, 1e-6f);
            };

            float total_ap = 0.f;
            int classes_with_gt = 0;
            for (int c = 0; c < N_cls; ++c)
            {
                int n_gt = 0;
                for (const auto& gts : val_gt)
                    for (const auto& g : gts) if (g.cls == c) ++n_gt;
                if (n_gt == 0) continue;
                ++classes_with_gt;

                auto& preds = cls_preds[c];
                std::sort(preds.begin(), preds.end(),
                          [](const Pred& a, const Pred& b){ return a.score > b.score; });

                std::vector<std::vector<bool>> matched(N_val);
                for (int k = 0; k < N_val; ++k)
                    matched[k].assign(val_gt[k].size(), false);

                float cum_tp = 0.f, cum_fp = 0.f;
                std::vector<float> prec, rec;
                for (const auto& p : preds)
                {
                    float best_iou = 0.5f; // IoU threshold for a true positive
                    int best_gi = -1;
                    for (int gi = 0; gi < int(val_gt[p.img_k].size()); ++gi)
                    {
                        const GtBox& g = val_gt[p.img_k][size_t(gi)];
                        if (g.cls != c || matched[p.img_k][size_t(gi)]) continue;
                        const float iou = iou_box(p.cx, p.cy, p.w, p.h,
                                                   g.cx, g.cy, g.w, g.h);
                        if (iou > best_iou) { best_iou = iou; best_gi = gi; }
                    }
                    if (best_gi >= 0)
                    {
                        ++cum_tp;
                        matched[p.img_k][size_t(best_gi)] = true;
                    }
                    else
                        ++cum_fp;

                    prec.push_back(cum_tp / (cum_tp + cum_fp));
                    rec .push_back(cum_tp / float(n_gt));
                }

                // 11-point VOC interpolation: max precision at each recall level.
                float ap = 0.f;
                for (int ri = 0; ri <= 10; ++ri)
                {
                    const float r = float(ri) * 0.1f;
                    float max_p = 0.f;
                    for (size_t pi = 0; pi < prec.size(); ++pi)
                        if (rec[pi] >= r) max_p = std::max(max_p, prec[pi]);
                    ap += max_p;
                }
                ap /= 11.f;
                total_ap += ap;
                std::cout << "  " << std::left << std::setw(14)
                          << class_names[size_t(c)]
                          << "AP=" << std::fixed << std::setprecision(3) << ap
                          << "  (GT=" << n_gt << " pred=" << preds.size() << ")\n";
            }

            const float mAP = classes_with_gt > 0
                ? total_ap / float(classes_with_gt) : 0.f;
            std::cout << "mAP@0.5: " << std::fixed << std::setprecision(3) << mAP
                      << "  (" << classes_with_gt << "/" << N_cls
                      << " classes with GT in validation set)\n";
        }

        std::cout << "Bye!" << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
