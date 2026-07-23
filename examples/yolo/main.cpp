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

struct Image24
{
    int width = 0;
    int height = 0;
    vector<uint8_t> rgb;
};

void write_bmp24_top_down(const filesystem::path& path, const Image24& img)
{
    const int row_bytes_unpadded = img.width * 3;
    const int row_pad = (4 - row_bytes_unpadded % 4) % 4;
    const int row_stride = row_bytes_unpadded + row_pad;
    const int pixel_data_size = row_stride * img.height;
    const int file_size = 54 + pixel_data_size;

    vector<uint8_t> file(size_t(file_size), 0);

    file[0] = 'B'; file[1] = 'M';
    file[2] = uint8_t(file_size & 0xff);
    file[3] = uint8_t((file_size >> 8) & 0xff);
    file[4] = uint8_t((file_size >> 16) & 0xff);
    file[5] = uint8_t((file_size >> 24) & 0xff);
    file[10] = 54;
    file[14] = 40;
    file[18] = uint8_t(img.width & 0xff);
    file[19] = uint8_t((img.width >> 8) & 0xff);
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
            file[dst + size_t(x) * 3 + 0] = img.rgb[src + size_t(x) * 3 + 2];
            file[dst + size_t(x) * 3 + 1] = img.rgb[src + size_t(x) * 3 + 1];
            file[dst + size_t(x) * 3 + 2] = img.rgb[src + size_t(x) * 3 + 0];
        }
    }

    ofstream out(path, ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), file.size());
}

Image24 read_bmp24(const filesystem::path& path)
{
    ifstream in(path, ios::binary);
    vector<char> buf((istreambuf_iterator<char>(in)),
                          istreambuf_iterator<char>());
    if (buf.size() < 54 || buf[0] != 'B' || buf[1] != 'M')
        throw runtime_error("not a 24-bit BMP: " + path.string());

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
        throw runtime_error("only 24-bit BMP supported: " + path.string());

    const bool top_down = height_signed < 0;
    const int height = abs(height_signed);
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
            img.rgb[dst + size_t(x) * 3 + 0] = uint8_t(buf[src + size_t(x) * 3 + 2]);
            img.rgb[dst + size_t(x) * 3 + 1] = uint8_t(buf[src + size_t(x) * 3 + 1]);
            img.rgb[dst + size_t(x) * 3 + 2] = uint8_t(buf[src + size_t(x) * 3 + 0]);
        }
    }
    return img;
}

void draw_rect_outline(Image24& img,
                       int x0, int y0, int x1, int y1,
                       std::array<uint8_t, 3> color, int thickness = 2)
{
    if (x1 < x0) swap(x0, x1);
    if (y1 < y0) swap(y0, y1);

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

void write_synthetic_bmp(const filesystem::path& path,
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

void write_label(const filesystem::path& path,
                 int class_id, float cx, float cy, float w, float h)
{
    ofstream out(path);
    out << class_id << ' ' << cx << ' ' << cy << ' ' << w << ' ' << h << '\n';
}

void generate_synthetic_dataset(const filesystem::path& images_dir,
                                const filesystem::path& labels_dir,
                                int samples_per_class)
{
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    constexpr int image_size = 128;
    constexpr int block = 32;
    constexpr std::array<uint8_t, 3> background{255, 255, 255};
    const std::array<std::array<uint8_t, 3>, 3> colors{{
        {220,  40,  40},
        { 40,  60, 220},
        { 40, 200,  40},
    }};

    mt19937 rng(0xC0FFEE);
    uniform_int_distribution<int> pos_dist(0, image_size - block);

    int index = 0;
    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < samples_per_class; ++i)
        {
            const int top_x = pos_dist(rng);
            const int top_y = pos_dist(rng);

            const string name = "sample_" + to_string(index);
            const filesystem::path image_path = images_dir / (name + ".bmp");
            const filesystem::path label_path = labels_dir / (name + ".txt");

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

    ofstream classes(labels_dir / "classes.names");
    classes << "red\n" << "blue\n" << "green\n";
}

}

int main()
{
    try
    {
        cout << "OpenNN. YOLO Example." << endl;

        {
            const float grad_err = opennn::yolo_loss_gradient_check_cpu();
            cout << "YOLO loss check 1 (CPU gradient self-consistency): max rel err = "
                      << scientific << setprecision(3) << grad_err;
            cout << (grad_err < 1e-2f ? "  [PASS]\n" : "  [FAIL — forward/backward inconsistent!]\n");
        }

        {
            cout << "YOLO loss check 2 (expected values & gradient directions):\n";
            const float ev_err = opennn::yolo_loss_expected_value_check_cpu();
            cout << "  max absolute error vs. hand-computed: "
                      << scientific << setprecision(3) << ev_err;
            cout << (ev_err < 1e-4f ? "  [PASS]\n" : "  [FAIL — math objective is wrong!]\n");
        }

        set_seed(42);
        Configuration::instance().set(Device::CUDA, Type::FP32);

        const auto class_activation = YoloNetwork::ClassActivation::Sigmoid;

        const bool use_voc     = true;
        const bool use_raccoon = false;
        const bool use_coco    = false;

        const bool use_panet   = false;
        const bool use_csp     = true;
        const bool use_sppf    = false;

        const auto backbone = (use_coco || use_voc)
            ? (use_csp ? YoloNetwork::Backbone::CSPDarknet53 : YoloNetwork::Backbone::Darknet53)
            : YoloNetwork::Backbone::Vgg;

        const auto head_style = (use_coco || use_voc)
            ? (use_panet ? YoloNetwork::HeadStyle::PANet : YoloNetwork::HeadStyle::FPN)
            : YoloNetwork::HeadStyle::Single;

        const auto body_activation = YoloNetwork::BodyActivation::LeakyReLU;

        const filesystem::path voc_root = "/home/alvaromartin/VOCdevkit/VOC2007";
        const string voc_image_set = "trainval";

        const vector<string> voc_class_filter = {};


        const filesystem::path data_dir = use_voc     ? "yolo_voc_data"
                                             : use_raccoon ? "yolo_raccoon_data"
                                             : use_coco   ? "yolo_coco_data"
                                             :               "yolo_data";
        filesystem::create_directories(data_dir);

        filesystem::path images_dir;
        filesystem::path labels_dir;
        Index grid_size;
        Index boxes_per_cell;
        Shape input_shape;
        vector<std::array<float, 2>> anchors;

        if (use_voc)
        {
            images_dir = voc_root / "JPEGImages";
            labels_dir = data_dir / (voc_class_filter.empty() ? "voc_labels" : "voc_labels_filtered");
            const Index converted =
                YoloDataset::convert_voc_to_yolo(voc_root, voc_image_set, labels_dir, voc_class_filter);
            cout << "Converted " << converted
                      << " VOC samples to YOLO format in " << labels_dir << "\n";

            if (!voc_class_filter.empty())
            {
                const filesystem::path filtered_images_dir = data_dir / "voc_images_filtered";
                filesystem::create_directories(filtered_images_dir);
                Index linked = 0;
                for (const auto& entry : filesystem::directory_iterator(labels_dir))
                {
                    if (entry.path().extension() != ".txt") continue;
                    if (entry.path().stem().string() == "voc") continue;
                    const string stem = entry.path().stem().string();
                    for (const char* ext : {".jpg", ".jpeg", ".JPG", ".JPEG"})
                    {
                        const auto src = images_dir / (stem + ext);
                        if (filesystem::exists(src))
                        {
                            const auto dst = filtered_images_dir / (stem + ext);
                            if (!filesystem::exists(dst))
                                filesystem::create_symlink(src, dst);
                            ++linked;
                            break;
                        }
                    }
                }
                images_dir = filtered_images_dir;
                cout << "Filtered to " << linked << " images containing the requested classes.\n";
            }

            grid_size = 13;
            boxes_per_cell = 3;
            input_shape = Shape{416, 416, 3};
            anchors = {
                {0.024f, 0.034f}, {0.055f, 0.065f}, {0.089f, 0.139f},
                {0.195f, 0.197f}, {0.325f, 0.406f}, {0.827f, 0.767f}};
        }
        else if (use_coco)
        {
            labels_dir = "/home/alvaromartin/coco_mini_data/train2017_labels";
            images_dir = data_dir / "labeled_images";
            filesystem::create_directories(images_dir);
            const filesystem::path src_images =
                "/home/alvaromartin/coco_mini_data/train2017";
            for (const auto& entry : filesystem::directory_iterator(labels_dir))
            {
                if (entry.path().extension() != ".txt") continue;
                if (entry.path().stem().string().find("coco_mini") != string::npos) continue;
                for (const char* ext : {".jpg", ".jpeg", ".png"})
                {
                    const auto src = src_images / (entry.path().stem().string() + ext);
                    const auto dst = images_dir / src.filename();
                    if (filesystem::exists(src) && !filesystem::exists(dst))
                        filesystem::create_symlink(src, dst);
                }
            }

            grid_size      = 13;
            boxes_per_cell = 3;
            input_shape    = Shape{416, 416, 3};
            anchors = {
                {0.0240f, 0.0444f}, {0.0859f, 0.2730f}, {0.1625f, 0.4495f},
                {0.4037f, 0.3609f}, {0.4677f, 0.7767f}, {0.7996f, 0.8253f},
            };
        }
        else if (use_raccoon)
        {
            images_dir = "/home/artelnics/Documents/opennn/raccoon_dataset/images";
            labels_dir = "/home/artelnics/Documents/opennn/raccoon_data/labels";

            grid_size      = 13;
            boxes_per_cell = 3;
            input_shape    = Shape{416, 416, 3};
            anchors = {{0.334f, 0.476f}, {0.519f, 0.818f}, {0.807f, 0.852f}};
        }
        else
        {
            images_dir = data_dir / "images";
            labels_dir = data_dir / "labels";
            generate_synthetic_dataset(images_dir, labels_dir,                       256);

            grid_size      = 4;
            boxes_per_cell = 2;
            input_shape    = Shape{128, 128, 3};
            anchors        = {{0.25f, 0.25f}, {0.5f, 0.5f}};
        }

        const bool is_v3std     = (backbone == YoloNetwork::Backbone::DarknetTinyV3);
        const bool is_darknet53 = (backbone == YoloNetwork::Backbone::Darknet53);
        const bool is_csp53     = (backbone == YoloNetwork::Backbone::CSPDarknet53);
        const bool is_large_backbone = is_darknet53 || is_csp53;
        if (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
        {
            if (is_v3std)
            {
                if (use_voc)
                {
                    anchors = {
                        {0.024f, 0.031f}, {0.038f, 0.072f}, {0.079f, 0.055f},
                        {0.279f, 0.216f}, {0.375f, 0.476f}, {0.896f, 0.783f}};
                }
                else if (!use_coco)
                {
                    anchors = {
                        {0.08f, 0.08f}, {0.15f, 0.15f}, {0.25f, 0.25f},
                        {0.40f, 0.40f}, {0.55f, 0.55f}, {0.75f, 0.75f}};
                }
            }
            else
            {
                if (use_voc)
                {
                    anchors = {
                        {0.024f, 0.031f}, {0.038f, 0.072f}, {0.079f, 0.055f},
                        {0.072f, 0.147f}, {0.149f, 0.108f}, {0.142f, 0.286f},
                        {0.279f, 0.216f}, {0.375f, 0.476f}, {0.896f, 0.783f}};
                }
                else
                {
                    anchors = {
                        {0.05f, 0.05f}, {0.08f, 0.08f}, {0.12f, 0.12f},
                        {0.18f, 0.18f}, {0.25f, 0.25f}, {0.32f, 0.32f},
                        {0.40f, 0.40f}, {0.55f, 0.55f}, {0.75f, 0.75f}};
                }
            }
            boxes_per_cell = 3;
        }

        const vector<std::array<float, 2>> ctor_anchors =
            (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
                ? vector<std::array<float, 2>>{anchors[0], anchors[1], anchors[2]}
                : anchors;

        YoloDataset dataset(images_dir, labels_dir, input_shape,
                            grid_size, boxes_per_cell, ctor_anchors);

        if (head_style == YoloNetwork::HeadStyle::FPN || head_style == YoloNetwork::HeadStyle::PANet)
        {
            if (is_v3std)
            {
                const vector<Index> head_grids = {
                    grid_size,
                    grid_size * 2,
                };
                const vector<vector<std::array<float, 2>>> head_anchors = {
                    {anchors[3], anchors[4], anchors[5]},
                    {anchors[0], anchors[1], anchors[2]},
                };
                dataset.set_multi_scale_heads(head_grids, head_anchors);
            }
            else
            {
                const vector<Index> head_grids = {
                    grid_size,
                    grid_size * 2,
                    grid_size * 4
                };
                const vector<vector<std::array<float, 2>>> head_anchors = {
                    {anchors[6], anchors[7], anchors[8]},
                    {anchors[3], anchors[4], anchors[5]},
                    {anchors[0], anchors[1], anchors[2]},
                };
                dataset.set_multi_scale_heads(head_grids, head_anchors);
            }
        }

        YoloDataset::AugmentationConfig aug;
        if (use_voc || use_raccoon || use_coco)
        {
            aug.jitter      = 0.2f;
            aug.flip        = true;
            aug.exposure    = 1.5f;
            aug.saturation  = 1.5f;
            aug.hue         = 0.1f;
            aug.enabled     = true;
            aug.mosaic      = use_voc || use_coco;
        }
        else
        {
            aug.jitter      = 0.2f;
            aug.flip        = true;
            aug.exposure    = 1.2f;
            aug.saturation  = 1.0f;
            aug.hue         = 0.0f;
            aug.enabled     = false;
        }
        dataset.set_augmentation(aug);

        const double train_frac = use_raccoon ? 0.8 : 0.7;
        const double val_frac   = use_raccoon ? 0.2 : 0.3;
        dataset.split_samples_random(train_frac, val_frac, 0.0);


        const vector<std::array<float, 2>>& network_anchors =
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

        cout << "Device: " << (yolo_network.is_gpu() ? "GPU" : "CPU")
                  << "  " << device::gpu_info_string() << "\n";
        cout << "Network: backbone="
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

        if (head_style != YoloNetwork::HeadStyle::FPN && head_style != YoloNetwork::HeadStyle::PANet)
        {
            auto* nms_layer = dynamic_cast<NonMaxSuppression*>(
                yolo_network.get_layer("non_max_suppression_layer").get());
            if (nms_layer)
                nms_layer->set(yolo_network.get_layer(yolo_network.get_layers_number() - 2)->get_output_shape(),
                               boxes_per_cell,                0.0f,         0.4f,
                               "non_max_suppression_layer");
        }


        TrainingStrategy training_strategy(&yolo_network, &dataset);
        training_strategy.set_loss("Yolo");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("L2");
        training_strategy.get_loss()->set_yolo_lambda_noobj(1.0f);
        training_strategy.get_loss()->set_yolo_lambda_class(2.0f);
        training_strategy.get_loss()->set_yolo_focal_gamma(2.0f);
        training_strategy.get_loss()->set_yolo_obj_focal_gamma(2.0f);

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        const int batch_size = is_large_backbone ? 4 : 16;
        adam->set_batch_size(batch_size);
        adam->set_display_period(1);
        adam->set_gradient_clip_norm(0.1f);
        adam->set_maximum_validation_failures(use_coco ? 50 : (use_voc && is_large_backbone) ? 40 : use_voc ? 25 : use_raccoon ? 25 : 15);

        const bool resume_training = true;

        const string dataset_tag  = use_voc ? "voc" : use_raccoon ? "raccoon" : use_coco ? "coco" : "synth";
        const string filter_tag   = voc_class_filter.empty() ? "" :
            "_" + to_string(voc_class_filter.size()) + "cls";
        const string weights_filename = string("yolo_weights_") + dataset_tag + "_" +
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
            string("_bce_ig_bgfocal.bin");
        filesystem::path weights_path = data_dir / weights_filename;
        const filesystem::path legacy_weights = data_dir / "yolo_weights.bin";
        if (backbone == YoloNetwork::Backbone::Vgg
        &&  !filesystem::exists(weights_path)
        &&   filesystem::exists(legacy_weights))
            weights_path = legacy_weights;

        filesystem::path states_path = weights_path;
        states_path.replace_extension(".states.bin");

        const bool weights_exist = filesystem::exists(weights_path);
        if (weights_exist)
        {
            yolo_network.load_parameters_binary(weights_path);
            if (filesystem::exists(states_path))
                yolo_network.load_states_binary(states_path);
            cout << "\nLoaded weights from \"" << weights_path.string() << "\".\n";
        }

        const bool needs_darknet_backbone =
            (backbone == YoloNetwork::Backbone::DarknetTinyV3 ||
             backbone == YoloNetwork::Backbone::Darknet53 ||
             backbone == YoloNetwork::Backbone::CSPDarknet53) && !weights_exist;
        bool backbone_pretrained_loaded = false;
        if (needs_darknet_backbone)
        {
            const bool is53  = (backbone == YoloNetwork::Backbone::Darknet53);
            const bool iscsp = (backbone == YoloNetwork::Backbone::CSPDarknet53);
            const string darknet_filename = is53 ? "darknet53.conv.74"
                                               : iscsp ? "yolov4.conv.137"
                                               : "yolov3-tiny.weights";
            filesystem::path darknet_weights = data_dir / darknet_filename;
            if (!filesystem::exists(darknet_weights))
                darknet_weights = filesystem::path("yolo_voc_data") / darknet_filename;
            const Index n_backbone_convs = is53 ? 52 : iscsp ? 72 : 8;
            if (filesystem::exists(darknet_weights))
            {
                const Index loaded = YoloDataset::load_darknet_backbone(
                    yolo_network, darknet_weights, n_backbone_convs);
                cout << "Loaded " << loaded
                          << " backbone layers from " << darknet_weights << "\n";
                backbone_pretrained_loaded = true;
            }
            else
            {
                cout << "Darknet pretrained weights not found at " << darknet_weights << ".\n";
                if (is53)
                    cout << "Download darknet53.conv.74 from "
                              << "https://pjreddie.com/media/files/darknet53.conv.74 "
                              << "and place it in " << data_dir << "\n";
                else if (iscsp)
                    cout << "Download yolov4.conv.137 from "
                              << "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 "
                              << "and place it in " << data_dir << "\n";
                else
                    cout << "Download yolov3-tiny.weights from "
                              << "https://pjreddie.com/media/files/yolov3-tiny.weights "
                              << "and place it in " << data_dir << "\n";
                cout << "Training from scratch instead.\n";
            }
        }

        bool backbone_frozen = false;
        auto set_backbone_trainable = [&](bool trainable) {
            const string prefix = (backbone == YoloNetwork::Backbone::Darknet53)    ? "dn53_"  :
                                       (backbone == YoloNetwork::Backbone::CSPDarknet53) ? "csp53_" :
                                       (backbone == YoloNetwork::Backbone::DarknetTinyV3) ? "dntv3_" : "";
            if (prefix.empty()) return;
            for (auto& layer : yolo_network.get_layers())
                if (layer && layer->get_label().rfind(prefix, 0) == 0)
                    layer->set_is_trainable(trainable);
            cout << (trainable ? "Unfreezing" : "Freezing") << " backbone layers (" << prefix << "*).\n";
        };

        struct TrainingRound { float lr; int epochs; };
        const vector<TrainingRound> lr_schedule =
            use_coco         ? vector<TrainingRound>{{1e-4f, 300}, {3e-5f, 200}}                     :
            (use_voc && is_csp53)
                             ? vector<TrainingRound>{{1.25e-4f, 200}, {2.5e-5f, 150}, {1e-5f, 100}} :
            (use_voc && is_darknet53)
                             ? vector<TrainingRound>{{1.25e-4f, 150}, {2.5e-5f, 150}, {1e-5f, 100}} :
            use_voc          ? vector<TrainingRound>{{5e-4f, 150}, {1e-4f, 150}, {3e-5f, 100}}      :
            use_raccoon      ? vector<TrainingRound>{{5e-4f, 400}, {1e-4f, 300}}                     :
                               vector<TrainingRound>{{1e-3f, 200}};

        const filesystem::path epochs_file =
            data_dir / (weights_filename.substr(0, weights_filename.size() - 4) + "_epochs.txt");
        int epochs_done = 0;
        if (filesystem::exists(epochs_file))
        {
            ifstream ef(epochs_file);
            ef >> epochs_done;
        }
        cout << "Epochs completed so far: " << epochs_done << "\n";

        if (resume_training || !filesystem::exists(weights_path))
        {
            int cumulative = 0;
            for (const TrainingRound& rnd : lr_schedule)
            {
                const int round_end = cumulative + rnd.epochs;
                if (epochs_done >= round_end) { cumulative = round_end; continue; }

                if (backbone_frozen && cumulative >= 5) {
                    set_backbone_trainable(true);
                    backbone_frozen = false;
                }

                const int to_run = round_end - epochs_done;
                adam->set_learning_rate(rnd.lr);
                adam->set_maximum_epochs(to_run);
                cout << "\nTraining: lr=" << rnd.lr
                          << " for " << to_run << " epochs"
                          << " (target epoch " << round_end << ").\n";
                const auto train_result = training_strategy.train();
                epochs_done += static_cast<int>(train_result.get_epochs_number());
                cumulative   = round_end;

                yolo_network.save_parameters_binary(weights_path);
                yolo_network.save_states_binary(states_path);
                { ofstream ef(epochs_file); ef << epochs_done; }
                cout << "Checkpoint saved: " << epochs_done << " total epochs.\n";
            }
            cout << "Training complete (" << epochs_done << " total epochs).\n";
        }

        const bool is_fpn = (head_style == YoloNetwork::HeadStyle::FPN ||
                              head_style == YoloNetwork::HeadStyle::PANet);


        const filesystem::path output_dir = data_dir / "annotated";
        if (filesystem::exists(output_dir))
            for (const auto& entry : filesystem::directory_iterator(output_dir))
                filesystem::remove(entry.path());
        filesystem::create_directories(output_dir);

        const Index image_floats = input_shape[0] * input_shape[1] * input_shape[2];
        vector<float> input_buffer(size_t(image_floats), 0.0f);
        Tensor4 input(1, input_shape[0], input_shape[1], input_shape[2]);

        const Index max_boxes = grid_size * grid_size * boxes_per_cell;
        const vector<string>& class_names = dataset.get_class_names();
        const std::array<std::array<uint8_t, 3>, 2> box_color_by_role{{
            { 30, 200,  30},
            {220,  40,  40},
        }};

        const vector<Index> selection_indices = dataset.get_sample_indices("Validation");
        const Index num_classes = dataset.get_classes_number();
        vector<Index> vis_indices;
        const bool is_synthetic = !use_voc && !use_raccoon && !use_coco;
        const Index max_vis = is_synthetic ? num_classes : Index(9);
        if (is_synthetic)
        {
            vector<bool> class_found(size_t(num_classes), false);
            constexpr int spc = 256;
            for (Index idx : selection_indices)
            {
                const string stem = dataset.get_image_path(idx).stem().string();
                const auto under = stem.rfind('_');
                if (under == string::npos) continue;
                const int n = stoi(stem.substr(under + 1));
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
            for (Index k = 0; k < max_vis; ++k)
            {
                const Index pos = k * Index(selection_indices.size()) / max_vis;
                vis_indices.push_back(selection_indices[size_t(pos)]);
            }
        }
        const Index samples_to_visualize = Index(vis_indices.size());
        cout << "\nVisualizing " << samples_to_visualize
                  << " held-out (validation) samples — model has never seen these. "
                  << "Annotated BMPs in " << output_dir << ":\n";

        for (Index k = 0; k < samples_to_visualize; ++k)
        {
            const Index s = vis_indices[size_t(k)];

            dataset.fill_inputs({s}, {}, input_buffer.data(),
                                FillMode::Inference,                -1);
            copy(input_buffer.begin(), input_buffer.end(), input.data());

            vector<YoloDetection> detections;

            if (is_fpn)
            {
                ForwardPropagation forward_propagation(               1, &yolo_network);
                const vector<TensorView> input_views = {
                    TensorView(input.data(),
                               {1, input.dimension(1), input.dimension(2), input.dimension(3)},
                               Type::FP32)
                };
                yolo_network.forward_propagate(input_views, forward_propagation,                 false);

                vector<YoloFpnHead> fpn_heads;
                vector<vector<float>> fpn_cpu_buffers;
                const auto& layers = yolo_network.get_layers();
                for (size_t li = 0; li < layers.size(); ++li)
                {
                    if (!layers[li] || layers[li]->get_type() != LayerType::Detection) continue;
                    const Shape head_shape = layers[li]->get_output_shape();
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
                                best_p = max(best_p, h.data[base + 5 + cl]);
                            const float score = obj * best_p;
                            max_obj   = max(max_obj,   obj);
                            max_score = max(max_score, score);
                            if (score >= 0.01f)  ++above_001;
                            if (score >= 0.1f)   ++above_01;
                            if (score >= 0.25f)  ++above_025;
                        }
                    }
                    cout << "  Raw : max_obj=" << max_obj
                              << " max_score=" << max_score
                              << " boxes≥0.01:" << above_001
                              << " ≥0.1:" << above_01
                              << " ≥0.25:" << above_025 << "\n";
                }

                detections = decode_yolo_fpn_detections(
                    fpn_heads,
                                        input_shape[0],
                                       input_shape[1],
                                       input_shape[0],
                                      input_shape[1],
                                             0.001f,
                                      0.45f);
            }
            else
            {
                const MatrixR outputs = yolo_network.calculate_outputs(input);
                detections = decode_yolo_detections(
                    outputs.data(), max_boxes,
                                        input_shape[0],
                                       input_shape[1],
                                       input_shape[0],
                                      input_shape[1]);
            }

            const filesystem::path image_path = dataset.get_image_path(s);
            const string name = image_path.stem().string();
            filesystem::path label_path = labels_dir / image_path.filename();
            label_path.replace_extension(".txt");

            struct GtBox { int cls; float cx, cy, w, h; };
            vector<GtBox> gt_boxes;
            {
                ifstream lf(label_path);
                int c; float cx, cy, w, h;
                while (lf >> c >> cx >> cy >> w >> h)
                    gt_boxes.push_back({c, cx, cy, w, h});
            }
            const int   gt_class = gt_boxes.empty() ? 0     : gt_boxes[0].cls;
            const float gt_cx    = gt_boxes.empty() ? 0.5f  : gt_boxes[0].cx;
            const float gt_cy    = gt_boxes.empty() ? 0.5f  : gt_boxes[0].cy;
            const float gt_w     = gt_boxes.empty() ? 0.0f  : gt_boxes[0].w;
            const float gt_h     = gt_boxes.empty() ? 0.0f  : gt_boxes[0].h;

            const int canvas_W = int(input_shape[1]);
            const int canvas_H = int(input_shape[0]);
            const int W = canvas_W, H = canvas_H;

            Image24 img;
            float lb_scale = 1.0f, lb_pad_x = 0.0f, lb_pad_y = 0.0f;
            int orig_W = canvas_W, orig_H = canvas_H;

            if (image_path.extension() == ".bmp" || image_path.extension() == ".BMP")
            {
                img    = read_bmp24(image_path);
                orig_W = img.width;
                orig_H = img.height;
            }
            else
            {
                const auto raw = opennn::load_image(image_path);
                orig_H  = int(raw.dimension(0));
                orig_W  = int(raw.dimension(1));
                lb_scale = min(float(canvas_W) / float(orig_W),
                                    float(canvas_H) / float(orig_H));
                lb_pad_x = (float(canvas_W) - float(orig_W) * lb_scale) * 0.5f;
                lb_pad_y = (float(canvas_H) - float(orig_H) * lb_scale) * 0.5f;
                const int scaled_W = int(round(float(orig_W) * lb_scale));
                const int scaled_H = int(round(float(orig_H) * lb_scale));
                const int off_x    = int(round(lb_pad_x));
                const int off_y    = int(round(lb_pad_y));

                img.width = canvas_W; img.height = canvas_H;
                img.rgb.assign(size_t(canvas_W * canvas_H * 3), 128);

                const int ch = int(raw.dimension(2));
                for (int py = 0; py < scaled_H; ++py)
                {
                    const int src_y = min(int(float(py) / lb_scale + 0.5f), orig_H - 1);
                    for (int px = 0; px < scaled_W; ++px)
                    {
                        const int src_x = min(int(float(px) / lb_scale + 0.5f), orig_W - 1);
                        const int dst_y = py + off_y, dst_x = px + off_x;
                        if (dst_y < 0 || dst_y >= canvas_H || dst_x < 0 || dst_x >= canvas_W) continue;
                        const float r = raw(src_y, src_x, 0);
                        const float g = ch > 1 ? raw(src_y, src_x, 1) : r;
                        const float b = ch > 2 ? raw(src_y, src_x, 2) : r;
                        const size_t idx = size_t((dst_y * canvas_W + dst_x) * 3);
                        img.rgb[idx]     = uint8_t(clamp(int(r), 0, 255));
                        img.rgb[idx + 1] = uint8_t(clamp(int(g), 0, 255));
                        img.rgb[idx + 2] = uint8_t(clamp(int(b), 0, 255));
                    }
                }
            }

            auto lb_px = [&](float cx, float cy, float w, float h,
                              int& x0, int& y0, int& x1, int& y1)
            {
                const float cx_px = cx * float(orig_W) * lb_scale + lb_pad_x;
                const float cy_px = cy * float(orig_H) * lb_scale + lb_pad_y;
                const float hw    = w * float(orig_W) * lb_scale * 0.5f;
                const float hh    = h * float(orig_H) * lb_scale * 0.5f;
                x0 = int(round(cx_px - hw));
                y0 = int(round(cy_px - hh));
                x1 = int(round(cx_px + hw)) - 1;
                y1 = int(round(cy_px + hh)) - 1;
            };

            int gt_x0, gt_y0, gt_x1, gt_y1;
            lb_px(gt_cx, gt_cy, gt_w, gt_h, gt_x0, gt_y0, gt_x1, gt_y1);

            for (const auto& gb : gt_boxes)
            {
                int x0, y0, x1, y1;
                lb_px(gb.cx, gb.cy, gb.w, gb.h, x0, y0, x1, y1);
                draw_rect_outline(img, x0, y0, x1, y1, box_color_by_role[0], 2);
            }

            sort(detections.begin(), detections.end(),
                      [](const YoloDetection& a, const YoloDetection& b)
                      { return a.score > b.score; });

            const string gt_class_name =
                (gt_class >= 0 && size_t(gt_class) < class_names.size())
                    ? class_names[size_t(gt_class)] : to_string(gt_class);

            const int gt_col_cell = min(int(grid_size) - 1,
                                              max(0, int(gt_cx * float(grid_size))));
            const int gt_row_cell = min(int(grid_size) - 1,
                                              max(0, int(gt_cy * float(grid_size))));
            const float cell_w_px = float(W) / float(grid_size);
            const float cell_h_px = float(H) / float(grid_size);

            cout << "\nSample " << s << " (" << name << ".bmp)\n"
                      << "  GT   : class=" << gt_class_name
                      << " box=(" << gt_x0 << "," << gt_y0 << ")-("
                      << gt_x1 << "," << gt_y1 << ")\n";

            auto iou_with_gt = [&](int x0, int y0, int x1, int y1) -> float
            {
                const int ix0 = max(x0, gt_x0);
                const int iy0 = max(y0, gt_y0);
                const int ix1 = min(x1, gt_x1);
                const int iy1 = min(y1, gt_y1);
                if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
                const float inter = float(ix1 - ix0) * float(iy1 - iy0);
                const float a = float(x1 - x0) * float(y1 - y0);
                const float b = float(gt_x1 - gt_x0) * float(gt_y1 - gt_y0);
                return inter / max(a + b - inter, 1e-6f);
            };

            const std::array<std::array<uint8_t, 3>, 3> rank_colors{{
                {220,  40,  40},
                {255, 140,   0},
                {235, 200,   0},
            }};
            const std::array<uint8_t, 3> best_iou_color{ 40, 200, 220};

            const Index shown = min(Index(detections.size()), Index(5));
            if (shown == 0)
            {
                cout << "  Pred : (no boxes survived NMS)\n";
            }

            Index best_iou_rank = -1;
            float best_iou_value = 0.0f;
            for (Index r = 0; r < Index(detections.size()); ++r)
            {
                const auto& d = detections[size_t(r)];
                const int x0 = int(round(d.center_x - d.width  * 0.5f));
                const int y0 = int(round(d.center_y - d.height * 0.5f));
                const int x1 = int(round(d.center_x + d.width  * 0.5f)) - 1;
                const int y1 = int(round(d.center_y + d.height * 0.5f)) - 1;
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
                const int x0 = int(round(d.center_x - d.width  * 0.5f));
                const int y0 = int(round(d.center_y - d.height * 0.5f));
                const int x1 = int(round(d.center_x + d.width  * 0.5f)) - 1;
                const int y1 = int(round(d.center_y + d.height * 0.5f)) - 1;
                const float iou = iou_with_gt(x0, y0, x1, y1);

                const string pr_class_name =
                    (d.class_id >= 0 && size_t(d.class_id) < class_names.size())
                        ? class_names[size_t(d.class_id)] : to_string(d.class_id);

                if (r < Index(rank_colors.size()))
                    draw_rect_outline(img, x0, y0, x1, y1, rank_colors[size_t(r)], 2);

                cout << "  Top-" << (r + 1) << ": class=" << pr_class_name
                          << " score=" << d.score
                          << " iou=" << iou
                          << " box=(" << x0 << "," << y0 << ")-("
                          << x1 << "," << y1 << ")\n";
            }

            cout << "  GT cell: (col=" << gt_col_cell << ", row=" << gt_row_cell << ")\n";
            int gt_cell_hits = 0;
            for (Index r = 0; r < Index(detections.size()); ++r)
            {
                const auto& d = detections[size_t(r)];
                const int det_col = min(int(grid_size) - 1,
                                              max(0, int(d.center_x / cell_w_px)));
                const int det_row = min(int(grid_size) - 1,
                                              max(0, int(d.center_y / cell_h_px)));
                if (det_col == gt_col_cell && det_row == gt_row_cell)
                {
                    const string nm =
                        (d.class_id >= 0 && size_t(d.class_id) < class_names.size())
                            ? class_names[size_t(d.class_id)] : to_string(d.class_id);
                    cout << "    rank=" << (r + 1) << " in GT cell: class=" << nm
                              << " score=" << d.score
                              << " center=(" << d.center_x << ", " << d.center_y << ")"
                              << " size=(" << d.width << " x " << d.height << ")\n";
                    if (++gt_cell_hits >= 4) break;
                }
            }
            if (gt_cell_hits == 0)
                cout << "    (no detections in GT cell — model never fired the responsible cell)\n";

            if (best_iou_rank >= 0
                && best_iou_rank >= Index(rank_colors.size())
                && best_iou_value > 0.0f)
            {
                const auto& d = detections[size_t(best_iou_rank)];
                const int x0 = int(round(d.center_x - d.width  * 0.5f));
                const int y0 = int(round(d.center_y - d.height * 0.5f));
                const int x1 = int(round(d.center_x + d.width  * 0.5f)) - 1;
                const int y1 = int(round(d.center_y + d.height * 0.5f)) - 1;
                draw_rect_outline(img, x0, y0, x1, y1, best_iou_color, 2);

                cout << "  Best-IoU box is rank " << (best_iou_rank + 1)
                          << " (iou=" << best_iou_value << ", score=" << d.score
                          << ") — drawn in cyan\n";
            }
            else if (best_iou_rank >= 0)
            {
                cout << "  Best-IoU box is also top-" << (best_iou_rank + 1)
                          << " (iou=" << best_iou_value << ")\n";
            }

            const string class_label =
                (gt_class >= 0 && size_t(gt_class) < class_names.size())
                    ? class_names[size_t(gt_class)] : to_string(gt_class);
            const string out_stem = is_synthetic
                ? ("annotated_" + class_label)
                : ("annotated_" + image_path.stem().string());
            const filesystem::path out_path = output_dir / (out_stem + ".bmp");
            write_bmp24_top_down(out_path, img);
            cout << "  -> " << out_path << "\n";
        }

        cout << "\nLegend: green = GT, red = top-1, orange = top-2, "
                  << "yellow = top-3, cyan = best-IoU-vs-GT (if outside top-3).\n";

        {
            cout << "\nComputing VOC mAP@0.5 on "
                      << selection_indices.size() << " validation images...\n";

            struct GtBox { int cls; float cx, cy, w, h; };
            struct Pred  { int img_k, cls; float score, cx, cy, w, h; };

            const int N_cls = int(dataset.get_classes_number());
            const int N_val = int(selection_indices.size());

            auto read_image_dims = [](const filesystem::path& p) -> pair<int,int> {
                ifstream f(p, ios::binary);
                unsigned char h[30] = {};
                f.read(reinterpret_cast<char*>(h), 30);
                if (!f.gcount()) return {0, 0};
                if (h[0] == 'B' && h[1] == 'M') {
                    int w = 0, ht = 0;
                    memcpy(&w,  h + 18, 4);
                    memcpy(&ht, h + 22, 4);
                    if (ht < 0) ht = -ht;
                    return {ht, w};
                }
                if (h[0] == 0x89 && h[1] == 'P' && h[2] == 'N' && h[3] == 'G') {
                    int w  = (h[16]<<24)|(h[17]<<16)|(h[18]<<8)|h[19];
                    int ht = (h[20]<<24)|(h[21]<<16)|(h[22]<<8)|h[23];
                    return {ht, w};
                }
                if (h[0] == 0xFF && h[1] == 0xD8) {
                    f.seekg(2);
                    for (int iter = 0; iter < 2000; ++iter) {
                        unsigned char m[2];
                        f.read(reinterpret_cast<char*>(m), 2);
                        if (f.gcount() < 2) break;
                        if (m[0] != 0xFF) { f.seekg(-1, ios::cur); continue; }
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
                        if (skip > 0) f.seekg(skip, ios::cur);
                    }
                }
                return {0, 0};
            };

            vector<vector<GtBox>> val_gt(N_val);
            for (int k = 0; k < N_val; ++k)
            {
                const Index s = selection_indices[size_t(k)];
                const filesystem::path img_path = dataset.get_image_path(s);
                filesystem::path lbl = labels_dir / img_path.filename();
                lbl.replace_extension(".txt");

                const auto [orig_H, orig_W] = read_image_dims(img_path);
                float lb_scale = 1.0f, lb_off_x = 0.0f, lb_off_y = 0.0f;
                if (orig_H > 0 && orig_W > 0) {
                    lb_scale = min(float(input_shape[0]) / float(orig_H),
                                        float(input_shape[1]) / float(orig_W));
                    lb_off_x = (float(input_shape[1]) - float(orig_W) * lb_scale) * 0.5f;
                    lb_off_y = (float(input_shape[0]) - float(orig_H) * lb_scale) * 0.5f;
                }
                const float inv_iW = 1.0f / float(input_shape[1]);
                const float inv_iH = 1.0f / float(input_shape[0]);

                ifstream f(lbl);
                int c; float cx, cy, w, h;
                while (f >> c >> cx >> cy >> w >> h) {
                    const float lb_cx = (cx * float(orig_W) * lb_scale + lb_off_x) * inv_iW;
                    const float lb_cy = (cy * float(orig_H) * lb_scale + lb_off_y) * inv_iH;
                    const float lb_w  = w * float(orig_W) * lb_scale * inv_iW;
                    const float lb_h  = h * float(orig_H) * lb_scale * inv_iH;
                    val_gt[k].push_back({c, lb_cx, lb_cy, lb_w, lb_h});
                }
            }

            vector<vector<Pred>> cls_preds(N_cls);
            for (int k = 0; k < N_val; ++k)
            {
                const Index s = selection_indices[size_t(k)];
                dataset.fill_inputs({s}, {}, input_buffer.data(), FillMode::Inference, -1);
                copy(input_buffer.begin(), input_buffer.end(), input.data());

                vector<YoloDetection> dets;
                if (is_fpn)
                {
                    ForwardPropagation fp_m(1, &yolo_network);
                    const vector<TensorView> iv = {
                        TensorView(input.data(),
                                   {1, input.dimension(1), input.dimension(2), input.dimension(3)},
                                   Type::FP32)
                    };
                    yolo_network.forward_propagate(iv, fp_m, false);

                    vector<YoloFpnHead> heads;
                    vector<vector<float>> cpu_bufs;
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

            auto iou_box = [](float cx1, float cy1, float w1, float h1,
                              float cx2, float cy2, float w2, float h2) -> float
            {
                const float lx = max(cx1 - w1 * 0.5f, cx2 - w2 * 0.5f);
                const float ly = max(cy1 - h1 * 0.5f, cy2 - h2 * 0.5f);
                const float rx = min(cx1 + w1 * 0.5f, cx2 + w2 * 0.5f);
                const float ry = min(cy1 + h1 * 0.5f, cy2 + h2 * 0.5f);
                const float inter = max(0.f, rx - lx) * max(0.f, ry - ly);
                return inter / max(w1 * h1 + w2 * h2 - inter, 1e-6f);
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
                sort(preds.begin(), preds.end(),
                          [](const Pred& a, const Pred& b){ return a.score > b.score; });

                vector<vector<bool>> matched(N_val);
                for (int k = 0; k < N_val; ++k)
                    matched[k].assign(val_gt[k].size(), false);

                float cum_tp = 0.f, cum_fp = 0.f;
                vector<float> prec, rec;
                for (const auto& p : preds)
                {
                    float best_iou = 0.5f;
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

                float ap = 0.f;
                for (int ri = 0; ri <= 10; ++ri)
                {
                    const float r = float(ri) * 0.1f;
                    float max_p = 0.f;
                    for (size_t pi = 0; pi < prec.size(); ++pi)
                        if (rec[pi] >= r) max_p = max(max_p, prec[pi]);
                    ap += max_p;
                }
                ap /= 11.f;
                total_ap += ap;
                cout << "  " << left << setw(14)
                          << class_names[size_t(c)]
                          << "AP=" << fixed << setprecision(3) << ap
                          << "  (GT=" << n_gt << " pred=" << preds.size() << ")\n";
            }

            const float mAP = classes_with_gt > 0
                ? total_ap / float(classes_with_gt) : 0.f;
            cout << "mAP@0.5: " << fixed << setprecision(3) << mAP
                      << "  (" << classes_with_gt << "/" << N_cls
                      << " classes with GT in validation set)\n";
        }

        cout << "Bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
