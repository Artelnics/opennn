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
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/configuration.h"
#include "../../opennn/forward_propagation.h"
#include "../../opennn/layer.h"
#include "../../opennn/non_max_suppression_layer.h"
#include "../../opennn/random_utilities.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/yolo_dataset.h"

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
    constexpr std::array<uint8_t, 3> background{240, 240, 240};
    const std::array<std::array<uint8_t, 3>, 2> colors{{
        {220,  40,  40},   // class 0 = "red"
        { 40,  60, 220},   // class 1 = "blue"
    }};

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> pos_dist(0, image_size - block);

    int index = 0;
    for (int c = 0; c < 2; ++c)
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
    classes << "red\n" << "blue\n";
}

}

int main()
{
    try
    {
        std::cout << "OpenNN. YOLO Example." << std::endl;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        // ===== Configuration toggles =====
        //
        // Two independent switches: backbone architecture and dataset source.
        // Keep both at their defaults to reproduce the Phase 2 baseline; flip
        // them to exercise the Phase 3 additions.
        //
        // PASCAL VOC: download VOCdevkit/VOC2007 and point voc_root at it.
        // The converter writes YOLO-format labels into <data_dir>/voc_labels
        // (only the first time; subsequent runs reuse them).
        // const auto backbone = YoloNetwork::Backbone::Vgg;
        const auto backbone = YoloNetwork::Backbone::DarknetTiny;

        // Softmax = mutually-exclusive classes (VOC, synthetic). Sigmoid =
        // independent per-class labels (YOLOv3-style, multi-label datasets).
        // Default stays Softmax so existing baselines reproduce exactly.
        const auto class_activation = YoloNetwork::ClassActivation::Softmax;
        // const auto class_activation = YoloNetwork::ClassActivation::Sigmoid;

        // FPN (YOLO v3): 3 detection heads at strides 32/16/8 with top-down
        // upsample+concatenation. Requires Backbone::DarknetTiny and 9 anchors
        // (3 per scale, smallest→stride-8, largest→stride-32). Cross-scale
        // NMS for inference runs via decode_yolo_fpn_detections below.
        const auto head_style = YoloNetwork::HeadStyle::FPN;
        // const auto head_style = YoloNetwork::HeadStyle::Single;

        // ReLU = Phase 1/2 default (preserves saved weights). LeakyReLU =
        // Darknet/YOLO-v3 convention (slope 0.1). Switching changes forward
        // outputs even at identical parameters, so the weights filename
        // suffixes with "_leaky" to avoid silently reloading ReLU-trained
        // weights under LeakyReLU semantics.
        const auto body_activation = YoloNetwork::BodyActivation::LeakyReLU;
        // const auto body_activation = YoloNetwork::BodyActivation::ReLU;

        const bool use_voc = false;
        const std::filesystem::path voc_root = "/home/alvaromartin/VOCdevkit/VOC2007";
        const std::string voc_image_set = "trainval_small";

        // ===== Dataset =====

        const std::filesystem::path data_dir = use_voc ? "yolo_voc_data" : "yolo_data";
        std::filesystem::path images_dir;
        std::filesystem::path labels_dir;
        Index grid_size;
        Index boxes_per_cell;
        Shape input_shape;
        std::vector<std::array<float, 2>> anchors;

        if (use_voc)
        {
            images_dir = voc_root / "JPEGImages";
            labels_dir = data_dir / "voc_labels";
            const Index converted =
                YoloDataset::convert_voc_to_yolo(voc_root, voc_image_set, labels_dir);
            std::cout << "Converted " << converted
                      << " VOC samples to YOLO format in " << labels_dir << "\n";

            grid_size = 13;
            boxes_per_cell = 5;
            input_shape = Shape{416, 416, 3};
            // Standard YOLOv2 k-means anchors on VOC (in image-fraction units).
            anchors = {{0.057f, 0.075f}, {0.169f, 0.130f}, {0.330f, 0.262f},
                       {0.279f, 0.640f}, {0.778f, 0.808f}};
        }
        else
        {
            // Synthetic dataset: 64x64 BMPs, one colored block per image, two classes.
            images_dir = data_dir / "images";
            labels_dir = data_dir / "labels";
            generate_synthetic_dataset(images_dir, labels_dir, /*samples_per_class=*/256);

            // Grid 4x4 over 128x128 input requires input H/W == grid*32.
            // The minimal Vgg builder ends with a 3x3 conv, so grid_size must be >= 3.
            grid_size = 4;
            boxes_per_cell = 2;
            input_shape = Shape{128, 128, 3};
            anchors = {{0.25f, 0.25f}, {0.5f, 0.5f}};
        }

        // FPN needs 9 anchors, 3 per scale. Override the dataset-builder anchors
        // for the smoke path. boxes_per_head ends up = 3 across all heads.
        if (head_style == YoloNetwork::HeadStyle::FPN)
        {
            anchors = {
                {0.05f, 0.05f}, {0.08f, 0.08f}, {0.12f, 0.12f},  // small head
                {0.18f, 0.18f}, {0.25f, 0.25f}, {0.32f, 0.32f},  // medium head
                {0.40f, 0.40f}, {0.55f, 0.55f}, {0.75f, 0.75f}}; // large head
            boxes_per_cell = 3;
        }

        // Dataset constructor validates anchors.size() == boxes_per_cell. In
        // FPN mode the single-scale path is unused after set_multi_scale_heads,
        // so any 3 anchors satisfy the constructor.
        const std::vector<std::array<float, 2>> ctor_anchors =
            head_style == YoloNetwork::HeadStyle::FPN
                ? std::vector<std::array<float, 2>>{anchors[0], anchors[1], anchors[2]}
                : anchors;

        YoloDataset dataset(images_dir, labels_dir, input_shape,
                            grid_size, boxes_per_cell, ctor_anchors);

        // FPN dataset configuration: 3 grid scales (stride 32 / 16 / 8) and
        // 3 anchor groups (small→medium→large by area) — must match the
        // YoloNetwork FPN constructor's auto-sorted assignment.
        if (head_style == YoloNetwork::HeadStyle::FPN)
        {
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

        // Class signal in this demo is color (red vs blue) — disable hue/sat jitter
        // so we don't erase it. Geometric augmentation (flip + crop) is what we need:
        // it attacks position-conditional overfit visible in earlier demo runs.
        YoloDataset::AugmentationConfig aug;
        aug.jitter = 0.2f;
        aug.flip = true;
        aug.exposure = 1.2f;
        aug.saturation = 1.0f;
        aug.hue = 0.0f;
        aug.enabled = false;
        dataset.set_augmentation(aug);

        // 70% training, 30% selection (held-out) — gives a generalization signal:
        // TrainingStrategy reports both training_error and selection_error per epoch,
        // and the visualization at the end runs inference on samples the model has
        // never seen during training.
        dataset.split_samples_random(0.7, 0.3, 0.0);

        // Network.

        // Network anchors: FPN needs all 9 (the dataset only stores its 3
        // single-scale ctor anchors after multi-scale config). Single-head
        // continues to use the dataset's anchor list.
        const std::vector<std::array<float, 2>>& network_anchors =
            head_style == YoloNetwork::HeadStyle::FPN ? anchors : dataset.get_anchors();

        YoloNetwork yolo_network(input_shape,
                                 dataset.get_classes_number(),
                                 network_anchors,
                                 grid_size,
                                 backbone,
                                 class_activation,
                                 head_style,
                                 body_activation);

        std::cout << "Network: backbone="
                  << (backbone == YoloNetwork::Backbone::Vgg ? "Vgg" : "DarknetTiny")
                  << ", class_activation="
                  << (class_activation == YoloNetwork::ClassActivation::Sigmoid ? "Sigmoid" : "Softmax")
                  << ", head_style="
                  << (head_style == YoloNetwork::HeadStyle::FPN ? "FPN" : "Single")
                  << ", body_activation="
                  << (body_activation == YoloNetwork::BodyActivation::LeakyReLU ? "LeakyReLU" : "ReLU")
                  << ", layers=" << yolo_network.get_layers_number()
                  << ", parameters=" << yolo_network.get_parameters_number() << "\n";

        // Loosen the NMS confidence threshold so the demo prints something visible
        // even on an under-trained tiny model. FPN mode has no NMS layer —
        // cross-scale NMS happens externally in decode_yolo_fpn_detections, so
        // skip the NMS configuration in that case.
        if (head_style != YoloNetwork::HeadStyle::FPN)
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

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(8);
        adam->set_maximum_epochs(head_style == YoloNetwork::HeadStyle::FPN ? 1 : 5);
        adam->set_display_period(1);
        // Tighter global gradient clip than the 1.0 default — Adam is
        // scale-invariant so this matters less for the step magnitude, but it
        // still bounds the second-moment estimate's growth rate, which helps
        // keep the t_w/t_h logits from saturating exp() under GIoU.
        adam->set_gradient_clip_norm(0.1f);

        // Skip training if weights already exist on disk — lets you iterate on
        // visualization / inference code without retraining. Delete the file to
        // force a fresh training run. The filename encodes the backbone so
        // switching architectures doesn't try to load incompatible weights.
        const std::string weights_filename = std::string("yolo_weights_") +
            (backbone == YoloNetwork::Backbone::Vgg ? "vgg" : "darknet") +
            (head_style == YoloNetwork::HeadStyle::FPN ? "_fpn" : "") +
            (body_activation == YoloNetwork::BodyActivation::LeakyReLU ? "_leaky" : "") +
            std::string(".bin");
        std::filesystem::path weights_path = data_dir / weights_filename;
        // Backward-compat: load the Phase 2 committed weights file if present.
        const std::filesystem::path legacy_weights = data_dir / "yolo_weights.bin";
        if (backbone == YoloNetwork::Backbone::Vgg
        &&  !std::filesystem::exists(weights_path)
        &&   std::filesystem::exists(legacy_weights))
            weights_path = legacy_weights;
        if (std::filesystem::exists(weights_path))
        {
            yolo_network.load_parameters_binary(weights_path);
            std::cout << "\nLoaded weights from " << weights_path
                      << " (skipping training). Delete this file to retrain.\n";
        }
        else
        {
            std::cout << "\nStarting training...\n";
            training_strategy.train();
            std::cout << "Training returned (no crash).\n";
            yolo_network.save_parameters_binary(weights_path);
            std::cout << "Saved trained weights to " << weights_path
                      << " (" << yolo_network.get_parameters_number() << " parameters)\n";
        }

        // FPN mode: gather per-head Detection layer outputs from a manual
        // forward pass (no NMS layer is appended in FPN networks), then run
        // cross-scale NMS in decode_yolo_fpn_detections. Single-head mode
        // continues to read from the appended NMS layer below.
        const bool is_fpn = (head_style == YoloNetwork::HeadStyle::FPN);

        // Inference on 5 SELECTION (held-out) samples — the model has never
        // seen these during training, so this is the real generalization test.
        // Each image is saved as an annotated BMP showing the input, the
        // ground-truth box (green), and the top-1 predicted box (red).

        const std::filesystem::path output_dir = data_dir / "annotated";
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

        const std::vector<Index> selection_indices = dataset.get_sample_indices("Validation");
        const Index samples_to_visualize = std::min(Index(5), Index(selection_indices.size()));
        std::cout << "\nVisualizing " << samples_to_visualize
                  << " held-out (validation) samples — model has never seen these. "
                  << "Annotated BMPs in " << output_dir << ":\n";

        for (Index k = 0; k < samples_to_visualize; ++k)
        {
            const Index s = selection_indices[size_t(k)];

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
                const auto& layers = yolo_network.get_layers();
                for (size_t li = 0; li < layers.size(); ++li)
                {
                    if (!layers[li] || layers[li]->get_type() != LayerType::Detection) continue;
                    const Shape head_shape = layers[li]->get_output_shape();
                    // Shape per Detection layer: [grid, grid, 3*(5+classes)] — drop batch.
                    const TensorView view = forward_propagation.views[li].back()[0];
                    const Index channels = head_shape[2];
                    const Index classes_n = Index(dataset.get_classes_number());
                    const Index boxes_per_head = channels / (5 + classes_n);
                    fpn_heads.push_back({ view.as<float>(), head_shape[0], boxes_per_head, classes_n });
                }

                detections = decode_yolo_fpn_detections(
                    fpn_heads,
                    /*original_height=*/input_shape[0],
                    /*original_width=*/input_shape[1],
                    /*network_height=*/input_shape[0],
                    /*network_width=*/input_shape[1]);
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

            std::ifstream label_in(label_path);
            int gt_class = 0;
            float gt_cx = 0, gt_cy = 0, gt_w = 0, gt_h = 0;
            label_in >> gt_class >> gt_cx >> gt_cy >> gt_w >> gt_h;

            const int W = int(input_shape[1]);
            const int H = int(input_shape[0]);
            const int gt_x0 = int(std::round((gt_cx - gt_w * 0.5f) * float(W)));
            const int gt_y0 = int(std::round((gt_cy - gt_h * 0.5f) * float(H)));
            const int gt_x1 = int(std::round((gt_cx + gt_w * 0.5f) * float(W))) - 1;
            const int gt_y1 = int(std::round((gt_cy + gt_h * 0.5f) * float(H))) - 1;

            Image24 img = read_bmp24(image_path);
            draw_rect_outline(img, gt_x0, gt_y0, gt_x1, gt_y1, box_color_by_role[0], 2);

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

            const std::filesystem::path out_path =
                output_dir / ("annotated_" + name + ".bmp");
            write_bmp24_top_down(out_path, img);
            std::cout << "  -> " << out_path << "\n";
        }

        std::cout << "\nLegend: green = GT, red = top-1, orange = top-2, "
                  << "yellow = top-3, cyan = best-IoU-vs-GT (if outside top-3).\n";
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
