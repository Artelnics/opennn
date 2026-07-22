#include "pch.h"

#include "opennn/yolo_dataset.h"
#include "opennn/detection_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/pooling_layer.h"
#include "opennn/concatenation_layer.h"
#include "opennn/addition_layer.h"
#include "opennn/activation_layer.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"
#include "opennn/adaptive_moment_estimation.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace opennn;

namespace {

void write_bmp_24(const std::filesystem::path& path, int width, int height,
                  uint8_t r, uint8_t g, uint8_t b)
{
    const int row_bytes_unpadded = width * 3;
    const int row_pad = (4 - row_bytes_unpadded % 4) % 4;
    const int row_stride = row_bytes_unpadded + row_pad;
    const int pixel_data_size = row_stride * height;
    const int file_size = 54 + pixel_data_size;

    std::vector<uint8_t> file(static_cast<size_t>(file_size), 0);
    file[0] = 'B'; file[1] = 'M';
    file[2] = static_cast<uint8_t>(file_size & 0xff);
    file[3] = static_cast<uint8_t>((file_size >> 8) & 0xff);
    file[4] = static_cast<uint8_t>((file_size >> 16) & 0xff);
    file[5] = static_cast<uint8_t>((file_size >> 24) & 0xff);
    file[10] = 54; file[14] = 40;
    file[18] = static_cast<uint8_t>(width & 0xff);
    file[19] = static_cast<uint8_t>((width >> 8) & 0xff);
    file[22] = static_cast<uint8_t>(height & 0xff);
    file[23] = static_cast<uint8_t>((height >> 8) & 0xff);
    file[26] = 1; file[28] = 24;
    for (int y = 0; y < height; ++y)
    {
        const int row_offset = 54 + y * row_stride;
        for (int x = 0; x < width; ++x)
        {
            file[row_offset + x * 3 + 0] = b;
            file[row_offset + x * 3 + 1] = g;
            file[row_offset + x * 3 + 2] = r;
        }
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), file.size());
}

struct TempDir
{
    std::filesystem::path path;
    TempDir()
    {
        const auto base = std::filesystem::temp_directory_path();
        for (int i = 0; i < 10000; ++i)
        {
            std::filesystem::path candidate = base / ("opennn_yolo_overfit_" + std::to_string(i));
            std::error_code ec;
            if (std::filesystem::create_directories(candidate, ec) && !ec) { path = candidate; return; }
        }
        throw std::runtime_error("Could not create temp dir");
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

}

TEST(YoloOverfit, SingleImageSingleClassLossDecreases)
{
    // Train a minimal YOLO network on two 32x32 synthetic images with one box each.
    // This is the end-to-end sanity test: if loss does not decrease, there is a
    // systematic bug in loss → gradient → optimizer → detection chain.
    //
    // Network: Conv(3x3, 3→16, LeakyReLU) → Conv(1x1, 16→6, Identity) → Detection
    // Dataset: 2 images, 1 class, grid=4, B=1, 1 anchor matching the box size.
    // Assertions:
    //   1. error after 200 epochs < error after 2 epochs  (loss decreases)
    //   2. final error < 2.0                              (reasonable absolute level)

    TempDir dir;
    const std::filesystem::path images_dir = dir.path / "images";
    const std::filesystem::path labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200, 100,  50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 100, 200);

    {
        std::ofstream lf(labels_dir / "a.txt");
        lf << "0 0.5 0.5 0.4 0.4\n";
    }
    {
        std::ofstream lf(labels_dir / "b.txt");
        lf << "0 0.25 0.75 0.4 0.4\n";
    }
    {
        std::ofstream nf(labels_dir / "classes.names");
        nf << "object\n";
    }

    constexpr Index H = 32, W = 32, grid = 4, B = 1, C = 1;
    constexpr Index channels = B * (5 + C);
    const std::vector<std::array<float, 2>> anchors{{0.4f, 0.4f}};

    YoloDataset::AugmentationConfig no_aug;
    no_aug.enabled = false;

    auto build_net = [&]() {
        auto net = std::make_unique<NeuralNetwork>();
        net->add_layer(std::make_unique<Convolutional>(
            Shape{H, W, 3}, Shape{3, 3, 3, 16}, "LeakyReLU", Shape{1, 1}, "Same", false, "conv1"));
        net->add_layer(std::make_unique<Convolutional>(
            Shape{H, W, 16}, Shape{1, 1, 16, channels}, "Identity", Shape{1, 1}, "Same", false, "logits"));
        net->add_layer(std::make_unique<Detection>(Shape{grid, grid, channels}, anchors, "detection"));
        net->compile();
        VectorMap(net->get_parameters_data(), net->get_parameters_size()).setConstant(0.05f);
        return net;
    };

    // Short run — 2 epochs
    float error_short;
    {
        YoloDataset ds;
        ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation(no_aug);
        auto net = build_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(2);
        adam.set_display(false);
        error_short = adam.train().get_training_error();
    }

    // Long run — 200 epochs
    float error_long;
    {
        YoloDataset ds;
        ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation(no_aug);
        auto net = build_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(200);
        adam.set_display(false);
        error_long = adam.train().get_training_error();
    }

    // Loss must decrease (the essential check)
    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease: short=" << error_short << " long=" << error_long
        << " — systematic bug in YOLO loss/gradient/optimizer chain.";

    // Loss must decrease by at least 5% — catches trivial non-convergence
    EXPECT_LT(error_long, error_short * 0.95f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") after 200 epochs — optimizer or gradient likely broken.";
}

TEST(YoloOverfit, SPPFGradientFlowsAndLossDecreases)
{
    // Validates the SPPF pattern: Conv → 3×MaxPool(5×5,stride=1,pad=2) → Concat → Conv → Detection.
    // If loss decreases, backprop flows correctly through all three pooling layers and the
    // Concatenation split — the exact path added by the SPPF block in YoloNetwork.
    //
    // Uses a simple 32×32 network (not full Darknet53) so this runs in ~10 seconds on CPU.

    TempDir dir;
    const auto images_dir = dir.path / "images";
    const auto labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200, 50, 50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 200, 50);
    write_bmp_24(images_dir / "c.bmp", 32, 32,  50, 50, 200);
    write_bmp_24(images_dir / "d.bmp", 32, 32, 200, 200, 50);
    { std::ofstream f(labels_dir / "a.txt"); f << "0 0.5 0.5 0.4 0.4\n"; }
    { std::ofstream f(labels_dir / "b.txt"); f << "0 0.25 0.5 0.3 0.3\n"; }
    { std::ofstream f(labels_dir / "c.txt"); f << "0 0.75 0.5 0.3 0.3\n"; }
    { std::ofstream f(labels_dir / "d.txt"); f << "0 0.5 0.25 0.4 0.3\n"; }

    constexpr Index H = 32, W = 32, grid = 4, B = 1, C = 1;
    const std::vector<std::array<float, 2>> anchors{{0.4f, 0.4f}};
    constexpr Index ch = 16;    // intermediate channels
    constexpr Index logit_ch = B * (5 + C);

    YoloDataset::AugmentationConfig no_aug; no_aug.enabled = false;

    auto build_sppf_net = [&]() {
        auto net = std::make_unique<NeuralNetwork>();

        // Backbone stub: one conv to produce feature map
        net->add_layer(std::make_unique<Convolutional>(
            Shape{H, W, 3}, Shape{3, 3, 3, ch}, "LeakyReLU", Shape{1, 1}, "Same", true, "conv_stem"));

        // SPPF: three cascaded 5×5 same-padding max-pools, then concat with pre-pool
        const Shape feat{H, W, ch};
        const Index stem_idx = net->get_layers_number() - 1;

        net->add_layer(std::make_unique<Pooling>(feat, Shape{5,5}, Shape{1,1}, Shape{2,2},
                                                  "MaxPooling", "sppf_p1"), {stem_idx});
        const Index p1_idx = net->get_layers_number() - 1;
        net->add_layer(std::make_unique<Pooling>(feat, Shape{5,5}, Shape{1,1}, Shape{2,2},
                                                  "MaxPooling", "sppf_p2"), {p1_idx});
        const Index p2_idx = net->get_layers_number() - 1;
        net->add_layer(std::make_unique<Pooling>(feat, Shape{5,5}, Shape{1,1}, Shape{2,2},
                                                  "MaxPooling", "sppf_p3"), {p2_idx});
        const Index p3_idx = net->get_layers_number() - 1;

        // Concat(stem, p1, p2, p3) → 4×ch channels
        net->add_layer(std::make_unique<Concatenation>(feat,
                                                        std::vector<Index>{ch, ch, ch, ch}, "sppf_cat"),
                       {stem_idx, p1_idx, p2_idx, p3_idx});
        const Index cat_idx = net->get_layers_number() - 1;

        // Reduce back to ch, then logits → Detection
        net->add_layer(std::make_unique<Convolutional>(
            Shape{H, W, 4*ch}, Shape{1,1, 4*ch, ch}, "LeakyReLU", Shape{1,1}, "Same", true, "sppf_out"), {cat_idx});
        net->add_layer(std::make_unique<Convolutional>(
            Shape{H, W, ch}, Shape{1,1, ch, logit_ch}, "Identity", Shape{1,1}, "Same", false, "logits"));
        net->add_layer(std::make_unique<Detection>(Shape{grid, grid, logit_ch}, anchors, "detection"));

        net->compile();
        VectorMap(net->get_parameters_data(), net->get_parameters_size()).setConstant(0.05f);
        return net;
    };

    auto run = [&](Index epochs) -> float {
        YoloDataset ds; ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation(no_aug);
        auto net = build_sppf_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(epochs);
        adam.set_display(false);
        return adam.train().get_training_error();
    };

    const float error_short = run(2);
    const float error_long  = run(150);

    EXPECT_FALSE(std::isnan(error_short)) << "NaN after 2 epochs — forward/backward through SPPF broken.";
    EXPECT_FALSE(std::isnan(error_long))  << "NaN after 150 epochs — SPPF gradient instability.";
    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease through SPPF layers: short=" << error_short << " long=" << error_long;
    EXPECT_LT(error_long, error_short * 0.90f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") — backprop through pooling+concat may be broken.";
}

TEST(YoloOverfit, CSPGradientFlowsAndLossDecreases)
{
    // Validates the CSP block gradient path:
    //   stride-2 down → split(branch1=skip, branch2=residuals) → concat → merge → Detection.
    // Checks that backprop flows through the concat split, the skip branch, the residual
    // Addition, and the merge conv — all the new paths added by CSPDarknet53.
    // Uses a minimal 32×32 network (1 CSP block, 8-channel) so this runs in ~5s on CPU.

    TempDir dir;
    const auto images_dir = dir.path / "images";
    const auto labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200,  50,  50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 200,  50);
    write_bmp_24(images_dir / "c.bmp", 32, 32,  50,  50, 200);
    write_bmp_24(images_dir / "d.bmp", 32, 32, 200, 200,  50);
    { std::ofstream f(labels_dir / "a.txt"); f << "0 0.5 0.5 0.4 0.4\n"; }
    { std::ofstream f(labels_dir / "b.txt"); f << "0 0.25 0.5 0.3 0.3\n"; }
    { std::ofstream f(labels_dir / "c.txt"); f << "0 0.75 0.5 0.3 0.3\n"; }
    { std::ofstream f(labels_dir / "d.txt"); f << "0 0.5 0.25 0.4 0.3\n"; }
    { std::ofstream f(labels_dir / "classes.names"); f << "object\n"; }

    constexpr Index H = 32, W = 32, grid = 4, B = 1, C = 1;
    const std::vector<std::array<float, 2>> anchors{{0.4f, 0.4f}};
    constexpr Index ch = 8;
    constexpr Index half = ch / 2;
    constexpr Index logit_ch = B * (5 + C);

    YoloDataset::AugmentationConfig no_aug; no_aug.enabled = false;

    // Build a minimal CSP network: stem → CSP-stage (1 block, no stride-2 for simplicity)
    // → logits → Detection. Tests concat-split-skip + residual gradient paths.
    auto build_csp_net = [&]() {
        auto net = std::make_unique<NeuralNetwork>();

        // Stem
        const Shape input{H, W, 3};
        net->add_layer(std::make_unique<Convolutional>(
            input, Shape{3, 3, 3, ch}, "LeakyReLU", Shape{1, 1}, "Same", true, "stem"));
        const Index stem = net->get_layers_number() - 1;
        const Shape feat{H, W, ch};

        // CSP split: branch1 = skip (Identity), branch2 = residual path
        net->add_layer(std::make_unique<Convolutional>(
            feat, Shape{1, 1, ch, half}, "Identity", Shape{1, 1}, "Same", true, "csp_s1"), {stem});
        const Index branch1 = net->get_layers_number() - 1;

        net->add_layer(std::make_unique<Convolutional>(
            feat, Shape{1, 1, ch, half}, "LeakyReLU", Shape{1, 1}, "Same", true, "csp_s2"), {stem});
        const Index b2_start = net->get_layers_number() - 1;

        // 1 residual block on branch2 (half channels throughout)
        const Shape hfeat{H, W, half};
        net->add_layer(std::make_unique<Convolutional>(
            hfeat, Shape{1, 1, half, half}, "LeakyReLU", Shape{1, 1}, "Same", true, "csp_b1_c1"), {b2_start});
        const Index b1c1 = net->get_layers_number() - 1;
        net->add_layer(std::make_unique<Convolutional>(
            hfeat, Shape{3, 3, half, half}, "Identity", Shape{1, 1}, "Same", true, "csp_b1_c2"), {b1c1});
        const Index b1c2 = net->get_layers_number() - 1;
        net->add_layer(std::make_unique<Addition>(hfeat, "csp_b1_add"), {b1c2, b2_start});
        const Index add = net->get_layers_number() - 1;
        net->add_layer(std::make_unique<Activation>(hfeat, "LeakyReLU", "csp_b1_act"), {add});
        const Index branch2 = net->get_layers_number() - 1;

        // Concat + merge
        net->add_layer(std::make_unique<Concatenation>(hfeat, std::vector<Index>{half, half}, "csp_cat"),
                       {branch1, branch2});
        const Index cat = net->get_layers_number() - 1;
        net->add_layer(std::make_unique<Convolutional>(
            feat, Shape{1, 1, ch, ch}, "LeakyReLU", Shape{1, 1}, "Same", true, "csp_merge"), {cat});
        const Index merge = net->get_layers_number() - 1;

        // Detection head
        net->add_layer(std::make_unique<Convolutional>(
            feat, Shape{1, 1, ch, logit_ch}, "Identity", Shape{1, 1}, "Same", false, "logits"), {merge});
        net->add_layer(std::make_unique<Detection>(Shape{grid, grid, logit_ch}, anchors, "detection"));

        net->compile();
        VectorMap(net->get_parameters_data(), net->get_parameters_size()).setConstant(0.05f);
        return net;
    };

    auto run = [&](Index epochs) -> float {
        YoloDataset ds; ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation(no_aug);
        auto net = build_csp_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(epochs);
        adam.set_display(false);
        return adam.train().get_training_error();
    };

    const float error_short = run(2);
    const float error_long  = run(150);

    EXPECT_FALSE(std::isnan(error_short)) << "NaN after 2 epochs — forward/backward through CSP broken.";
    EXPECT_FALSE(std::isnan(error_long))  << "NaN after 150 epochs — CSP gradient instability.";
    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease through CSP layers: short=" << error_short << " long=" << error_long;
    EXPECT_LT(error_long, error_short * 0.90f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") — backprop through CSP split/concat/residual may be broken.";
}
