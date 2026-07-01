#include "pch.h"

#include "../opennn/yolo_dataset.h"
#include "../opennn/detection_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/loss.h"
#include "../opennn/adaptive_moment_estimation.h"

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
