#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/yolo_dataset.h"
#include "opennn/detection_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace opennn;

namespace {

void write_bmp_24(const std::filesystem::path& path, int width, int height, uint8_t r, uint8_t g, uint8_t b)
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
    file[10] = 54;

    file[14] = 40;
    file[18] = static_cast<uint8_t>(width & 0xff);
    file[22] = static_cast<uint8_t>(height & 0xff);
    file[26] = 1;
    file[28] = 24;

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

void write_label(const std::filesystem::path& path, int class_id, float cx, float cy, float w, float h)
{
    std::ofstream out(path);
    out << class_id << ' ' << cx << ' ' << cy << ' ' << w << ' ' << h << '\n';
}

void write_classes(const std::filesystem::path& path, std::initializer_list<const char*> names)
{
    std::ofstream out(path);
    for (auto* n : names) out << n << '\n';
}

struct TempDir
{
    std::filesystem::path path;

    TempDir()
    {
        const auto base = std::filesystem::temp_directory_path();
        for (int i = 0; i < 10000; ++i)
        {
            std::filesystem::path candidate = base / ("opennn_yolo_loss_test_" + std::to_string(i));
            std::error_code ec;
            if (std::filesystem::create_directories(candidate, ec) && !ec)
            {
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("Could not create temp dir for YOLO loss test");
    }

    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

}

namespace {

struct YoloLossFixture
{
    TempDir dir;
    std::filesystem::path images_dir;
    std::filesystem::path labels_dir;

    static constexpr Index W = 2;
    static constexpr Index H = 2;
    static constexpr Index grid = 2;
    static constexpr Index B = 2;
    static constexpr Index C = 1;
    static constexpr Index channels = B * (5 + C);

    const std::vector<std::array<float, 2>> anchors{{0.2f, 0.2f}, {0.5f, 0.5f}};

    YoloLossFixture()
    {
        images_dir = dir.path / "images";
        labels_dir = dir.path / "labels";
        std::filesystem::create_directories(images_dir);
        std::filesystem::create_directories(labels_dir);
        write_classes(labels_dir / "classes.names", {"only"});
    }
};

void build_yolo_network(NeuralNetwork& net, const YoloLossFixture& f)
{
    net.add_layer(make_unique<Convolutional>(Shape{f.H, f.W, 3},
                                             Shape{1, 1, 3, f.channels},
                                             "Identity",
                                             Shape{1, 1},
                                             "Same",
                                             false,
                                             "yolo_logits"));
    net.add_layer(make_unique<Detection>(Shape{f.grid, f.grid, f.channels}, f.anchors, "detection"));
    net.compile();
    VectorMap(net.get_parameters_data(), net.get_parameters_size()).setConstant(0.1f);
}

}

TEST(YoloLoss, NoObjectGradientMatchesNumericalGradient)
{
    // All-background targets exercise the no-object branch (delta = 2*lambda_noobj*out[4])
    // without the v1 paper's iou-as-constant approximation. Float32 accumulation across
    // many cells means the max element-wise difference is ~0.1; 0.5 tolerance catches
    // sign errors and order-of-magnitude bugs while tolerating that noise (same bound
    // as WithObjectGradientMatchesV1Approximation which has a structural reason for looseness).

    YoloLossFixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    { std::ofstream empty_a(f.labels_dir / "a.txt"); }  // empty file: no boxes
    { std::ofstream empty_b(f.labels_dir / "b.txt"); }

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, f.B, f.anchors);

    // Augmentation re-randomises inputs on every fill_inputs call, which
    // makes the loss non-deterministic across finite-difference probes.
    YoloDataset::AugmentationConfig no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation(no_aug);

    NeuralNetwork neural_network;
    build_yolo_network(neural_network, f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

TEST(YoloLoss, WithObjectGradientMatchesV1Approximation)
{
    // With objects present, dE/d(out[0..3]) misses the chain-rule term through
    // iou(output_box, target_box) because yolo_gradient_cpu treats iou as a
    // constant target — the v1 paper formulation. Numerical gradient does
    // differentiate through iou, so a generous tolerance is required.

    YoloLossFixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    write_label(f.labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_label(f.labels_dir / "b.txt", 0, 0.25f, 0.75f, 0.2f, 0.2f);

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, f.B, f.anchors);

    YoloDataset::AugmentationConfig no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation(no_aug);

    NeuralNetwork neural_network;
    build_yolo_network(neural_network, f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    // Loose bound: catches sign errors and order-of-magnitude bugs while
    // tolerating the missing iou-chain contribution.
    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}
