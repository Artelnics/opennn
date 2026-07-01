#include "pch.h"

#include "../opennn/yolo_dataset.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

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
    file[19] = static_cast<uint8_t>((width >> 8) & 0xff);
    file[22] = static_cast<uint8_t>(height & 0xff);
    file[23] = static_cast<uint8_t>((height >> 8) & 0xff);
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
            std::filesystem::path candidate = base / ("opennn_yolo_test_" + std::to_string(i));
            std::error_code ec;
            if (std::filesystem::create_directories(candidate, ec) && !ec)
            {
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("Could not create temp dir for YoloDataset test");
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

TEST(YoloDataset, EncodesTargetsIntoExpectedGridCellAndAnchor)
{
    TempDir dir;
    const std::filesystem::path images_dir = dir.path / "images";
    const std::filesystem::path labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);

    constexpr int W = 16;
    constexpr int H = 16;

    write_bmp_24(images_dir / "a.bmp", W, H, 200, 100,  50);
    write_bmp_24(images_dir / "b.bmp", W, H,  50, 100, 200);

    // Sample 0: cx=0.5, cy=0.5, w=0.4, h=0.4, class=0.
    //   With grid_size=4: col = floor(0.5*4) = 2, row = floor(0.5*4) = 2.
    //   Anchor IoU on (w,h) only:
    //     vs (0.2,0.2): inter = 0.04, area = 0.16  -> IoU = 0.25
    //     vs (0.5,0.5): inter = 0.16, area = 0.25  -> IoU = 0.64  -> best_anchor = 1
    write_label(labels_dir / "a.txt", 0, 0.5f,  0.5f,  0.4f, 0.4f);

    // Sample 1: cx=0.125, cy=0.875, w=0.2, h=0.2, class=1.
    //   col = floor(0.125*4) = 0, row = floor(0.875*4) = 3.
    //   vs (0.2,0.2): IoU = 1.0  -> best_anchor = 0
    write_label(labels_dir / "b.txt", 1, 0.125f, 0.875f, 0.2f, 0.2f);

    write_classes(labels_dir / "classes.names", {"cat", "dog"});

    const std::vector<std::array<float, 2>> anchors{{0.2f, 0.2f}, {0.5f, 0.5f}};

    constexpr Index grid_size = 4;
    constexpr Index B = 2;
    constexpr Index C = 2;

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{H, W, 3}, grid_size, B, anchors);

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_classes_number(), C);
    EXPECT_EQ(dataset.get_boxes_per_cell(), B);
    EXPECT_EQ(dataset.get_grid_size(), grid_size);
    ASSERT_EQ(dataset.get_anchors().size(), 2u);
    EXPECT_FLOAT_EQ(dataset.get_anchors()[0][0], 0.2f);
    EXPECT_FLOAT_EQ(dataset.get_anchors()[0][1], 0.2f);
    EXPECT_FLOAT_EQ(dataset.get_anchors()[1][0], 0.5f);
    EXPECT_FLOAT_EQ(dataset.get_anchors()[1][1], 0.5f);

    const Index values_per_box = 5 + C;
    const Index channels = B * values_per_box;
    const Index target_floats_per_sample = grid_size * grid_size * channels;

    std::vector<float> targets(static_cast<size_t>(2 * target_floats_per_sample), 0.0f);
    dataset.fill_targets({0, 1}, {}, targets.data(), /*is_training=*/false);

    // ---- Sample 0 ----------------------------------------------------------
    {
        const float* t = targets.data() + 0 * target_floats_per_sample;
        const Index row = 2;
        const Index col = 2;
        const Index anchor = 1;
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        EXPECT_FLOAT_EQ(t[base + 0], 0.5f * grid_size - float(col));   // cell-relative x = 0.0
        EXPECT_FLOAT_EQ(t[base + 1], 0.5f * grid_size - float(row));   // cell-relative y = 0.0
        EXPECT_FLOAT_EQ(t[base + 2], 0.4f);                            // image-relative w
        EXPECT_FLOAT_EQ(t[base + 3], 0.4f);                            // image-relative h
        EXPECT_NEAR(t[base + 4], 0.64f, 1e-4f);                         // max(iou(0.4x0.4, 0.5x0.5)=0.64, 0.5)
        EXPECT_FLOAT_EQ(t[base + 5 + 0], 1.0f);                        // class 0 one-hot
        EXPECT_FLOAT_EQ(t[base + 5 + 1], 0.0f);

        // The other anchor in the same cell must be empty.
        const Index other_base = (row * grid_size + col) * channels + 0 * values_per_box;
        EXPECT_FLOAT_EQ(t[other_base + 4], 0.0f);
    }

    // ---- Sample 1 ----------------------------------------------------------
    {
        const float* t = targets.data() + 1 * target_floats_per_sample;
        const Index row = 3;
        const Index col = 0;
        const Index anchor = 0;
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        EXPECT_FLOAT_EQ(t[base + 0], 0.125f * grid_size - float(col)); // 0.5
        EXPECT_FLOAT_EQ(t[base + 1], 0.875f * grid_size - float(row)); // 0.5
        EXPECT_FLOAT_EQ(t[base + 2], 0.2f);
        EXPECT_FLOAT_EQ(t[base + 3], 0.2f);
        EXPECT_FLOAT_EQ(t[base + 4], 1.0f);
        EXPECT_FLOAT_EQ(t[base + 5 + 0], 0.0f);
        EXPECT_FLOAT_EQ(t[base + 5 + 1], 1.0f);
    }

    // ---- Background cells must be all zero in sample 0 ---------------------
    {
        const float* t = targets.data() + 0 * target_floats_per_sample;
        const Index row = 0;
        const Index col = 0;

        for (Index anchor = 0; anchor < B; ++anchor)
        {
            const Index base = (row * grid_size + col) * channels + anchor * values_per_box;
            for (Index k = 0; k < values_per_box; ++k)
                EXPECT_FLOAT_EQ(t[base + k], 0.0f);
        }
    }
}

TEST(YoloDataset, FillsInputsWithExpectedShapeAndPixelValues)
{
    TempDir dir;
    const std::filesystem::path images_dir = dir.path / "images";
    const std::filesystem::path labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);

    constexpr int W = 8;
    constexpr int H = 8;
    write_bmp_24(images_dir / "solid.bmp", W, H, 200, 100, 50);
    write_label(labels_dir / "solid.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_classes(labels_dir / "classes.names", {"only"});

    const std::vector<std::array<float, 2>> anchors{{0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{H, W, 3}, 2, 1, anchors);

    YoloDataset::AugmentationConfig no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation(no_aug);

    const Index pixels = H * W * 3;
    std::vector<float> inputs(static_cast<size_t>(pixels), -1.0f);

    // is_training=true scales to [0,1].
    dataset.fill_inputs({0}, {}, inputs.data(), /*is_training=*/true);

    // Every pixel should be the same solid color, divided by 255.
    constexpr float expected_r = 200.0f / 255.0f;
    constexpr float expected_g = 100.0f / 255.0f;
    constexpr float expected_b =  50.0f / 255.0f;

    for (Index i = 0; i < pixels; i += 3)
    {
        EXPECT_NEAR(inputs[static_cast<size_t>(i + 0)], expected_r, 1e-6f);
        EXPECT_NEAR(inputs[static_cast<size_t>(i + 1)], expected_g, 1e-6f);
        EXPECT_NEAR(inputs[static_cast<size_t>(i + 2)], expected_b, 1e-6f);
    }
}

TEST(YoloDataset, MultiScaleTargetsRouteBoxesToCorrectHead)
{
    // Two heads: head0 grid=2 large anchors, head1 grid=4 small anchors.
    // One large box (0.6x0.6) must land in head0 anchor 0; one small box
    // (0.1x0.1) must land in head1 anchor 0.  Verifies make_target_multi_scale
    // routing, buffer layout (head0 floats then head1 floats), and objectness.

    TempDir dir;
    const std::filesystem::path images_dir = dir.path / "images";
    const std::filesystem::path labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "classes.names", {"only"});

    constexpr int W = 16;
    constexpr int H = 16;
    write_bmp_24(images_dir / "a.bmp", W, H, 128, 128, 128);

    // Label: large box at (0.25, 0.25) size 0.6x0.6, small box at (0.875, 0.875) size 0.1x0.1
    {
        std::ofstream lf(labels_dir / "a.txt");
        lf << "0 0.25 0.25 0.6 0.6\n";
        lf << "0 0.875 0.875 0.1 0.1\n";
    }

    const std::vector<std::array<float, 2>> anchors_large{{0.6f, 0.6f}, {0.7f, 0.7f}};
    const std::vector<std::array<float, 2>> anchors_small{{0.1f, 0.1f}, {0.2f, 0.2f}};

    constexpr Index C = 1;
    constexpr Index B = 2;  // boxes per head
    constexpr Index values_per_box = 5 + C;
    constexpr Index head_channels = B * values_per_box;
    constexpr Index grid0 = 2;
    constexpr Index grid1 = 4;
    const Index head0_floats = grid0 * grid0 * head_channels;  // 2*2*12 = 48
    const Index head1_floats = grid1 * grid1 * head_channels;  // 4*4*12 = 192
    const Index total_floats = head0_floats + head1_floats;    // 240

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{H, W, 3},
                /*grid_size=*/grid0, /*boxes_per_cell=*/B, anchors_large);
    dataset.set_multi_scale_heads({grid0, grid1}, {anchors_large, anchors_small});

    std::vector<float> target(static_cast<size_t>(total_floats), -99.0f);
    dataset.fill_targets({0}, {}, target.data(), /*is_training=*/false);

    // ---- Head 0: large box at col=0,row=0, anchor=0 -------------------------
    // iou(0.6x0.6, 0.6x0.6) = 1.0 → objectness = max(1.0, 0.5) = 1.0
    {
        const Index col = 0, row = 0, anchor = 0;
        const Index base = (row * grid0 + col) * head_channels + anchor * values_per_box;
        EXPECT_NEAR(target[base + 0], 0.25f * grid0 - float(col), 1e-4f);  // 0.5
        EXPECT_NEAR(target[base + 1], 0.25f * grid0 - float(row), 1e-4f);  // 0.5
        EXPECT_NEAR(target[base + 2], 0.6f, 1e-4f);
        EXPECT_NEAR(target[base + 3], 0.6f, 1e-4f);
        EXPECT_NEAR(target[base + 4], 1.0f, 1e-4f);  // max(iou=1.0, 0.5)
        EXPECT_NEAR(target[base + 5], 1.0f, 1e-4f);  // class 0 one-hot
    }

    // ---- Head 1: small box at col=3,row=3, anchor=0 -------------------------
    // iou(0.1x0.1, 0.1x0.1) = 1.0 → objectness = 1.0
    {
        const Index col = 3, row = 3, anchor = 0;
        const Index base_in_head1 = (row * grid1 + col) * head_channels + anchor * values_per_box;
        const Index base = head0_floats + base_in_head1;
        EXPECT_NEAR(target[base + 0], 0.875f * grid1 - float(col), 1e-4f);  // 0.5
        EXPECT_NEAR(target[base + 1], 0.875f * grid1 - float(row), 1e-4f);  // 0.5
        EXPECT_NEAR(target[base + 2], 0.1f, 1e-4f);
        EXPECT_NEAR(target[base + 3], 0.1f, 1e-4f);
        EXPECT_NEAR(target[base + 4], 1.0f, 1e-4f);
        EXPECT_NEAR(target[base + 5], 1.0f, 1e-4f);
    }

    // ---- Head 0 must have no entry for the small box -------------------------
    {
        // small box would land at col=floor(0.875*2)=1, row=1 in grid0
        const Index col = 1, row = 1;
        for (Index anchor = 0; anchor < B; ++anchor)
        {
            const Index base = (row * grid0 + col) * head_channels + anchor * values_per_box;
            for (Index k = 0; k < values_per_box; ++k)
                EXPECT_FLOAT_EQ(target[base + k], 0.0f) << "head0 cell(1,1) should be empty";
        }
    }
}
