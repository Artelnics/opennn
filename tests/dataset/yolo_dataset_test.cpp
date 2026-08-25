#include "tests/pch.h"

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/core/device_backend.h"

#include "tests/test_helpers.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <system_error>

using namespace opennn;
using namespace opennn_test;

TEST(YoloDataset, AugmentationControlsDeviceResidency)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    TempDir dir("opennn_yolo_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "sample.bmp", 8, 8, 200, 100, 50);
    write_label(labels_dir / "sample.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_classes(labels_dir / "classes.names", {"only"});

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{8, 8, 3}, 2, 1,
                {{0.2f, 0.2f}});
    dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

    Dataset& base_dataset = dataset;
    base_dataset.enable_device_residency();
    EXPECT_FALSE(dataset.is_device_resident());

    YoloDataset::AugmentationPolicy policy;
    policy.enabled = false;
    dataset.set_augmentation_policy(policy);

    base_dataset.enable_device_residency();
    ASSERT_TRUE(dataset.is_device_resident());

    policy.enabled = true;
    dataset.set_augmentation_policy(policy);
    EXPECT_FALSE(dataset.is_device_resident());
}

TEST(YoloDataset, EncodesTargetsIntoExpectedGridCellAndAnchor)
{
    TempDir dir("opennn_yolo_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    constexpr int W = 16;
    constexpr int H = 16;

    write_bmp_24(images_dir / "a.bmp", W, H, 200, 100,  50);
    write_bmp_24(images_dir / "b.bmp", W, H,  50, 100, 200);

    write_label(labels_dir / "a.txt", 0, 0.5f,  0.5f,  0.4f, 0.4f);

    write_label(labels_dir / "b.txt", 1, 0.125f, 0.875f, 0.2f, 0.2f);

    write_classes(labels_dir / "classes.names", {"cat", "dog"});

    const vector<std::array<float, 2>> anchors{{0.2f, 0.2f}, {0.5f, 0.5f}};

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

    vector<float> targets(static_cast<size_t>(2 * target_floats_per_sample), 0.0f);
    dataset.fill_targets({0, 1}, {}, targets.data(), FillMode::Inference);

    {
        const float* t = targets.data() + 0 * target_floats_per_sample;
        const Index row = 2;
        const Index col = 2;
        const Index anchor = 1;
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        EXPECT_FLOAT_EQ(t[base + 0], 0.5f * grid_size - float(col));
        EXPECT_FLOAT_EQ(t[base + 1], 0.5f * grid_size - float(row));
        EXPECT_FLOAT_EQ(t[base + 2], 0.4f);
        EXPECT_FLOAT_EQ(t[base + 3], 0.4f);
        EXPECT_NEAR(t[base + 4], 0.64f, 1e-4f);
        EXPECT_FLOAT_EQ(t[base + 5 + 0], 1.0f);
        EXPECT_FLOAT_EQ(t[base + 5 + 1], 0.0f);

        const Index other_base = (row * grid_size + col) * channels + 0 * values_per_box;
        EXPECT_FLOAT_EQ(t[other_base + 4], 0.0f);
    }

    {
        const float* t = targets.data() + 1 * target_floats_per_sample;
        const Index row = 3;
        const Index col = 0;
        const Index anchor = 0;
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        EXPECT_FLOAT_EQ(t[base + 0], 0.125f * grid_size - float(col));
        EXPECT_FLOAT_EQ(t[base + 1], 0.875f * grid_size - float(row));
        EXPECT_FLOAT_EQ(t[base + 2], 0.2f);
        EXPECT_FLOAT_EQ(t[base + 3], 0.2f);
        EXPECT_FLOAT_EQ(t[base + 4], 1.0f);
        EXPECT_FLOAT_EQ(t[base + 5 + 0], 0.0f);
        EXPECT_FLOAT_EQ(t[base + 5 + 1], 1.0f);
    }

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
    TempDir dir("opennn_yolo_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    constexpr int W = 8;
    constexpr int H = 8;
    write_bmp_24(images_dir / "solid.bmp", W, H, 200, 100, 50);
    write_label(labels_dir / "solid.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_classes(labels_dir / "classes.names", {"only"});

    const vector<std::array<float, 2>> anchors{{0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{H, W, 3}, 2, 1, anchors);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    const Index pixels = H * W * 3;
    vector<float> inputs(static_cast<size_t>(pixels), -1.0f);

    dataset.fill_inputs({0}, {}, inputs.data(), FillMode::Training);

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

TEST(YoloDataset, MatrixStorageSupportsConcurrentReadsOfOneDataset)
{
    TempDir dir("opennn_yolo_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    constexpr Index width = 8;
    constexpr Index height = 8;
    write_bmp_24(images_dir / "a.bmp", int(width), int(height), 200, 100, 50);
    write_bmp_24(images_dir / "b.bmp", int(width), int(height), 50, 100, 200);
    write_label(labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_label(labels_dir / "b.txt", 0, 0.25f, 0.75f, 0.2f, 0.2f);
    write_classes(labels_dir / "classes.names", {"only"});

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{height, width, 3}, 2, 1,
                {{0.25f, 0.25f}});
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);

    YoloDataset::AugmentationPolicy no_augmentation;
    no_augmentation.enabled = false;
    dataset.set_augmentation_policy(no_augmentation);

    const auto read_batch = [&dataset]
    {
        vector<float> inputs(size_t(2 * height * width * 3));
        vector<float> targets(size_t(2 * 2 * 2 * (5 + 1)));
        for (Index repetition = 0; repetition < 16; ++repetition)
        {
            dataset.fill_inputs({0, 1}, {}, inputs.data(), FillMode::Inference);
            dataset.fill_targets({0, 1}, {}, targets.data(), FillMode::Inference);
        }
        return pair{std::move(inputs), std::move(targets)};
    };

    auto first = async(launch::async, read_batch);
    auto second = async(launch::async, read_batch);
    const auto first_batch = first.get();
    const auto second_batch = second.get();

    EXPECT_EQ(first_batch, second_batch);
}

TEST(YoloDataset, RepeatedMosaicBatchesKeepWorkerScratchIndependent)
{
    TempDir dir("opennn_yolo_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "classes.names", {"only"});

    constexpr Index width = 12;
    constexpr Index height = 10;
    constexpr Index samples = 4;
    for (Index sample = 0; sample < samples; ++sample)
    {
        const string name = "sample_" + to_string(sample);
        write_bmp_24(images_dir / (name + ".bmp"), int(width), int(height),
                     uint8_t(40 + sample * 30), uint8_t(80 + sample * 20),
                     uint8_t(120 + sample * 10));
        write_label(labels_dir / (name + ".txt"), 0,
                    0.25f + 0.1f * float(sample), 0.5f, 0.2f, 0.3f);
    }

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{height, width, 3}, 2, 1,
                {{0.25f, 0.25f}});

    YoloDataset::AugmentationPolicy policy;
    policy.jitter = 0.0f;
    policy.exposure = 1.0f;
    policy.saturation = 1.0f;
    policy.hue = 0.0f;
    policy.flip = false;
    policy.mosaic = true;
    dataset.set_augmentation_policy(policy);

    const vector<Index> sample_indices = {0, 1, 2, 3};
    constexpr Index input_values = samples * height * width * 3;
    constexpr Index target_values = samples * 2 * 2 * (5 + 1);

    for (Index repetition = 0; repetition < 8; ++repetition)
    {
        vector<float> inputs(static_cast<size_t>(input_values));
        vector<float> targets(static_cast<size_t>(target_values));
        dataset.fill_inputs(sample_indices, {}, inputs.data(), FillMode::Training);
        dataset.fill_targets(sample_indices, {}, targets.data(), FillMode::Training);

        EXPECT_TRUE(ranges::all_of(inputs, [](float value)
        {
            return isfinite(value) && value >= 0.0f && value <= 1.0f;
        }));
        EXPECT_TRUE(ranges::all_of(targets, [](float value)
        {
            return isfinite(value);
        }));
        EXPECT_GT(ranges::count_if(targets, [](float value) { return value != 0.0f; }), 0);
    }
}

TEST(YoloDataset, MultiScaleTargetsRouteBoxesToCorrectHead)
{

    TempDir dir("opennn_yolo_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "classes.names", {"only"});

    constexpr int W = 16;
    constexpr int H = 16;
    write_bmp_24(images_dir / "a.bmp", W, H, 128, 128, 128);

    {
        ofstream lf(labels_dir / "a.txt");
        lf << "0 0.25 0.25 0.6 0.6\n";
        lf << "0 0.875 0.875 0.1 0.1\n";
    }

    const vector<std::array<float, 2>> anchors_large{{0.6f, 0.6f}, {0.7f, 0.7f}};
    const vector<std::array<float, 2>> anchors_small{{0.1f, 0.1f}, {0.2f, 0.2f}};

    constexpr Index C = 1;
    constexpr Index B = 2;
    constexpr Index values_per_box = 5 + C;
    constexpr Index head_channels = B * values_per_box;
    constexpr Index grid0 = 2;
    constexpr Index grid1 = 4;
    const Index head0_floats = grid0 * grid0 * head_channels;
    const Index head1_floats = grid1 * grid1 * head_channels;
    const Index total_floats = head0_floats + head1_floats;

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{H, W, 3},
                              grid0,                    B, anchors_large);
    dataset.set_multi_scale_heads({grid0, grid1}, {anchors_large, anchors_small});

    vector<float> target(static_cast<size_t>(total_floats), -99.0f);
    dataset.fill_targets({0}, {}, target.data(), FillMode::Inference);

    {
        const Index col = 0, row = 0, anchor = 0;
        const Index base = (row * grid0 + col) * head_channels + anchor * values_per_box;
        EXPECT_NEAR(target[base + 0], 0.25f * grid0 - float(col), 1e-4f);
        EXPECT_NEAR(target[base + 1], 0.25f * grid0 - float(row), 1e-4f);
        EXPECT_NEAR(target[base + 2], 0.6f, 1e-4f);
        EXPECT_NEAR(target[base + 3], 0.6f, 1e-4f);
        EXPECT_NEAR(target[base + 4], 1.0f, 1e-4f);
        EXPECT_NEAR(target[base + 5], 1.0f, 1e-4f);
    }

    {
        const Index col = 3, row = 3, anchor = 0;
        const Index base_in_head1 = (row * grid1 + col) * head_channels + anchor * values_per_box;
        const Index base = head0_floats + base_in_head1;
        EXPECT_NEAR(target[base + 0], 0.875f * grid1 - float(col), 1e-4f);
        EXPECT_NEAR(target[base + 1], 0.875f * grid1 - float(row), 1e-4f);
        EXPECT_NEAR(target[base + 2], 0.1f, 1e-4f);
        EXPECT_NEAR(target[base + 3], 0.1f, 1e-4f);
        EXPECT_NEAR(target[base + 4], 1.0f, 1e-4f);
        EXPECT_NEAR(target[base + 5], 1.0f, 1e-4f);
    }

    {
        const Index col = 1, row = 1;
        for (Index anchor = 0; anchor < B; ++anchor)
        {
            const Index base = (row * grid0 + col) * head_channels + anchor * values_per_box;
            for (Index k = 0; k < values_per_box; ++k)
                EXPECT_FLOAT_EQ(target[base + k], 0.0f) << "head0 cell(1,1) should be empty";
        }
    }
}
