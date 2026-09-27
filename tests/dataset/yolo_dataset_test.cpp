#include "tests/pch.h"
#include "opennn/core/json.h"

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

    const auto expect_staged_samples = [&]
    {
        const Index inputs = dataset.get_features_number(VariableRole::Input);
        const Index targets = dataset.get_features_number(VariableRole::Target);
        vector<float> expected(size_t(inputs + targets));
        dataset.fill_inputs({0}, {}, expected.data(), FillMode::Training);
        dataset.fill_targets({0}, {}, expected.data() + inputs, FillMode::Training);
        ASSERT_EQ(dataset.get_device_data_columns(), inputs + targets);
        vector<float> staged(expected.size());
        const DeviceStream stream = device::get_compute_stream();
        device::copy_async(staged.data(), dataset.get_device_data(),
                           Index(staged.size() * sizeof(float)), device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
        EXPECT_EQ(staged, expected);
    };
    expect_staged_samples();

    dataset.set_multi_scale_heads({2, 4}, {{{0.4f, 0.4f}}, {{0.2f, 0.2f}}});
    EXPECT_FALSE(dataset.is_device_resident());
    base_dataset.enable_device_residency();
    ASSERT_TRUE(dataset.is_device_resident());
    expect_staged_samples();

    for (bool v8 : {true, false})
    {
        dataset.set_v8_mode(v8);
        EXPECT_FALSE(dataset.is_device_resident());
        base_dataset.enable_device_residency();
        ASSERT_TRUE(dataset.is_device_resident());
        expect_staged_samples();
    }

    policy.enabled = true;
    dataset.set_augmentation_policy(policy);
    EXPECT_FALSE(dataset.is_device_resident());
}

TEST(YoloDataset, SerializationRestoresTargetLayoutsWithoutExistingCaches)
{
    enum class Layout { Anchored, MultiHead, V8, V8WithHeads };
    for (Layout layout : {Layout::Anchored, Layout::MultiHead, Layout::V8, Layout::V8WithHeads})
    {
        SCOPED_TRACE(static_cast<int>(layout));
        TempDir dir("opennn_yolo_json_");
        const filesystem::path images_dir = dir.path / "images";
        const filesystem::path labels_dir = dir.path / "labels";
        filesystem::create_directories(images_dir);
        filesystem::create_directories(labels_dir);
        write_bmp_24(images_dir / "sample.bmp", 8, 8, 200, 100, 50);
        write_label(labels_dir / "sample.txt", 1, 0.25f, 0.75f, 0.4f, 0.6f);
        write_classes(labels_dir / "classes.names", {"cat", "dog"});

        const bool multi_head = layout == Layout::MultiHead || layout == Layout::V8WithHeads;
        const bool v8 = layout == Layout::V8 || layout == Layout::V8WithHeads;
        const vector<std::array<float, 2>> anchors = layout == Layout::V8
            ? vector<std::array<float, 2>>{} : vector<std::array<float, 2>>{{0.57f, 0.63f}, {0.73f, 0.77f}};
        const auto read_targets = [](const YoloDataset& dataset)
        {
            vector<float> targets(size_t(dataset.get_target_shape().size()));
            if (!targets.empty()) dataset.fill_targets({0}, {}, targets.data(), FillMode::Inference);
            return targets;
        };

        JsonDocument document;
        Shape expected_shape;
        Shape anchor_shape;
        vector<float> expected_targets;
        vector<float> anchor_targets;
        {
            YoloDataset source;
            source.set_display(false);
            source.set(images_dir, labels_dir, {8, 8, 3}, 2, ssize(anchors), anchors);
            if (multi_head)
                source.set_multi_scale_heads({2, 3}, {anchors, {{0.2f, 0.2f}, {0.4f, 0.6f}}});
            anchor_shape = source.get_target_shape();
            anchor_targets = read_targets(source);
            source.set_v8_mode(v8);
            source.set_storage_mode(Dataset::StorageMode::Matrix);
            source.set_sample_roles(SampleRole::Testing);
            source.set_display_confidence_threshold(0.375f);
            YoloDataset::AugmentationPolicy policy;
            policy.enabled = false;
            source.set_augmentation_policy(policy);
            expected_shape = source.get_target_shape();
            expected_targets = read_targets(source);
            ASSERT_EQ(expected_shape.size(), v8 ? 500 : (multi_head ? 182 : 56));
            if (v8)
            {
                const vector<float> first_box(expected_targets.begin(), expected_targets.begin() + 5);
                EXPECT_EQ(first_box, (vector<float>{0.25f, 0.75f, 0.4f, 0.6f, 2.0f}));
            }
            JsonWriter writer;
            source.to_JSON(writer);
            document.set_root(Json::parse(writer.c_str()));
        }

        // A fresh source cache must obtain custom anchors and head settings from JSON.
        for (const char* file : {"yolo_images.bin", "yolo_targets.bin", "yolo_boxes.bin"})
            ASSERT_TRUE(filesystem::remove(images_dir / ".cache" / file));

        YoloDataset restored;
        restored.set_display(false);
        restored.set_v8_mode(true);
        restored.from_JSON(document);
        EXPECT_EQ(restored.get_anchors(), anchors);
        EXPECT_EQ(restored.is_multi_scale(), multi_head);
        EXPECT_EQ(restored.get_target_shape(), expected_shape);
        EXPECT_EQ(restored.get_features_number(VariableRole::Target), expected_shape.size());
        EXPECT_EQ(restored.get_storage_mode(), Dataset::StorageMode::Matrix);
        EXPECT_EQ(restored.get_sample_roles(), vector<SampleRole>{SampleRole::Testing});
        EXPECT_FALSE(restored.get_augmentation_policy().enabled);
        EXPECT_FLOAT_EQ(restored.get_display_confidence_threshold(), 0.375f);
        EXPECT_EQ(read_targets(restored), expected_targets);

        Batch batch(1, &restored, EffectiveConfig{});
        batch.fill({0}, restored.get_feature_selection(), FillMode::Training);
        const float* targets = batch.get_targets().as_float();
        EXPECT_EQ(vector<float>(targets, targets + expected_targets.size()), expected_targets);
        EXPECT_FLOAT_EQ(batch.get_inputs()[0].as_float()[0], 200.0f / 255.0f);

        restored.set_v8_mode(false);
        EXPECT_EQ(restored.get_target_shape(), anchor_shape);
        EXPECT_EQ(restored.get_features_number(VariableRole::Target), anchor_shape.size());
        EXPECT_EQ(read_targets(restored), anchor_targets);
    }
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
    write_bmp_24(images_dir / "a.bmp", 16, 16, 128, 128, 128);

    // write_label holds a single box and truncates, so the two-box file is
    // written here. The scope closes the stream before the dataset reads it.
    {
        ofstream label_file(labels_dir / "a.txt");
        label_file << "0 0.25 0.25 0.6 0.6\n"
                   << "0 0.875 0.875 0.1 0.1\n";
    }

    // The large box matches the coarse set exactly, the small one the fine set.
    const vector<std::array<float, 2>> coarse_anchors{{0.6f, 0.6f}, {0.7f, 0.7f}};
    const vector<std::array<float, 2>> fine_anchors{{0.1f, 0.1f}, {0.2f, 0.2f}};

    constexpr Index classes_number = 1;
    constexpr Index anchors_per_head = 2;
    constexpr Index values_per_anchor = 5 + classes_number;
    constexpr Index cell_channels = anchors_per_head * values_per_anchor;

    // Heads are concatenated: the 2 x 2 coarse grid (48 floats), then the
    // 4 x 4 fine grid (192 floats).
    constexpr Index coarse_grid = 2;
    constexpr Index fine_grid = 4;
    constexpr Index coarse_start = 0;
    constexpr Index fine_start = coarse_start + coarse_grid * coarse_grid * cell_channels;
    constexpr Index target_floats = fine_start + fine_grid * fine_grid * cell_channels;

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{16, 16, 3}, coarse_grid, anchors_per_head, coarse_anchors);
    dataset.set_multi_scale_heads({coarse_grid, fine_grid}, {coarse_anchors, fine_anchors});

    // The sentinel exposes any float the encoder neither wrote nor cleared.
    vector<float> targets(size_t(target_floats), -99.0f);
    dataset.fill_targets({0}, {}, targets.data(), FillMode::Inference);

    const auto anchor_start = [&](Index head_start, Index grid, Index x, Index y, Index anchor)
    {
        return head_start + (y * grid + x) * cell_channels + anchor * values_per_anchor;
    };

    const auto expect_box = [&](Index start, const std::array<float, 6>& expected)
    {
        for (size_t k = 0; k < expected.size(); ++k)
            EXPECT_NEAR(targets[size_t(start) + k], expected[k], 1e-4f)
                << "component " << k << " of the box written at offset " << start;
    };

    // First box: coarse head, cell (0, 0), anchor 0, an exact anchor match.
    expect_box(anchor_start(coarse_start, coarse_grid, 0, 0, 0),
               {0.5f, 0.5f, 0.6f, 0.6f, 1.0f, 1.0f});

    // Second box: fine head, cell (3, 3), anchor 0, which is offset 228.
    expect_box(anchor_start(fine_start, fine_grid, 3, 3, 0),
               {0.5f, 0.5f, 0.1f, 0.1f, 1.0f, 1.0f});

    // The second box's centre lies in coarse cell (1, 1), but the box went to
    // the fine head, so both anchors of that cell must come out zero-filled.
    const Index empty_cell_start = anchor_start(coarse_start, coarse_grid, 1, 1, 0);
    for (Index k = 0; k < cell_channels; ++k)
        EXPECT_FLOAT_EQ(targets[size_t(empty_cell_start + k)], 0.0f)
            << "head 0 cell (1, 1) was expected to be empty, but offset "
            << empty_cell_start + k << " holds a value";
}
