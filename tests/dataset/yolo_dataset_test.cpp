// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "tests/common/pch.h"
#include "opennn/core/json.h"

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/dataset/yolo_dataset_internal.h"
#include "opennn/core/device_backend.h"

#include "tests/common/test_helpers.h"
#include "tests/common/augmentation_helpers.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <system_error>

using namespace opennn;
using namespace opennn_test;

namespace
{

struct YoloAugmentationFixture
{
    TempDir directory{"opennn_yolo_augmentation_partition_"};
    YoloDataset dataset;

    explicit YoloAugmentationFixture(Dataset::StorageMode storage, int width = 16, int height = 16,
                                     int boxes_per_image = 1)
    {
        const auto images = directory.path / "images";
        const auto labels = directory.path / "labels";
        filesystem::create_directories(images);
        filesystem::create_directories(labels);
        write_classes(labels / "classes.names", {"zero", "one", "two", "three", "four"});
        for (int i = 0; i < 5; ++i)
        {
            const string name = to_string(i);
            write_bmp_24(images / (name + ".bmp"), width, height, uint8_t(40 + 40 * i), 30, 10);
            ofstream label(labels / (name + ".txt"));
            for (int box = 0; box < boxes_per_image; ++box)
                label << i << " 0.5 0.5 1 1\n";
        }
        dataset.set_display(false);
        dataset.set(images, labels, {height, width, 3}, 2, 1, {{0.5f, 0.5f}});
        dataset.set_storage_mode(storage);
        dataset.set_sample_roles({"Validation", "Testing", "Training", "None", "Validation"});
        YoloDataset::AugmentationPolicy policy;
        policy.enabled = true;
        policy.jitter = 0.0f;
        policy.exposure = policy.saturation = 1.0f;
        policy.hue = 0.0f;
        policy.flip = false;
        policy.mosaic = true;
        dataset.set_augmentation_policy(policy);
    }
};

// Training reads the first augmented copy of sample 2; other modes read sample 2.
pair<vector<float>, vector<float>> read_augmented_sample(
    const YoloDataset& dataset, FillMode mode = FillMode::Training)
{
    const Index index = mode == FillMode::Training ? 2 + dataset.get_samples_number() : 2;
    pair<vector<float>, vector<float>> result{
        vector<float>(size_t(dataset.get_input_shape().size())),
        vector<float>(size_t(dataset.get_target_shape().size()))};
    dataset.fill_inputs({index}, {}, result.first.data(), mode);
    dataset.fill_targets({index}, {}, result.second.data(), mode);
    return result;
}

void expect_mosaic_pixels_match_boxes(const pair<vector<float>, vector<float>>& batch, int width, int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const float* pixel = batch.first.data() + (y * width + x) * 3;
            bool near_box = false;
            for (size_t i = 0; i < batch.second.size() && batch.second[i + 4] > 0.0f; i += 5)
            {
                const float* box = batch.second.data() + i;
                const float left = (box[0] - box[2] / 2) * width;
                const float right = (box[0] + box[2] / 2) * width;
                const float top = (box[1] - box[3] / 2) * height;
                const float bottom = (box[1] + box[3] / 2) * height;
                constexpr float margin = 6.0f;
                near_box |= x + 0.5f > left - margin && x + 0.5f < right + margin
                         && y + 0.5f > top - margin && y + 0.5f < bottom + margin;
                if (x + 0.5f <= left + margin || x + 0.5f >= right - margin
                    || y + 0.5f <= top + margin || y + 0.5f >= bottom - margin) continue;
                EXPECT_NEAR(pixel[0], (40 + 40 * (box[4] - 1)) / 255.0f, 1e-6f);
                EXPECT_NEAR(pixel[1], 30.0f / 255, 1e-6f);
                EXPECT_NEAR(pixel[2], 10.0f / 255, 1e-6f);
            }
            if (!near_box)
                for (int c = 0; c < 3; ++c) EXPECT_NEAR(pixel[c], 128.0f / 255, 1e-6f);
        }
}

void expect_mosaic_sources(YoloDataset& dataset, Index primary, Index only_source = -1)
{
    const Index copy = primary + dataset.get_samples_number();
    vector<float> inputs(16 * 16 * 3), targets(size_t(dataset.get_target_shape().size()));
    dataset.fill_inputs({copy}, {}, inputs.data(), FillMode::Training);
    dataset.fill_targets({copy}, {}, targets.data(), FillMode::Training);
    for (size_t box = 0; box < 4; ++box)
    {
        const float* target = targets.data() + box * 5;
        const Index source = Index(target[4]) - 1;
        ASSERT_GE(source, 0);
        ASSERT_LT(source, 5);
        if (only_source >= 0) EXPECT_EQ(source, only_source);
        else EXPECT_TRUE(source == 1 || source == 3);
        const Index x = Index(target[0] * 16), y = Index(target[1] * 16);
        ASSERT_GE(x, 0);
        ASSERT_LT(x, 16);
        ASSERT_GE(y, 0);
        ASSERT_LT(y, 16);
        EXPECT_NEAR(inputs[size_t((y * 16 + x) * 3)], float(40 + 40 * source) / 255, 1e-6f);
    }
    if (only_source >= 0)
        for (size_t pixel = 0; pixel < inputs.size(); pixel += 3)
            EXPECT_NEAR(inputs[pixel], float(40 + 40 * only_source) / 255, 1e-6f);
}

}

TEST(YoloDataset, MosaicUsesOnlyCurrentTrainingImagesAndBoxes)
{
    for (auto storage : {Dataset::StorageMode::BinaryFile, Dataset::StorageMode::Matrix})
    {
        YoloAugmentationFixture fixture(storage);
        auto& dataset = fixture.dataset;
        dataset.set_v8_mode(true);
        auto first = async(launch::async, [&] { return read_augmented_sample(dataset); });
        auto second = async(launch::async, [&] { return read_augmented_sample(dataset); });
        EXPECT_EQ(first.get(), second.get());
        expect_mosaic_sources(dataset, 2, 2);
        dataset.set_sample_roles(SampleRole::None);
        dataset.set_sample_role(4, SampleRole::Training);
        expect_mosaic_sources(dataset, 4, 4);
        dataset.set_sample_roles(SampleRole::None);
        dataset.set_sample_roles(vector<Index>{1, 3}, SampleRole::Training);
        expect_mosaic_sources(dataset, 1);
        expect_mosaic_sources(dataset, 3);
        {
            FoldScope fold(dataset, {3}, {1});
            expect_mosaic_sources(dataset, 3, 3);
        }
        expect_mosaic_sources(dataset, 1);
        Dataset& base = dataset;
        base.split_samples_sequential(0.5f, 0.5f, 0.0f);
        expect_mosaic_sources(dataset, 1, 1);
        base.set_sample_roles({"Validation", "Validation", "Training", "Validation", "Validation"});
        expect_mosaic_sources(dataset, 2, 2);
        JsonDocument saved;
        saved.set_root(Json::parse(augmentation_json(dataset)));
        base.set_sample_role(2, SampleRole::Validation);
        base.set_sample_role(4, SampleRole::Training);
        expect_mosaic_sources(dataset, 4, 4);
        dataset.from_JSON(saved);
        expect_mosaic_sources(dataset, 2, 2);
    }
}

TEST(YoloDataset, V8PreservesAllBoxesWithAndWithoutMosaic)
{
    for (auto storage : {Dataset::StorageMode::BinaryFile, Dataset::StorageMode::Matrix})
        for (int count : {31, 101})
        {
            YoloAugmentationFixture fixture(storage, 16, 16, count);
            auto& dataset = fixture.dataset;
            dataset.set_v8_mode(true);
            auto policy = dataset.get_augmentation_policy();
            for (bool mosaic : {false, true})
                for (bool enabled : {true, false})
                {
                    policy.mosaic = mosaic;
                    policy.enabled = enabled;
                    dataset.set_augmentation_policy(policy);
                    const Index capacity = max<Index>(100, count * (enabled && mosaic ? 4 : 1));
                    ASSERT_EQ(dataset.get_target_shape().size(), capacity * 5);
                    for (FillMode mode : {FillMode::Training, FillMode::Validation})
                    {
                        const bool augmented = enabled && mosaic && mode == FillMode::Training;
                        const Index index = augmented ? 2 + dataset.get_samples_number() : 2;
                        vector<float> targets(size_t(capacity * 10 + 1), -99.0f);
                        dataset.fill_targets({index, index}, {}, targets.data(), mode);
                        const Index expected = count * (augmented ? 4 : 1);
                        for (Index row = 0; row < 2; ++row)
                            for (Index box = 0; box < capacity; ++box)
                                EXPECT_EQ(targets[size_t((row * capacity + box) * 5 + 4)],
                                          box < expected ? 3.0f : 0.0f);
                        EXPECT_EQ(targets.back(), -99.0f);
                    }
                }
        }
}

TEST(YoloDataset, AugmentationRejectsNonTrainingInputsAndTargets)
{
    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile);
    auto& dataset = fixture.dataset;
    auto policy = dataset.get_augmentation_policy();
    for (bool mosaic : {false, true})
    {
        policy.mosaic = mosaic;
        dataset.set_augmentation_policy(policy);
        vector<float> input(2 * 16 * 16 * 3, -99.0f);
        vector<float> target(size_t(2 * dataset.get_target_shape().size()), -99.0f);
        for (Index excluded : {0, 1, 3, 4, -1, 5})
        {
            EXPECT_THROW(dataset.fill_inputs({2, excluded}, {}, input.data(), FillMode::Training), runtime_error);
            EXPECT_THROW(dataset.fill_targets({2, excluded}, {}, target.data(), FillMode::Training), runtime_error);
            EXPECT_TRUE(ranges::all_of(input, [](float x) { return x == -99.0f; }));
            EXPECT_TRUE(ranges::all_of(target, [](float x) { return x == -99.0f; }));
        }
    }
    dataset.set_sample_roles(SampleRole::Validation);
    EXPECT_THROW(read_augmented_sample(dataset), runtime_error);
    for (FillMode mode : {FillMode::Validation, FillMode::Inference})
    {
        const auto expected = read_augmented_sample(dataset, mode);
        policy.enabled = false;
        dataset.set_augmentation_policy(policy);
        EXPECT_EQ(read_augmented_sample(dataset, mode), expected);
        policy.enabled = true;
        dataset.set_augmentation_policy(policy);
    }
}

TEST(YoloDataset, AugmentationReplaysImagesAndBoxesAcrossWorkersBatchesAndEpochs)
{
    for (bool mosaic : {false, true})
    {
        YoloAugmentationFixture fixture(mosaic ? Dataset::StorageMode::Matrix : Dataset::StorageMode::BinaryFile);
        auto& dataset = fixture.dataset;
        dataset.set_sample_roles(SampleRole::Training);
        dataset.set_v8_mode(mosaic);
        YoloDataset::AugmentationPolicy policy;
        policy.enabled = true;
        policy.mosaic = mosaic;
        policy.cutout_count = 2;
        policy.cutout_max_size_ratio = 0.2f;
        dataset.set_augmentation_policy(policy);
        expect_augmentation_replay(dataset, true);
    }
}

TEST(YoloDataset, CudaAugmentationMatchesHostImagesAndBoxes)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    for (auto storage : {Dataset::StorageMode::BinaryFile, Dataset::StorageMode::Matrix})
        for (bool mosaic : {false, true})
        {
            YoloAugmentationFixture fixture(storage, 17, 13, 31);
            for (const auto& entry : filesystem::directory_iterator(fixture.directory.path / "images"))
                if (entry.path().extension() == ".bmp") add_image_pattern(entry.path());
            auto& dataset = fixture.dataset;
            dataset.set(fixture.directory.path / "images", fixture.directory.path / "labels", {13, 17, 3}, 2, 1, {{0.5f, 0.5f}});
            dataset.set_storage_mode(storage);
            dataset.set_sample_roles({"Validation", "Training", "Training", "Training", "Testing"});
            dataset.set_v8_mode(mosaic);
            YoloDataset::AugmentationPolicy policy;
            policy.enabled = true;
            policy.mosaic = mosaic;
            policy.cutout_count = 3;
            policy.cutout_max_size_ratio = 0.3f;
            dataset.set_augmentation_policy(policy);
            expect_cuda_augmentation(dataset, 2.0f / 255.0f);
            dataset.set_shape(VariableRole::Input, {9, 11, 3});
            expect_cuda_augmentation(dataset, 2.0f / 255.0f);
        }
}

TEST(YoloDataset, AugmentationJsonRoundTripsAndRejectsInvalidPolicies)
{
    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile);
    auto& source = fixture.dataset;
    auto policy = source.get_augmentation_policy();
    policy.cutout_count = 3;
    policy.cutout_max_size_ratio = 0.75f;
    policy.factor = 4;
    source.set_augmentation_policy(policy);
    JsonDocument document;
    document.set_root(Json::parse(augmentation_json(source)));
    YoloDataset restored;
    restored.set_display(false);
    restored.from_JSON(document);
    EXPECT_EQ(restored.get_augmentation_policy().cutout_count, 3);
    EXPECT_EQ(restored.get_augmentation_policy().cutout_max_size_ratio, 0.75f);
    EXPECT_EQ(restored.get_augmentation_policy().factor, 4);
    EXPECT_EQ(restored.get_augmentation_factor(), 4);
    EXPECT_EQ(read_augmented_sample(restored), read_augmented_sample(source));
    auto& json_source = document.get_root()["YoloDataset"]["DataSource"];
    erase_if(json_source.as_object(), [](const auto& field)
    {
        return field.first == "AugCutoutCount" || field.first == "AugCutoutMaxSizeRatio"
            || field.first == "AugFactor" || field.first == "AugEnabled";
    });
    restored.from_JSON(document);
    EXPECT_EQ(restored.get_augmentation_policy().cutout_count, 0);
    EXPECT_EQ(restored.get_augmentation_policy().cutout_max_size_ratio, 0.04f);
    EXPECT_EQ(restored.get_augmentation_policy().factor, 1);
    EXPECT_FALSE(restored.get_augmentation_policy().enabled);
    const Json valid = json_source;
    for (const auto& [field, value] : vector<pair<string, double>>{
        {"AugCutoutCount", -1}, {"AugCutoutCount", 1.5}, {"AugCutoutCount", 2147483648.0},
        {"AugCutoutMaxSizeRatio", 1.1}, {"AugJitter", 0.5}, {"AugExposure", 0},
        {"AugSaturation", -1}, {"AugHue", 0.6}, {"AugFactor", 0}, {"AugFactor", 11}})
    {
        json_source = valid;
        json_source.set(field, value);
        EXPECT_THROW(restored.from_JSON(document), runtime_error);
    }
}

TEST(YoloDataset, AugmentationPolicyValidationIsAtomic)
{
    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile);
    auto& dataset = fixture.dataset;
    const string original = augmentation_json(dataset);
    using Policy = YoloDataset::AugmentationPolicy;
    const auto reject = [&](const Policy& policy)
    {
        EXPECT_THROW(dataset.set_augmentation_policy(policy), runtime_error);
        EXPECT_EQ(augmentation_json(dataset), original);
    };
    for (bool enabled : {false, true})
    {
        Policy policy;
        policy.enabled = enabled;
        for (auto field : {&Policy::jitter, &Policy::exposure, &Policy::saturation,
                           &Policy::hue, &Policy::cutout_max_size_ratio})
            for (float value : {numeric_limits<float>::quiet_NaN(), numeric_limits<float>::infinity(), -0.1f})
            {
                auto invalid = policy;
                invalid.*field = value;
                reject(invalid);
            }
        for (auto [field, value] : vector<pair<float Policy::*, float>>{
            {&Policy::jitter, 0.5f}, {&Policy::hue, nextafter(0.5f, 1.0f)},
            {&Policy::cutout_max_size_ratio, 1.1f}, {&Policy::exposure, 0.0f},
            {&Policy::saturation, numeric_limits<float>::denorm_min()},
            {&Policy::exposure, numeric_limits<float>::max()}})
        {
            auto invalid = policy;
            invalid.*field = value;
            reject(invalid);
        }
        policy.cutout_count = -1;
        reject(policy);
    }
    dataset.set_v8_mode(true);
    auto policy = dataset.get_augmentation_policy();
    policy.jitter = nextafter(0.5f, 0.0f);
    policy.hue = 0.5f;
    policy.cutout_max_size_ratio = 1.0f;
    for (float scale : {0.25f, 1.0f, numeric_limits<float>::min(), 1.0e30f})
    {
        policy.exposure = policy.saturation = scale;
        dataset.set_augmentation_policy(policy);
        const auto [inputs, targets] = read_augmented_sample(dataset);
        EXPECT_TRUE(ranges::all_of(inputs, [](float x) { return isfinite(x) && x >= 0 && x <= 1; }));
        EXPECT_TRUE(ranges::all_of(targets, [](float x) { return isfinite(x); }));
    }
}

TEST(YoloDataset, AugmentedCopiesKeepTheOriginalAndDifferFromEachOther)
{
    EXPECT_FALSE(YoloDataset::AugmentationPolicy{}.enabled);

    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile);
    auto& dataset = fixture.dataset;
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_v8_mode(true);
    const Index samples = dataset.get_samples_number();

    // Flip alone yields one distinct copy; Mosaic redraws a layout for every copy.
    auto policy = dataset.get_augmentation_policy();
    policy.mosaic = false;
    policy.flip = true;
    policy.factor = 2;
    EXPECT_THROW(dataset.set_augmentation_policy(policy), runtime_error);
    policy.mosaic = true;
    EXPECT_NO_THROW(dataset.set_augmentation_policy(policy));

    policy.mosaic = false;
    policy.flip = false;
    policy.jitter = 0.3f;
    policy.factor = 3;
    dataset.set_augmentation_policy(policy);
    EXPECT_EQ(dataset.get_augmentation_factor(), 3);
    EXPECT_EQ(dataset.get_augmented_samples_number(), 4 * samples);

    const auto read_targets = [&](Index index, FillMode mode)
    {
        vector<float> targets(size_t(dataset.get_target_shape().size()));
        dataset.fill_targets({index}, {}, targets.data(), mode);
        return targets;
    };
    vector<vector<float>> copies;
    for (Index copy = 0; copy <= 3; ++copy)
        copies.push_back(read_targets(2 + copy * samples, FillMode::Training));
    EXPECT_EQ(copies[0], read_targets(2, FillMode::Validation));
    for (size_t a = 0; a < copies.size(); ++a)
        for (size_t b = a + 1; b < copies.size(); ++b)
            EXPECT_NE(copies[a], copies[b]);
    EXPECT_THROW(read_targets(2 + 4 * samples, FillMode::Training), runtime_error);
}

TEST(YoloDataset, CutoutIsBoundedOnRectangularAndSinglePixelImagesAndReachesBorders)
{
    set_seed(37);
    for (const auto& [width, height] : vector<pair<int, int>>{{1, 1}, {1, 32}, {32, 1}, {3, 31}, {31, 3}})
    {
        SCOPED_TRACE(to_string(width) + "x" + to_string(height));
        YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile, width, height);
        auto& dataset = fixture.dataset;
        auto policy = dataset.get_augmentation_policy();
        policy.mosaic = false;
        dataset.set_augmentation_policy(policy);
        // Without transformations or effective Cutout there is nothing to copy.
        EXPECT_EQ(dataset.get_augmentation_factor(), 0);
        const auto original = read_augmented_sample(dataset, FillMode::Validation);
        policy.cutout_count = 1;
        policy.cutout_max_size_ratio = 0.0f;
        dataset.set_augmentation_policy(policy);
        EXPECT_EQ(dataset.get_augmentation_factor(), 0);
        policy.cutout_max_size_ratio = 1.0f;
        dataset.set_augmentation_policy(policy);
        bool bottom = false, right = false;
        for (Index epoch = 0; epoch < 64; ++epoch)
        {
            dataset.set_augmentation_epoch(epoch);
            const auto actual = read_augmented_sample(dataset);
            EXPECT_EQ(actual.second, original.second);
            int zeros = 0, left = width, top = height, last_x = -1, last_y = -1;
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                {
                    const size_t offset = size_t((y * width + x) * 3);
                    const bool masked = actual.first[offset] == 0.0f;
                    for (size_t c = 0; c < 3; ++c)
                        EXPECT_EQ(actual.first[offset + c], masked ? 0.0f : original.first[offset + c]);
                    if (!masked) continue;
                    ++zeros;
                    left = min(left, x); top = min(top, y);
                    last_x = max(last_x, x); last_y = max(last_y, y);
                }
            EXPECT_GT(zeros, 0);
            EXPECT_EQ(zeros, (last_x - left + 1) * (last_y - top + 1));
            bottom |= last_y == height - 1;
            right |= last_x == width - 1;
        }
        EXPECT_TRUE(bottom);
        EXPECT_TRUE(right);
        for (FillMode mode : {FillMode::Validation, FillMode::Inference})
            EXPECT_EQ(read_augmented_sample(dataset, mode), original);
        policy.cutout_count = 0;
        dataset.set_augmentation_policy(policy);
        EXPECT_EQ(dataset.get_augmentation_factor(), 0);
    }
}

TEST(YoloDataset, MosaicFlipMirrorsTheWholeImageAndEveryBox)
{
    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile);
    auto& dataset = fixture.dataset;
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_v8_mode(true);
    auto policy = dataset.get_augmentation_policy();
    bool observed_flip = false;
    for (unsigned seed = 0; seed < 16; ++seed)
    {
        set_seed(seed);
        policy.flip = false;
        dataset.set_augmentation_policy(policy);
        const auto original = read_augmented_sample(dataset);
        auto mirrored = original;
        for (size_t y = 0; y < 16; ++y)
            for (size_t x = 0; x < 16; ++x)
                for (size_t c = 0; c < 3; ++c)
                    mirrored.first[(y * 16 + x) * 3 + c] = original.first[(y * 16 + 15 - x) * 3 + c];
        for (size_t box = 0; box < 4; ++box)
            mirrored.second[box * 5] = 1.0f - original.second[box * 5];
        policy.flip = true;
        dataset.set_augmentation_policy(policy);
        const auto actual = read_augmented_sample(dataset);
        if (actual == original) continue;
        EXPECT_EQ(actual, mirrored);
        observed_flip = true;
    }
    EXPECT_TRUE(observed_flip);
}

TEST(YoloDataset, MosaicJitterMovesClipsAndRemovesBoxesWithTheirImageContent)
{
    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile, 64, 48);
    auto& dataset = fixture.dataset;
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_v8_mode(true);
    auto policy = dataset.get_augmentation_policy();
    bool observed_removal = false;
    for (unsigned seed = 0; seed < 64; ++seed)
    {
        SCOPED_TRACE(seed);
        set_seed(seed);
        policy.jitter = 0.0f;
        dataset.set_augmentation_policy(policy);
        const auto original = read_augmented_sample(dataset);
        policy.jitter = 0.45f;
        dataset.set_augmentation_policy(policy);
        const auto actual = read_augmented_sample(dataset);
        EXPECT_NE(actual.first, original.first);
        EXPECT_NE(actual.second, original.second);
        observed_removal |= actual.second[19] == 0.0f;
        for (size_t i = 0; i < 20 && actual.second[i + 4] > 0.0f; i += 5)
        {
            const float* box = actual.second.data() + i;
            EXPECT_GT(box[2], 0.0f);
            EXPECT_GT(box[3], 0.0f);
            EXPECT_GE(box[0] - box[2] / 2, -1e-6f);
            EXPECT_LE(box[0] + box[2] / 2, 1 + 1e-6f);
            EXPECT_GE(box[1] - box[3] / 2, -1e-6f);
            EXPECT_LE(box[1] + box[3] / 2, 1 + 1e-6f);
        }
        expect_mosaic_pixels_match_boxes(actual, 64, 48);
    }
    EXPECT_TRUE(observed_removal);
}

TEST(YoloDataset, CutoutOccludesAfterMosaicGeometryWithoutChangingBoxesOrOtherPixels)
{
    YoloAugmentationFixture fixture(Dataset::StorageMode::BinaryFile);
    auto& dataset = fixture.dataset;
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_v8_mode(true);
    auto policy = dataset.get_augmentation_policy();
    policy.jitter = 0.2f;
    policy.flip = true;
    dataset.set_augmentation_policy(policy);
    const auto original = read_augmented_sample(dataset);
    policy.cutout_count = 3;
    policy.cutout_max_size_ratio = 0.5f;
    dataset.set_augmentation_policy(policy);
    const auto actual = read_augmented_sample(dataset);
    EXPECT_NE(actual.first, original.first);
    EXPECT_EQ(actual.second, original.second);
    for (size_t pixel = 0; pixel < actual.first.size(); pixel += 3)
        for (size_t c = 0; c < 3; ++c)
            EXPECT_EQ(actual.first[pixel + c], actual.first[pixel] == 0.0f ? 0.0f : original.first[pixel + c]);
}

TEST(YoloDataset, RejectsClassIdsOutsideClassesFile)
{
    TempDir dir("opennn_yolo_invalid_class_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "sample.bmp", 8, 8, 200, 100, 50);
    write_label(labels_dir / "sample.txt", 2, 0.5f, 0.5f, 0.4f, 0.4f);
    write_classes(labels_dir / "obj.names", {"cat", "dog"});

    YoloDataset dataset;
    dataset.set_display(false);

    try
    {
        dataset.set(images_dir, labels_dir, Shape{8, 8, 3}, 2, 1,
                    {{0.2f, 0.2f}});
        FAIL() << "Expected an out-of-range YOLO class id to be rejected.";
    }
    catch (const runtime_error& error)
    {
        const string message = error.what();
        EXPECT_NE(message.find("class id 2"), string::npos);
        EXPECT_NE(message.find("sample.txt"), string::npos);
        EXPECT_NE(message.find("defines 2 classes"), string::npos);
        EXPECT_NE(message.find("0..1"), string::npos);
    }
}

TEST(YoloDataset, RejectsNegativeClassIdsWithoutClassesFile)
{
    TempDir dir("opennn_yolo_negative_class_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "sample.bmp", 8, 8, 200, 100, 50);
    write_label(labels_dir / "sample.txt", -1, 0.5f, 0.5f, 0.4f, 0.4f);

    YoloDataset dataset;
    dataset.set_display(false);

    EXPECT_THROW(dataset.set(images_dir, labels_dir, Shape{8, 8, 3}, 2, 1,
                             {{0.2f, 0.2f}}), runtime_error);
}

TEST(YoloDataset, InfersClassesWhenClassesFileIsAbsent)
{
    TempDir dir("opennn_yolo_inferred_classes_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "sample.bmp", 8, 8, 200, 100, 50);
    write_label(labels_dir / "sample.txt", 2, 0.5f, 0.5f, 0.4f, 0.4f);

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{8, 8, 3}, 2, 1,
                {{0.2f, 0.2f}});

    EXPECT_EQ(dataset.get_classes_number(), 3);
}

TEST(YoloDataset, RejectsOutOfRangeClassIdsFromExistingCache)
{
    TempDir dir("opennn_yolo_cached_invalid_class_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    const filesystem::path label_path = labels_dir / "sample.txt";
    write_bmp_24(images_dir / "sample.bmp", 8, 8, 200, 100, 50);
    write_label(label_path, 1, 0.5f, 0.5f, 0.4f, 0.4f);
    write_classes(labels_dir / "obj.names", {"cat", "dog"});

    {
        YoloDataset dataset;
        dataset.set_display(false);
        dataset.set(images_dir, labels_dir, Shape{8, 8, 3}, 2, 1,
                    {{0.2f, 0.2f}});
    }

    const auto label_time = filesystem::last_write_time(label_path);
    write_label(label_path, 2, 0.5f, 0.5f, 0.4f, 0.4f);
    filesystem::last_write_time(label_path, label_time);

    const filesystem::path boxes_path = images_dir / ".cache" / "yolo_boxes.bin";
    fstream boxes_file(boxes_path, ios::binary | ios::in | ios::out);
    ASSERT_TRUE(boxes_file.good());
    yolo_detail::YoloBoxesCacheHeader header{};
    boxes_file.read(reinterpret_cast<char*>(&header), sizeof(header));
    ASSERT_TRUE(boxes_file.good());
    const int32_t invalid_class_id = 2;
    boxes_file.seekp(streamoff(header.boxes_byte_offset));
    boxes_file.write(reinterpret_cast<const char*>(&invalid_class_id), sizeof(invalid_class_id));
    boxes_file.close();

    YoloDataset dataset;
    dataset.set_display(false);
    EXPECT_THROW(dataset.set(images_dir, labels_dir, Shape{8, 8, 3}, 2, 1,
                             {{0.2f, 0.2f}}), runtime_error);
}

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
    EXPECT_FALSE(dataset.get_augmentation_policy().enabled);

    YoloDataset::AugmentationPolicy policy;
    policy.enabled = true;
    dataset.set_augmentation_policy(policy);

    Dataset& base_dataset = dataset;
    base_dataset.enable_device_residency();
    EXPECT_FALSE(dataset.is_device_resident());

    policy.enabled = false;
    dataset.set_augmentation_policy(policy);

    base_dataset.enable_device_residency();
    ASSERT_TRUE(dataset.is_device_resident());

    auto invalid = policy;
    invalid.hue = 0.6f;
    EXPECT_THROW(dataset.set_augmentation_policy(invalid), runtime_error);
    EXPECT_TRUE(dataset.is_device_resident());
    EXPECT_FALSE(dataset.get_augmentation_policy().enabled);

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
    dataset.set_sample_roles(SampleRole::Training);

    YoloDataset::AugmentationPolicy policy;
    policy.enabled = true;
    policy.jitter = 0.0f;
    policy.exposure = 1.0f;
    policy.saturation = 1.0f;
    policy.hue = 0.0f;
    policy.flip = false;
    policy.mosaic = true;
    dataset.set_augmentation_policy(policy);

    // The first augmented copy of every sample.
    const vector<Index> sample_indices = {4, 5, 6, 7};
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
    TempDir fixture("opennn_yolo_three_heads_");
    const auto images = fixture.path / "images";
    const auto labels = fixture.path / "labels";
    filesystem::create_directories(images);
    filesystem::create_directories(labels);
    write_bmp_24(images / "objects.bmp", 24, 24, 90, 170, 220);
    write_classes(labels / "classes.names", {"first", "second"});
    {
        ofstream annotations(labels / "objects.txt");
        annotations << "0 0.75 0.25 0.5 0.375\n"
                    << "1 0.375 0.625 0.25 0.1875\n"
                    << "0 0.8125 0.4375 0.0625 0.09375\n";
    }
    const vector<vector<std::array<float, 2>>> anchors{
        {{0.5f, 0.375f}, {0.9f, 0.9f}},
        {{0.4f, 0.1f}, {0.25f, 0.1875f}},
        {{0.0625f, 0.09375f}, {0.125f, 0.03125f}}};
    YoloDataset observations;
    observations.set_display(false);
    observations.set(images, labels, Shape{24, 24, 3}, 2, 2, anchors.front());
    observations.set_multi_scale_heads({2, 4, 8}, anchors);

    // Two classes and two anchors give 14 values per cell. The three occupied
    // slots are (head, row, column, anchor): (0,0,1,0), (1,2,1,1), (2,3,6,0).
    vector<float> expected(14 * (4 + 16 + 64), 0.0f);
    const std::array<pair<size_t, std::array<float, 7>>, 3> occupied{{
        {14,  {0.5f, 0.5f, 0.5f, 0.375f, 1, 1, 0}},
        {189, {0.5f, 0.5f, 0.25f, 0.1875f, 1, 0, 1}},
        {700, {0.5f, 0.5f, 0.0625f, 0.09375f, 1, 1, 0}}}};
    for (const auto& [offset, box] : occupied)
        ranges::copy(box, expected.begin() + offset);

    vector<float> actual(expected.size(), numeric_limits<float>::quiet_NaN());
    observations.fill_targets({0}, {}, actual.data(), FillMode::Inference);
    EXPECT_EQ(actual, expected);
}

TEST(YoloDataset, MosaicClipsInSourceCoordinatesAndPreservesAppendOrder)
{
    const vector<YoloDataset::Box> boxes{
        {2, 0.5f, 0.5f, 1.5f, 1.5f},
        {3, 0.0f, 0.25f, 0.5f, 0.5f},
        {4, 2.0f, 0.5f, 0.5f, 0.5f},
        {5, 0.5f, 0.5f, 0.001f, 0.5f}};
    vector<YoloDataset::Box> mapped{{1, 0.125f, 0.125f, 0.125f, 0.125f}};
    yolo_detail::append_mosaic_boxes(boxes, {0.5f, 0.25f}, {0.25f, 0.5f}, mapped);
    ASSERT_EQ(mapped.size(), 3);
    EXPECT_EQ(mapped[0].class_id, 1);
    EXPECT_FLOAT_EQ(mapped[0].x, 0.125f);
    EXPECT_EQ(mapped[1].class_id, 2);
    EXPECT_FLOAT_EQ(mapped[1].x, 0.625f);
    EXPECT_FLOAT_EQ(mapped[1].y, 0.5f);
    EXPECT_FLOAT_EQ(mapped[1].w, 0.25f);
    EXPECT_FLOAT_EQ(mapped[1].h, 0.5f);
    EXPECT_EQ(mapped[2].class_id, 3);
    EXPECT_FLOAT_EQ(mapped[2].x, 0.53125f);
    EXPECT_FLOAT_EQ(mapped[2].y, 0.375f);
    EXPECT_FLOAT_EQ(mapped[2].w, 0.0625f);
    EXPECT_FLOAT_EQ(mapped[2].h, 0.25f);
}
