// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "tests/common/pch.h"

#include "opennn/dataset/image_dataset.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/json.h"
#include "opennn/core/variable.h"
#include "opennn/core/statistics.h"

#include "tests/common/test_helpers.h"
#include "tests/common/augmentation_helpers.h"

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <future>

using namespace opennn;
using namespace opennn_test;

namespace
{

struct ImageFixture
{
    filesystem::path root;

    ImageFixture(int width, int height, int images_per_class)
    {
        root = filesystem::temp_directory_path()
             / ("opennn_image_dataset_test_" + to_string(uint64_t(this) ^ uint64_t(width * 131 + height)));

        filesystem::remove_all(root);

        const filesystem::path class_a = root / "cats";
        const filesystem::path class_b = root / "dogs";

        filesystem::create_directories(class_a);
        filesystem::create_directories(class_b);

        for (int i = 0; i < images_per_class; ++i)
        {
            write_bmp_24(class_a / ("a_" + to_string(i) + ".bmp"), width, height, 200, 10, 10);
            write_bmp_24(class_b / ("b_" + to_string(i) + ".bmp"), width, height, 10, 10, 200);
        }
    }

    ~ImageFixture()
    {
        error_code ec;
        filesystem::remove_all(root, ec);
    }
};

struct ImageFixtureThreeClasses
{
    filesystem::path root;

    ImageFixtureThreeClasses(int width, int height, int images_per_class)
    {
        root = filesystem::temp_directory_path()
             / ("opennn_image_dataset_test3_" + to_string(uint64_t(this) ^ uint64_t(width * 17 + height)));

        filesystem::remove_all(root);

        const vector<string> class_names = { "red", "green", "blue" };

        for (size_t c = 0; c < class_names.size(); ++c)
        {
            const filesystem::path folder = root / class_names[c];
            filesystem::create_directories(folder);

            const uint8_t r = (c == 0) ? 255 : 0;
            const uint8_t g = (c == 1) ? 255 : 0;
            const uint8_t b = (c == 2) ? 255 : 0;

            for (int i = 0; i < images_per_class; ++i)
                write_bmp_24(folder / ("img_" + to_string(i) + ".bmp"), width, height, r, g, b);
        }
    }

    ~ImageFixtureThreeClasses()
    {
        error_code ec;
        filesystem::remove_all(root, ec);
    }
};

}

TEST(ImageDataset, DefaultConstructorIsEmpty)
{
    ImageDataset image_dataset;

    EXPECT_EQ(image_dataset.get_samples_number(), 0);
    EXPECT_EQ(image_dataset.get_variables_number(), 0);
    EXPECT_TRUE(image_dataset.is_empty());
}

TEST(ImageDataset, PackedCudaBatchesMatchHostScalingAndAugmentationFallback)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    ImageFixtureThreeClasses fixture(7, 5, 2);
    ImageDataset dataset(fixture.root);
    dataset.set_sample_roles(SampleRole::Training);
    const vector<Index> indices{5, 0, 3};
    const FeatureSelection features = dataset.get_feature_selection();
    for (Type dtype : {Type::FP32, Type::BF16})
    {
        if (dtype == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        Batch host(3, &dataset, {Device::CPU, Type::FP32, 0});
        Batch staged(3, &dataset, {Device::CUDA, dtype, 0}, true);
        Batch destination(3, &dataset, {Device::CUDA, dtype, 0});
        staged.pack_image_input = true;
        for (bool configured : {false, true})
        {
            dataset.clear_training_scaling();
            if (configured)
            {
                vector<Descriptives> descriptives(3);
                for (int i = 0; i < 3; ++i)
                {
                    descriptives[size_t(i)].minimum = float(i * 7);
                    descriptives[size_t(i)].maximum = float(255 - i * 11);
                }
                dataset.set_input_scaling(descriptives, vector<ScalerMethod>(3, ScalerMethod::MinimumMaximum),
                                          -0.5f, 1.5f);
            }
            for (bool augmentation : {false, true})
            {
                dataset.set_augmentation_policy({.enabled = augmentation});
                for (FillMode mode : {FillMode::Training, FillMode::Validation, FillMode::Inference})
                {
                    staged.wait_h2d_complete();
                    host.fill(indices, features, mode);
                    staged.fill(indices, features, mode);
                    EXPECT_EQ(staged.packed_image_input.has_value(), !augmentation && mode == FillMode::Training);
                    staged.upload_to_device_batch_async(destination, device::get_transfer_stream());
                    staged.wait_h2d_on_compute_stream();
                    vector<float> actual(size_t(host.input.shape.size()));
                    copy_device_to_host_float(destination.input.buffer.data(), destination.input.type,
                                              host.input.shape.size(), actual.data(), device::get_compute_stream());
                    device::synchronize(device::get_compute_stream());
                    for (size_t i = 0; i < actual.size(); ++i)
                    {
                        const float expected = dtype == Type::BF16
                            ? float(Eigen::bfloat16(host.input.buffer.as<float>()[i])) : host.input.buffer.as<float>()[i];
                        EXPECT_NEAR(actual[i], expected, 1.0e-6f);
                    }
                    EXPECT_THROW(staged.fill({99, 0, 1}, features, mode), runtime_error);
                }
            }
        }
    }
}

TEST(ImageDataset, ConstructFromPathTwoClasses)
{
    ImageFixture fixture(4, 3, 2);

    ImageDataset image_dataset(fixture.root);

    EXPECT_EQ(image_dataset.get_samples_number(), 4);

    const Shape input_shape = image_dataset.get_input_shape();
    ASSERT_EQ(input_shape.get_rank(), 3u);
    EXPECT_EQ(input_shape[0], 3);
    EXPECT_EQ(input_shape[1], 4);
    EXPECT_EQ(input_shape[2], 3);

    EXPECT_EQ(image_dataset.get_channels_number(), 3);

    const Shape target_shape = image_dataset.get_target_shape();
    ASSERT_EQ(target_shape.get_rank(), 1u);
    EXPECT_EQ(target_shape[0], 1);

    EXPECT_EQ(image_dataset.get_variables_number(), 2);
    EXPECT_EQ(image_dataset.get_features_number(), 3 * 4 * 3 + 1);
    EXPECT_EQ(image_dataset.get_features_number("Input"), 3 * 4 * 3);
}

TEST(ImageDataset, StorageModeIsBinaryFileWhenConstructed)
{
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    EXPECT_EQ(image_dataset.get_storage_mode(), Dataset::StorageMode::BinaryFile);
}

TEST(ImageDataset, TargetDistributionTwoClasses)
{
    ImageFixture fixture(2, 2, 3);

    ImageDataset image_dataset(fixture.root);

    const VectorI distribution = image_dataset.calculate_target_distribution();

    ASSERT_EQ(distribution.size(), 2);
    EXPECT_EQ(distribution(0), 3);
    EXPECT_EQ(distribution(1), 3);
}

TEST(ImageDataset, ThreeClassesUseCategoricalTargets)
{
    ImageFixtureThreeClasses fixture(2, 2, 2);

    ImageDataset image_dataset(fixture.root);

    EXPECT_EQ(image_dataset.get_samples_number(), 6);

    const Shape target_shape = image_dataset.get_target_shape();
    ASSERT_EQ(target_shape.get_rank(), 1u);
    EXPECT_EQ(target_shape[0], 3);

    const VectorI distribution = image_dataset.calculate_target_distribution();
    ASSERT_EQ(distribution.size(), 3);
    EXPECT_EQ(distribution(0), 2);
    EXPECT_EQ(distribution(1), 2);
    EXPECT_EQ(distribution(2), 2);
}

TEST(ImageDataset, FillTargetsBinary)
{
    ImageFixture fixture(2, 2, 2);

    ImageDataset image_dataset(fixture.root);

    const vector<Index> sample_indices = { 0, 1, 2, 3 };
    const vector<Index> target_indices = image_dataset.get_feature_indices("Target");

    ASSERT_EQ(ssize(target_indices), 1);

    vector<float> targets(sample_indices.size(), -1.0f);
    image_dataset.fill_targets(sample_indices, target_indices, targets.data(), FillMode::Inference,
                               ColumnContiguity::Unknown);

    EXPECT_FLOAT_EQ(targets[0], 0.0f);
    EXPECT_FLOAT_EQ(targets[1], 0.0f);
    EXPECT_FLOAT_EQ(targets[2], 1.0f);
    EXPECT_FLOAT_EQ(targets[3], 1.0f);
}

TEST(ImageDataset, FillTargetsOneHotThreeClasses)
{
    ImageFixtureThreeClasses fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    const vector<Index> sample_indices = { 0, 1, 2 };
    const vector<Index> target_indices = image_dataset.get_feature_indices("Target");

    ASSERT_EQ(ssize(target_indices), 3);

    vector<float> targets(sample_indices.size() * 3, -1.0f);
    image_dataset.fill_targets(sample_indices, target_indices, targets.data(), FillMode::Inference,
                               ColumnContiguity::Unknown);

    EXPECT_FLOAT_EQ(targets[0], 1.0f);
    EXPECT_FLOAT_EQ(targets[1], 0.0f);
    EXPECT_FLOAT_EQ(targets[2], 0.0f);

    EXPECT_FLOAT_EQ(targets[3], 0.0f);
    EXPECT_FLOAT_EQ(targets[4], 1.0f);
    EXPECT_FLOAT_EQ(targets[5], 0.0f);

    EXPECT_FLOAT_EQ(targets[6], 0.0f);
    EXPECT_FLOAT_EQ(targets[7], 0.0f);
    EXPECT_FLOAT_EQ(targets[8], 1.0f);
}

TEST(ImageDataset, FillInputsDefaultScalingFromCache)
{
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    const Index pixels = image_dataset.get_input_shape()[0]
                       * image_dataset.get_input_shape()[1]
                       * image_dataset.get_input_shape()[2];

    const vector<Index> input_indices = image_dataset.get_feature_indices("Input");
    const vector<Index> sample_indices = { 0 };

    vector<float> inputs(size_t(pixels), -7.0f);
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), FillMode::Training,
                              ColumnContiguity::Unknown);

    for (Index i = 0; i < pixels; ++i)
    {
        EXPECT_GE(inputs[size_t(i)], 0.0f);
        EXPECT_LE(inputs[size_t(i)], 1.0f);
    }

    float maximum = 0.0f;
    for (Index i = 0; i < pixels; ++i)
        maximum = max(maximum, inputs[size_t(i)]);

    EXPECT_GT(maximum, 0.5f);
}

TEST(ImageDataset, FillInputsRawWhenNotTraining)
{
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    const Index pixels = image_dataset.get_input_shape()[0]
                       * image_dataset.get_input_shape()[1]
                       * image_dataset.get_input_shape()[2];

    const vector<Index> input_indices = image_dataset.get_feature_indices("Input");
    const vector<Index> sample_indices = { 0 };

    vector<float> inputs(size_t(pixels), -1.0f);
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), FillMode::Inference,
                              ColumnContiguity::Unknown);

    float maximum = 0.0f;
    for (Index i = 0; i < pixels; ++i)
    {
        EXPECT_GE(inputs[size_t(i)], 0.0f);
        EXPECT_LE(inputs[size_t(i)], 255.0f);
        maximum = max(maximum, inputs[size_t(i)]);
    }

    EXPECT_GT(maximum, 1.0f);
}

TEST(ImageDataset, ConcurrentMixedSizeLoadsAndCacheReadsAreIndependent)
{
    ImageFixture small_fixture(3, 2, 3);
    ImageFixture large_fixture(7, 5, 4);

    const auto load = [](const filesystem::path& root)
    {
        ImageDataset dataset(root);
        dataset.set_display(false);

        vector<Index> samples(size_t(dataset.get_samples_number()));
        iota(samples.begin(), samples.end(), Index(0));
        const Shape& shape = dataset.get_input_shape();
        const Index sample_size = shape[0] * shape[1] * shape[2];
        vector<float> output(size_t(ssize(samples) * sample_size));
        dataset.fill_inputs(samples, dataset.get_feature_indices("Input"),
                            output.data(), FillMode::Inference);
        return pair{shape, std::move(output)};
    };

    future<pair<Shape, vector<float>>> small =
        async(launch::async, load, small_fixture.root);
    future<pair<Shape, vector<float>>> large =
        async(launch::async, load, large_fixture.root);

    const auto validate = [](const pair<Shape, vector<float>>& result)
    {
        const Shape& shape = result.first;
        const vector<float>& pixels = result.second;
        EXPECT_EQ(shape.get_rank(), 3);
        ASSERT_EQ(pixels.size() % 3, size_t(0));
        for (size_t i = 0; i < pixels.size(); i += 3)
        {
            const bool red = pixels[i] == 200.0f && pixels[i + 1] == 10.0f
                          && pixels[i + 2] == 10.0f;
            const bool blue = pixels[i] == 10.0f && pixels[i + 1] == 10.0f
                           && pixels[i + 2] == 200.0f;
            EXPECT_TRUE(red || blue);
        }
    };

    validate(small.get());
    validate(large.get());
}

TEST(ImageDataset, SetInputScalingMinimumMaximum)
{
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    const Index channels = image_dataset.get_channels_number();
    ASSERT_EQ(channels, 3);

    const size_t channels_count = size_t(channels);
    vector<Descriptives> descriptives(channels_count);
    vector<ScalerMethod> scalers(channels_count, ScalerMethod::MinimumMaximum);

    for (Index c = 0; c < channels; ++c)
    {
        descriptives[size_t(c)].minimum = 0.0f;
        descriptives[size_t(c)].maximum = 255.0f;
    }

    EXPECT_NO_THROW(image_dataset.set_input_scaling(descriptives, scalers, 0.0f, 1.0f));

    const Index pixels = image_dataset.get_input_shape()[0]
                       * image_dataset.get_input_shape()[1]
                       * image_dataset.get_input_shape()[2];

    const vector<Index> input_indices = image_dataset.get_feature_indices("Input");
    const vector<Index> sample_indices = { 0 };

    vector<float> inputs(size_t(pixels), 0.0f);
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), FillMode::Training,
                              ColumnContiguity::Unknown);

    for (Index i = 0; i < pixels; ++i)
    {
        EXPECT_GE(inputs[size_t(i)], 0.0f);
        EXPECT_LE(inputs[size_t(i)], 1.0f);
    }
}

TEST(ImageDataset, SetInputScalingChannelMismatchThrows)
{
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    vector<Descriptives> descriptives(1);
    vector<ScalerMethod> scalers(1, ScalerMethod::MinimumMaximum);

    EXPECT_ANY_THROW(image_dataset.set_input_scaling(descriptives, scalers, 0.0f, 1.0f));
}

TEST(ImageDataset, SetInputScalingDisablesDeviceResidency)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";
    ImageFixture fixture(2, 2, 1);
    ImageDataset dataset(fixture.root);
    dataset.enable_device_residency();
    ASSERT_TRUE(dataset.is_device_resident());

    const Index channels = dataset.get_input_shape()[2];
    vector<Descriptives> descriptives(size_t(channels), Descriptives{});
    for (Descriptives& descriptive : descriptives)
    {
        descriptive.minimum = 0.0f;
        descriptive.maximum = 255.0f;
    }
    dataset.set_input_scaling(descriptives,
                              vector<ScalerMethod>(size_t(channels), ScalerMethod::MinimumMaximum),
                              -1.0f, 1.0f);
    EXPECT_FALSE(dataset.is_device_resident());
    EXPECT_EQ(dataset.get_device_data_columns(), 0);
}

TEST(ImageDataset, SetAugmentationPolicyDisablesDeviceResidency)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);
    image_dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

    image_dataset.enable_device_residency();
    ASSERT_TRUE(image_dataset.is_device_resident());

    ImageAugmentationPolicy invalid;
    invalid.rotation_minimum = 1.0f;
    EXPECT_THROW(image_dataset.set_augmentation_policy(invalid), runtime_error);
    EXPECT_TRUE(image_dataset.is_device_resident());

    ImageAugmentationPolicy policy;
    policy.enabled = true;
    policy.reflection_axis_x = true;

    image_dataset.set_augmentation_policy(policy);

    EXPECT_FALSE(image_dataset.is_device_resident());

    image_dataset.enable_device_residency();
    EXPECT_FALSE(image_dataset.is_device_resident());
}

TEST(ImageDataset, LoadingAugmentationDisablesDeviceResidency)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    ImageFixture fixture(2, 2, 1);

    ImageDataset configured_dataset(fixture.root);
    ImageAugmentationPolicy policy;
    policy.enabled = true;
    policy.reflection_axis_x = true;
    configured_dataset.set_augmentation_policy(policy);

    JsonWriter writer;
    configured_dataset.to_JSON(writer);

    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));

    ImageDataset resident_dataset(fixture.root);
    resident_dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
    resident_dataset.enable_device_residency();
    ASSERT_TRUE(resident_dataset.is_device_resident());

    resident_dataset.from_JSON(document);

    EXPECT_FALSE(resident_dataset.is_device_resident());

    resident_dataset.enable_device_residency();
    EXPECT_FALSE(resident_dataset.is_device_resident());
}

TEST(ImageDataset, AugmentInputsDisabledLeavesDataUnchanged)
{
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    ImageAugmentationPolicy policy;
    policy.enabled = false;
    image_dataset.set_augmentation_policy(policy);

    const Index pixels = image_dataset.get_input_shape()[0]
                       * image_dataset.get_input_shape()[1]
                       * image_dataset.get_input_shape()[2];

    const size_t pixel_count = size_t(pixels);
    vector<float> data(pixel_count);
    for (Index i = 0; i < pixels; ++i)
        data[size_t(i)] = float(i);

    const vector<float> original = data;

    image_dataset.augment_inputs(data, 1);

    for (Index i = 0; i < pixels; ++i)
        EXPECT_FLOAT_EQ(data[size_t(i)], original[size_t(i)]);
}

TEST(ImageDataset, AugmentationRequiresTheActiveTrainingPartition)
{
    ImageFixture fixture(4, 3, 2);
    ImageDataset dataset(fixture.root);
    dataset.set_sample_roles({"Validation", "Training", "Testing", "None"});
    ImageAugmentationPolicy policy;
    policy.enabled = true;
    policy.horizontal_translation_minimum = policy.horizontal_translation_maximum = 1.0f;
    dataset.set_augmentation_policy(policy);
    const auto features = dataset.get_feature_indices(VariableRole::Input);
    vector<float> input(size_t(2 * dataset.get_input_shape().size()), -99.0f);
    for (Index excluded : {0, 2, 3, -1, 4})
    {
        EXPECT_THROW(dataset.fill_inputs({1, excluded}, features, input.data(), FillMode::Training), runtime_error);
        EXPECT_TRUE(ranges::all_of(input, [](float value) { return value == -99.0f; }));
    }
    dataset.fill_inputs({1}, features, input.data(), FillMode::Training);
    EXPECT_FLOAT_EQ(input[0], 0.0f);
    for (FillMode mode : {FillMode::Validation, FillMode::Inference})
    {
        dataset.fill_inputs({0}, features, input.data(), mode);
        const auto expected = input;
        policy.enabled = false;
        dataset.set_augmentation_policy(policy);
        dataset.fill_inputs({0}, features, input.data(), mode);
        EXPECT_EQ(input, expected);
        policy.enabled = true;
        dataset.set_augmentation_policy(policy);
    }
    {
        FoldScope fold(dataset, {0}, {1});
        EXPECT_NO_THROW(dataset.fill_inputs({0}, features, input.data(), FillMode::Training));
        EXPECT_THROW(dataset.fill_inputs({1}, features, input.data(), FillMode::Training), runtime_error);
    }
    EXPECT_NO_THROW(dataset.fill_inputs({1}, features, input.data(), FillMode::Training));
    EXPECT_THROW(dataset.fill_inputs({0}, features, input.data(), FillMode::Training), runtime_error);
}

TEST(ImageDataset, AugmentationPolicyValidationIsAtomicAndAppliesToJson)
{
    ImageFixture fixture(4, 3, 2);
    ImageDataset dataset(fixture.root);
    const string original = augmentation_json(dataset);
    using Policy = ImageAugmentationPolicy;
    const float limit = -float(numeric_limits<Index>::min());
    for (bool enabled : {false, true})
        for (auto [low, high] : {pair{&Policy::rotation_minimum, &Policy::rotation_maximum},
                                 pair{&Policy::horizontal_translation_minimum, &Policy::horizontal_translation_maximum},
                                 pair{&Policy::vertical_translation_minimum, &Policy::vertical_translation_maximum}})
        {
            Policy policy;
            policy.enabled = enabled;
            policy.*low = 1.0f;
            EXPECT_THROW(dataset.set_augmentation_policy(policy), runtime_error);
            for (auto field : {low, high})
                for (float value : {numeric_limits<float>::quiet_NaN(), numeric_limits<float>::infinity(),
                                    -numeric_limits<float>::infinity(), limit, -limit})
                {
                    if (low == &Policy::rotation_minimum && isfinite(value)) continue;
                    policy.*low = policy.*high = 0.0f;
                    policy.*field = value;
                    EXPECT_THROW(dataset.set_augmentation_policy(policy), runtime_error);
                    EXPECT_EQ(augmentation_json(dataset), original);
                }
        }
    for (const char* field : {"RandomRotationMinimum", "RandomHorizontalTranslationMinimum",
                             "RandomVerticalTranslationMinimum"})
    {
        JsonDocument document;
        document.set_root(Json::parse(original));
        document.get_root()["Dataset"]["DataSource"].set(field, 1.0f);
        EXPECT_THROW(dataset.from_JSON(document), runtime_error);
    }
}

TEST(ImageDataset, AugmentationAcceptsFiniteAnglesAndSafePixelShifts)
{
    ImageFixture fixture(16, 12, 1);
    ImageDataset dataset(fixture.root);
    vector<float> original(16 * 12 * 3, 1.0f);
    ImageAugmentationPolicy policy;
    policy.enabled = true;
    policy.rotation_minimum = policy.rotation_maximum = 720.0f;
    dataset.set_augmentation_policy(policy);
    auto input = original;
    dataset.augment_inputs(input, 1);
    EXPECT_EQ(input, original);
    policy.rotation_minimum = -numeric_limits<float>::max();
    policy.rotation_maximum = numeric_limits<float>::max();
    dataset.set_augmentation_policy(policy);
    dataset.augment_inputs(input, 1);
    EXPECT_TRUE(ranges::all_of(input, [](float x) { return isfinite(x); }));
    EXPECT_TRUE(ranges::any_of(input, [](float x) { return x > 0.0f; }));
    policy.rotation_minimum = policy.rotation_maximum = 0.0f;
    const float shift = nextafter(-float(numeric_limits<Index>::min()), 0.0f);
    for (float amount : {-shift, shift})
    {
        policy.horizontal_translation_minimum = policy.horizontal_translation_maximum = amount;
        policy.vertical_translation_minimum = policy.vertical_translation_maximum = amount;
        dataset.set_augmentation_policy(policy);
        input = original;
        dataset.augment_inputs(input, 1);
        EXPECT_TRUE(ranges::all_of(input, [](float x) { return x == 0.0f; }));
    }
}

TEST(ImageDataset, AugmentationReplaysAcrossWorkersBatchesAndEpochs)
{
    ImageFixture fixture(16, 12, 4);
    for (auto storage : {Dataset::StorageMode::BinaryFile, Dataset::StorageMode::Matrix})
    {
        ImageDataset dataset;
        dataset.set_display(false);
        dataset.set_storage_mode(storage);
        dataset.set_data_path(fixture.root);
        dataset.read_images();
        dataset.set_sample_roles(SampleRole::Training);
        ImageAugmentationPolicy policy;
        policy.enabled = policy.reflection_axis_x = policy.reflection_axis_y = true;
        policy.rotation_minimum = -20.0f;
        policy.rotation_maximum = 20.0f;
        policy.horizontal_translation_minimum = policy.vertical_translation_minimum = -3.0f;
        policy.horizontal_translation_maximum = policy.vertical_translation_maximum = 3.0f;
        dataset.set_augmentation_policy(policy);
        expect_augmentation_replay(dataset, false);
    }
}

TEST(ImageDataset, ConstructFromPathSingleClassThrows)
{
    const filesystem::path root = filesystem::temp_directory_path()
                                / "opennn_image_dataset_single_class_test";

    filesystem::remove_all(root);
    filesystem::create_directories(root / "only");
    write_bmp_24(root / "only" / "img.bmp", 2, 2, 100, 100, 100);

    EXPECT_ANY_THROW({ ImageDataset image_dataset(root); });

    error_code ec;
    filesystem::remove_all(root, ec);
}
