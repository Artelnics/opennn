#include "tests/pch.h"

#include "opennn/dataset/image_dataset.h"
#include "opennn/dataset/dataset.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/json.h"
#include "opennn/core/variable.h"
#include "opennn/core/statistics.h"

#include "tests/test_helpers.h"

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
    image_dataset.fill_targets(sample_indices, target_indices, targets.data(), FillMode::Inference, -1);

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
    image_dataset.fill_targets(sample_indices, target_indices, targets.data(), FillMode::Inference, -1);

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
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), FillMode::Training, -1);

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
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), FillMode::Inference, -1);

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
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), FillMode::Training, -1);

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

TEST(ImageDataset, SetAugmentationDisablesDeviceResidency)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);
    image_dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

    image_dataset.enable_device_residency();
    ASSERT_TRUE(image_dataset.is_device_resident());

    AugmentationSettings augmentation;
    augmentation.enabled = true;
    augmentation.reflection_axis_x = true;

    image_dataset.set_augmentation(augmentation);

    EXPECT_FALSE(image_dataset.is_device_resident());

    image_dataset.enable_device_residency();
    EXPECT_FALSE(image_dataset.is_device_resident());
}

TEST(ImageDataset, LoadingAugmentationDisablesDeviceResidency)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    ImageFixture fixture(2, 2, 1);

    ImageDataset configured_dataset(fixture.root);
    AugmentationSettings augmentation;
    augmentation.enabled = true;
    augmentation.reflection_axis_x = true;
    configured_dataset.set_augmentation(augmentation);

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

    AugmentationSettings augmentation;
    augmentation.enabled = false;
    image_dataset.set_augmentation(augmentation);

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
