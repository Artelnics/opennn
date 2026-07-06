#include "pch.h"

#include "../opennn/image_dataset.h"
#include "../opennn/dataset.h"
#include "../opennn/variable.h"
#include "../opennn/statistics.h"

#include <filesystem>
#include <fstream>
#include <cstdint>

using namespace opennn;

namespace
{

void write_u16(vector<uint8_t>& bytes, uint16_t value)
{
    bytes.push_back(uint8_t(value & 0xFF));
    bytes.push_back(uint8_t((value >> 8) & 0xFF));
}

void write_u32(vector<uint8_t>& bytes, uint32_t value)
{
    bytes.push_back(uint8_t(value & 0xFF));
    bytes.push_back(uint8_t((value >> 8) & 0xFF));
    bytes.push_back(uint8_t((value >> 16) & 0xFF));
    bytes.push_back(uint8_t((value >> 24) & 0xFF));
}

void write_bmp_24(const filesystem::path& path, int width, int height, uint8_t red, uint8_t green, uint8_t blue)
{
    const int bytes_per_pixel = 3;
    const int row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;
    const uint32_t pixel_array_size = uint32_t(row_stride * height);
    const uint32_t header_size = 54;
    const uint32_t file_size = header_size + pixel_array_size;

    vector<uint8_t> bytes;

    bytes.push_back('B');
    bytes.push_back('M');
    write_u32(bytes, file_size);
    write_u16(bytes, 0);
    write_u16(bytes, 0);
    write_u32(bytes, header_size);

    write_u32(bytes, 40);
    write_u32(bytes, uint32_t(width));
    write_u32(bytes, uint32_t(height));
    write_u16(bytes, 1);
    write_u16(bytes, 24);
    write_u32(bytes, 0);
    write_u32(bytes, pixel_array_size);
    write_u32(bytes, 2835);
    write_u32(bytes, 2835);
    write_u32(bytes, 0);
    write_u32(bytes, 0);

    for (int y = 0; y < height; ++y)
    {
        int written = 0;
        for (int x = 0; x < width; ++x)
        {
            bytes.push_back(blue);
            bytes.push_back(green);
            bytes.push_back(red);
            written += bytes_per_pixel;
        }
        while (written < row_stride)
        {
            bytes.push_back(0);
            ++written;
        }
    }

    ofstream out(path, ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), streamsize(bytes.size()));
    out.close();
}

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
    ASSERT_EQ(input_shape.rank, 3u);
    EXPECT_EQ(input_shape[0], 3);
    EXPECT_EQ(input_shape[1], 4);
    EXPECT_EQ(input_shape[2], 3);

    EXPECT_EQ(image_dataset.get_channels_number(), 3);

    const Shape target_shape = image_dataset.get_target_shape();
    ASSERT_EQ(target_shape.rank, 1u);
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
    ASSERT_EQ(target_shape.rank, 1u);
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
    image_dataset.fill_targets(sample_indices, target_indices, targets.data(), false, -1);

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
    image_dataset.fill_targets(sample_indices, target_indices, targets.data(), false, -1);

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
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), true, -1);

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
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), false, -1);

    float maximum = 0.0f;
    for (Index i = 0; i < pixels; ++i)
    {
        EXPECT_GE(inputs[size_t(i)], 0.0f);
        EXPECT_LE(inputs[size_t(i)], 255.0f);
        maximum = max(maximum, inputs[size_t(i)]);
    }

    EXPECT_GT(maximum, 1.0f);
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
    image_dataset.fill_inputs(sample_indices, input_indices, inputs.data(), true, -1);

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
    ImageFixture fixture(2, 2, 1);

    ImageDataset image_dataset(fixture.root);

    AugmentationSettings augmentation;
    augmentation.enabled = true;
    augmentation.reflection_axis_x = true;

    EXPECT_NO_THROW(image_dataset.set_augmentation(augmentation));
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

    image_dataset.augment_inputs(data.data(), 1);

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
