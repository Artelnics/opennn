#include "tests/common/pch.h"

#include "opennn/dataset/segmentation_dataset.h"
#include "opennn/core/json.h"
#include "opennn/core/variable.h"

#include "tests/common/test_helpers.h"

#include <algorithm>
#include <filesystem>

using namespace opennn;
using namespace opennn_test;

namespace
{

// A single-channel 8-bit BMP with a grayscale palette -- the form a binary mask
// actually takes, and the only one SegmentationDataset accepts. The left half
// of every row is foreground, so a nearest-neighbour resample to a 2x2 grid
// must keep that split rather than blur it.
void write_mask_bmp(const filesystem::path& path, int width, int height)
{
    constexpr int bytes_per_pixel = 1;
    constexpr std::uint32_t header_size = 54;
    constexpr std::uint32_t palette_entries = 256;
    constexpr std::uint32_t pixel_offset = header_size + palette_entries * 4;

    const int row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;
    const std::uint32_t pixel_array_size = std::uint32_t(row_stride * height);

    std::vector<std::uint8_t> bytes;
    bytes.reserve(pixel_offset + pixel_array_size);

    const auto push_u16 = [&bytes](std::uint16_t value)
    {
        bytes.push_back(std::uint8_t(value & 0xFF));
        bytes.push_back(std::uint8_t((value >> 8) & 0xFF));
    };

    const auto push_u32 = [&bytes](std::uint32_t value)
    {
        bytes.push_back(std::uint8_t(value & 0xFF));
        bytes.push_back(std::uint8_t((value >> 8) & 0xFF));
        bytes.push_back(std::uint8_t((value >> 16) & 0xFF));
        bytes.push_back(std::uint8_t((value >> 24) & 0xFF));
    };

    bytes.push_back('B');
    bytes.push_back('M');
    push_u32(pixel_offset + pixel_array_size);
    push_u16(0);
    push_u16(0);
    push_u32(pixel_offset);

    push_u32(40);
    push_u32(std::uint32_t(width));
    push_u32(std::uint32_t(height));
    push_u16(1);
    push_u16(8);
    push_u32(0);
    push_u32(pixel_array_size);
    push_u32(2835);
    push_u32(2835);
    push_u32(palette_entries);
    push_u32(palette_entries);

    for (std::uint32_t i = 0; i < palette_entries; ++i)
    {
        bytes.push_back(std::uint8_t(i));      // blue
        bytes.push_back(std::uint8_t(i));      // green
        bytes.push_back(std::uint8_t(i));      // red
        bytes.push_back(0);                    // reserved
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
            bytes.push_back(x < width / 2 ? 255 : 0);
        for (int pad = width * bytes_per_pixel; pad < row_stride; ++pad)
            bytes.push_back(0);
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}


struct SegmentationFixture
{
    filesystem::path root;

    SegmentationFixture(int width, int height, int samples)
    {
        root = filesystem::temp_directory_path()
             / ("opennn_segmentation_dataset_test_" + to_string(uint64_t(this) ^ uint64_t(width * 131 + height)));

        filesystem::remove_all(root);

        const filesystem::path images = root / "images";
        const filesystem::path masks  = root / "masks";

        filesystem::create_directories(images);
        filesystem::create_directories(masks);

        for (int i = 0; i < samples; ++i)
        {
            write_bmp_24(images / ("sample_" + to_string(i) + ".bmp"), width, height, 200, 100, 50);
            write_mask_bmp(masks / ("sample_" + to_string(i) + ".bmp"), width, height);
        }
    }

    ~SegmentationFixture()
    {
        error_code ec;
        filesystem::remove_all(root, ec);
    }

    filesystem::path images() const { return root / "images"; }
    filesystem::path masks() const { return root / "masks"; }
};

}

TEST(SegmentationDataset, DefaultConstructorIsEmpty)
{
    SegmentationDataset dataset;

    EXPECT_EQ(dataset.get_samples_number(), 0);
    EXPECT_TRUE(dataset.is_empty());
}

TEST(SegmentationDataset, ConstructFromImageAndMaskDirectories)
{
    SegmentationFixture fixture(4, 3, 2);

    SegmentationDataset dataset(fixture.images(), fixture.masks());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_masks_number(), 2);

    const Shape input_shape = dataset.get_input_shape();
    ASSERT_EQ(input_shape.get_rank(), 3);
    EXPECT_EQ(input_shape[0], 3);
    EXPECT_EQ(input_shape[1], 4);
    EXPECT_EQ(input_shape[2], 3);

    const Shape target_shape = dataset.get_target_shape();
    ASSERT_EQ(target_shape.get_rank(), 3);
    EXPECT_EQ(target_shape[0], 3);
    EXPECT_EQ(target_shape[1], 4);
    EXPECT_EQ(target_shape[2], 1);

    EXPECT_EQ(dataset.get_variables_number(), 2);
    EXPECT_EQ(dataset.get_features_number("Input"), 3 * 4 * 3);
    EXPECT_EQ(dataset.get_features_number("Target"), 3 * 4);
    EXPECT_EQ(dataset.get_variable_type(1), VariableType::Binary);
}

TEST(SegmentationDataset, RequestedShapeResizesImages)
{
    SegmentationFixture fixture(8, 8, 1);

    SegmentationDataset dataset(fixture.images(), fixture.masks(), Shape{4, 4, 3});

    EXPECT_EQ(dataset.get_input_shape(), (Shape{4, 4, 3}));
    EXPECT_EQ(dataset.get_target_shape(), (Shape{4, 4, 1}));
}

TEST(SegmentationDataset, FillInputsNormalizesToUnitRange)
{
    SegmentationFixture fixture(2, 2, 1);

    SegmentationDataset dataset(fixture.images(), fixture.masks());

    vector<float> inputs(size_t(2 * 2 * 3), -1.0f);
    dataset.fill_inputs({ 0 }, dataset.get_feature_indices("Input"),
                        inputs.data(), FillMode::Training);

    for (const float value : inputs)
    {
        EXPECT_GE(value, 0.0f);
        EXPECT_LE(value, 1.0f);
    }

    // The fixture writes (200, 100, 50).
    EXPECT_FLOAT_EQ(inputs[0], 200.0f / 255.0f);
    EXPECT_FLOAT_EQ(inputs[1], 100.0f / 255.0f);
    EXPECT_FLOAT_EQ(inputs[2],  50.0f / 255.0f);
}

TEST(SegmentationDataset, FillTargetsIsBinaryMask)
{
    SegmentationFixture fixture(4, 4, 2);

    SegmentationDataset dataset(fixture.images(), fixture.masks());

    const vector<Index> target_indices = dataset.get_feature_indices("Target");
    ASSERT_EQ(ssize(target_indices), 16);

    vector<float> masks(size_t(2 * 16), -1.0f);
    dataset.fill_targets({ 0, 1 }, target_indices, masks.data(), FillMode::Training);

    for (const float value : masks)
    {
        EXPECT_TRUE(value == 0.0f || value == 1.0f);
    }

    // The fixture masks are foreground on the left half of every row.
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            EXPECT_FLOAT_EQ(masks[size_t(y * 4 + x)], x < 2 ? 1.0f : 0.0f);
}

TEST(SegmentationDataset, NearestNeighbourResizeKeepsMaskHard)
{
    SegmentationFixture fixture(4, 4, 1);

    SegmentationDataset dataset(fixture.images(), fixture.masks(), Shape{2, 2, 3});

    vector<float> masks(size_t(2 * 2), -1.0f);
    dataset.fill_targets({ 0 }, dataset.get_feature_indices("Target"),
                         masks.data(), FillMode::Inference);

    EXPECT_FLOAT_EQ(masks[0], 1.0f);
    EXPECT_FLOAT_EQ(masks[1], 0.0f);
    EXPECT_FLOAT_EQ(masks[2], 1.0f);
    EXPECT_FLOAT_EQ(masks[3], 0.0f);
}

TEST(SegmentationDataset, SplitRolesCoverEverySample)
{
    SegmentationFixture fixture(2, 2, 10);

    SegmentationDataset dataset(fixture.images(), fixture.masks());
    dataset.set_display(false);
    dataset.split_samples_random(0.6f, 0.2f, 0.2f);

    EXPECT_EQ(dataset.get_samples_number(), 10);
    EXPECT_GT(dataset.get_samples_number("Training"), 0);
    EXPECT_GT(dataset.get_samples_number("Validation"), 0);
    EXPECT_GT(dataset.get_samples_number("Testing"), 0);

    const vector<Index> training = dataset.get_sample_indices("Training");
    const vector<Index> validation = dataset.get_sample_indices("Validation");

    for (const Index index : validation)
        EXPECT_EQ(std::find(training.begin(), training.end(), index), training.end());
}

TEST(SegmentationDataset, MissingMaskThrows)
{
    SegmentationFixture fixture(2, 2, 2);

    error_code ec;
    filesystem::remove(fixture.masks() / "sample_1.bmp", ec);

    EXPECT_ANY_THROW({ SegmentationDataset dataset(fixture.images(), fixture.masks()); });
}

TEST(SegmentationDataset, PrepareTrainingScalingComputesRealStats)
{
    // Two solid-colour RGB images with known pixel values so we can compute
    // expected per-channel mean and std analytically.
    //
    // Sample 0: R=100, G=150, B=200  →  [100/255, 150/255, 200/255]
    // Sample 1: R= 50, G= 75, B=100  →  [ 50/255,  75/255, 100/255]
    //
    // All pixels in each sample are uniform, so for C images of size HxW:
    //   mean_c = (v0_c + v1_c) / (2 * 255)
    //   std_c  = |v0_c - v1_c| / (2 * 255)          (population std)

    SegmentationFixture fixture(4, 4, 0);  // 0 samples — we add manually below
    filesystem::create_directories(fixture.images());
    filesystem::create_directories(fixture.masks());

    write_bmp_24(fixture.images() / "s0.bmp", 4, 4, 100, 150, 200);
    write_bmp_24(fixture.images() / "s1.bmp", 4, 4,  50,  75, 100);
    write_mask_bmp(fixture.masks() / "s0.bmp", 4, 4);
    write_mask_bmp(fixture.masks() / "s1.bmp", 4, 4);

    SegmentationDataset dataset(fixture.images(), fixture.masks(), Shape{4, 4, 3});
    dataset.set_display(false);
    dataset.set_sample_roles(SampleRole::Training);  // all training

    FeatureScaling requested;
    requested.scalers.assign(3, ScalerMethod::MeanStandardDeviation);
    requested.descriptives.assign(3, Descriptives(-1.f, 1.f, 0.f, 1.f));

    const FeatureScaling fs = dataset.prepare_training_scaling(
        VariableRole::Input, requested, 3);

    ASSERT_EQ(fs.descriptives.size(), 3u);

    constexpr float tol = 1e-4f;

    // channel 0 (R): values 100/255 and 50/255
    EXPECT_NEAR(fs.descriptives[0].mean,               150.f / (2.f * 255.f), tol);
    EXPECT_NEAR(fs.descriptives[0].standard_deviation,  50.f / (2.f * 255.f), tol);

    // channel 1 (G): 150/255 and 75/255
    EXPECT_NEAR(fs.descriptives[1].mean,               225.f / (2.f * 255.f), tol);
    EXPECT_NEAR(fs.descriptives[1].standard_deviation,  75.f / (2.f * 255.f), tol);

    // channel 2 (B): 200/255 and 100/255
    EXPECT_NEAR(fs.descriptives[2].mean,               300.f / (2.f * 255.f), tol);
    EXPECT_NEAR(fs.descriptives[2].standard_deviation, 100.f / (2.f * 255.f), tol);
}

TEST(SegmentationDataset, JsonRoundTripRestoresShapes)
{
    SegmentationFixture fixture(4, 2, 2);

    SegmentationDataset dataset(fixture.images(), fixture.masks());

    JsonWriter writer;
    dataset.to_JSON(writer);

    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));

    SegmentationDataset loaded(fixture.images(), fixture.masks());
    loaded.from_JSON(document);

    EXPECT_EQ(loaded.get_samples_number(), dataset.get_samples_number());
    EXPECT_EQ(loaded.get_input_shape(), dataset.get_input_shape());
    EXPECT_EQ(loaded.get_target_shape(), dataset.get_target_shape());
    EXPECT_EQ(loaded.get_features_number("Input"), dataset.get_features_number("Input"));
    EXPECT_EQ(loaded.get_features_number("Target"), dataset.get_features_number("Target"));
}