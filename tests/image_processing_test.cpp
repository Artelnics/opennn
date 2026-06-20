#include "pch.h"
#include "../opennn/image_processing.h"

using namespace opennn;

TEST(ImageProcessingTest, IsSupportedImageFile)
{
    EXPECT_TRUE(is_supported_image_file("photo.bmp"));
    EXPECT_TRUE(is_supported_image_file("photo.png"));
    EXPECT_TRUE(is_supported_image_file("photo.jpg"));
    EXPECT_TRUE(is_supported_image_file("photo.jpeg"));

    EXPECT_TRUE(is_supported_image_file("PHOTO.BMP"));
    EXPECT_TRUE(is_supported_image_file("PHOTO.PNG"));
    EXPECT_TRUE(is_supported_image_file("PHOTO.JPG"));

    EXPECT_FALSE(is_supported_image_file("data.csv"));
    EXPECT_FALSE(is_supported_image_file("model.txt"));
    EXPECT_FALSE(is_supported_image_file("noextension"));
    EXPECT_FALSE(is_supported_image_file("archive.gif"));
}


TEST(ImageProcessingTest, ResizeIdentitySameSize)
{
    const Index height = 3;
    const Index width = 3;
    const Index channels = 1;

    Tensor3 input(height, width, channels);
    for (Index y = 0; y < height; ++y)
        for (Index x = 0; x < width; ++x)
            input(y, x, 0) = float(y * width + x);

    const Tensor3 output = resize_image(input, height, width);

    ASSERT_EQ(output.dimension(0), height);
    ASSERT_EQ(output.dimension(1), width);
    ASSERT_EQ(output.dimension(2), channels);

    for (Index y = 0; y < height; ++y)
        for (Index x = 0; x < width; ++x)
            EXPECT_NEAR(output(y, x, 0), input(y, x, 0), 1e-5f);
}


TEST(ImageProcessingTest, ResizeConstantPreservesValue)
{
    const Index in_height = 2;
    const Index in_width = 2;
    const Index channels = 3;

    Tensor3 input(in_height, in_width, channels);
    input.setConstant(7.5f);

    const Tensor3 output = resize_image(input, 5, 4);

    ASSERT_EQ(output.dimension(0), 5);
    ASSERT_EQ(output.dimension(1), 4);
    ASSERT_EQ(output.dimension(2), channels);

    for (Index i = 0; i < output.size(); ++i)
        EXPECT_NEAR(output.data()[i], 7.5f, 1e-5f);
}


TEST(ImageProcessingTest, ResizeKeepsTopLeftCorner)
{
    const Index in_height = 4;
    const Index in_width = 4;
    const Index channels = 1;

    Tensor3 input(in_height, in_width, channels);
    input.setConstant(0.0f);
    input(0, 0, 0) = 100.0f;

    const Tensor3 output = resize_image(input, 8, 8);

    EXPECT_NEAR(output(0, 0, 0), 100.0f, 1e-5f);
}


TEST(ImageProcessingTest, ResizeUpsampleMidpointInterpolation)
{
    const Index channels = 1;

    Tensor3 input(1, 2, channels);
    input(0, 0, 0) = 0.0f;
    input(0, 1, 0) = 10.0f;

    const Tensor3 output = resize_image(input, 1, 3);

    ASSERT_EQ(output.dimension(1), 3);

    EXPECT_NEAR(output(0, 0, 0), 0.0f, 1e-5f);
    EXPECT_GT(output(0, 1, 0), 0.0f);
    EXPECT_LT(output(0, 1, 0), 10.0f);
    EXPECT_NEAR(output(0, 1, 0), 6.6666667f, 1e-4f);
    EXPECT_NEAR(output(0, 2, 0), 10.0f, 1e-5f);
}


TEST(ImageProcessingTest, ReflectHorizontalSwapsColumns)
{
    const Index height = 2;
    const Index width = 3;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    for (Index y = 0; y < height; ++y)
        for (Index x = 0; x < width; ++x)
            image(y, x, 0) = float(x);

    TensorMap3 image_map(image.data(), height, width, channels);
    reflect_image_horizontal(image_map);

    for (Index y = 0; y < height; ++y)
    {
        EXPECT_NEAR(image(y, 0, 0), 2.0f, 1e-6f);
        EXPECT_NEAR(image(y, 1, 0), 1.0f, 1e-6f);
        EXPECT_NEAR(image(y, 2, 0), 0.0f, 1e-6f);
    }
}


TEST(ImageProcessingTest, ReflectHorizontalTwiceIsIdentity)
{
    const Index height = 3;
    const Index width = 4;
    const Index channels = 2;

    Tensor3 image(height, width, channels);
    for (Index i = 0; i < image.size(); ++i)
        image.data()[i] = float(i);

    Tensor3 original = image;

    TensorMap3 image_map(image.data(), height, width, channels);
    reflect_image_horizontal(image_map);
    reflect_image_horizontal(image_map);

    for (Index i = 0; i < image.size(); ++i)
        EXPECT_NEAR(image.data()[i], original.data()[i], 1e-6f);
}


TEST(ImageProcessingTest, ReflectVerticalSwapsRows)
{
    const Index height = 3;
    const Index width = 2;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    for (Index y = 0; y < height; ++y)
        for (Index x = 0; x < width; ++x)
            image(y, x, 0) = float(y);

    TensorMap3 image_map(image.data(), height, width, channels);
    reflect_image_vertical(image_map);

    for (Index x = 0; x < width; ++x)
    {
        EXPECT_NEAR(image(0, x, 0), 2.0f, 1e-6f);
        EXPECT_NEAR(image(1, x, 0), 1.0f, 1e-6f);
        EXPECT_NEAR(image(2, x, 0), 0.0f, 1e-6f);
    }
}


TEST(ImageProcessingTest, ReflectVerticalTwiceIsIdentity)
{
    const Index height = 4;
    const Index width = 3;
    const Index channels = 3;

    Tensor3 image(height, width, channels);
    for (Index i = 0; i < image.size(); ++i)
        image.data()[i] = float(i * 2 + 1);

    Tensor3 original = image;

    TensorMap3 image_map(image.data(), height, width, channels);
    reflect_image_vertical(image_map);
    reflect_image_vertical(image_map);

    for (Index i = 0; i < image.size(); ++i)
        EXPECT_NEAR(image.data()[i], original.data()[i], 1e-6f);
}


TEST(ImageProcessingTest, TranslateXPositiveShift)
{
    const Index height = 1;
    const Index width = 4;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    for (Index x = 0; x < width; ++x)
        image(0, x, 0) = float(x + 1);

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_x(image_map, 1);

    EXPECT_NEAR(image(0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(image(0, 1, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(image(0, 2, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(image(0, 3, 0), 3.0f, 1e-6f);
}


TEST(ImageProcessingTest, TranslateXNegativeShift)
{
    const Index height = 1;
    const Index width = 4;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    for (Index x = 0; x < width; ++x)
        image(0, x, 0) = float(x + 1);

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_x(image_map, -1);

    EXPECT_NEAR(image(0, 0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(image(0, 1, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(image(0, 2, 0), 4.0f, 1e-6f);
    EXPECT_NEAR(image(0, 3, 0), 0.0f, 1e-6f);
}


TEST(ImageProcessingTest, TranslateXZeroShiftUnchanged)
{
    const Index height = 2;
    const Index width = 3;
    const Index channels = 2;

    Tensor3 image(height, width, channels);
    for (Index i = 0; i < image.size(); ++i)
        image.data()[i] = float(i + 5);

    Tensor3 original = image;

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_x(image_map, 0);

    for (Index i = 0; i < image.size(); ++i)
        EXPECT_NEAR(image.data()[i], original.data()[i], 1e-6f);
}


TEST(ImageProcessingTest, TranslateXShiftBeyondWidthClearsImage)
{
    const Index height = 2;
    const Index width = 3;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    image.setConstant(9.0f);

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_x(image_map, 5);

    for (Index i = 0; i < image.size(); ++i)
        EXPECT_NEAR(image.data()[i], 0.0f, 1e-6f);
}


TEST(ImageProcessingTest, TranslateXPreservesChannels)
{
    const Index height = 1;
    const Index width = 3;
    const Index channels = 3;

    Tensor3 image(height, width, channels);
    image(0, 0, 0) = 1.0f; image(0, 0, 1) = 2.0f; image(0, 0, 2) = 3.0f;
    image(0, 1, 0) = 4.0f; image(0, 1, 1) = 5.0f; image(0, 1, 2) = 6.0f;
    image(0, 2, 0) = 7.0f; image(0, 2, 1) = 8.0f; image(0, 2, 2) = 9.0f;

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_x(image_map, 1);

    EXPECT_NEAR(image(0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(image(0, 0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(image(0, 0, 2), 0.0f, 1e-6f);

    EXPECT_NEAR(image(0, 1, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(image(0, 1, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(image(0, 1, 2), 3.0f, 1e-6f);

    EXPECT_NEAR(image(0, 2, 0), 4.0f, 1e-6f);
    EXPECT_NEAR(image(0, 2, 1), 5.0f, 1e-6f);
    EXPECT_NEAR(image(0, 2, 2), 6.0f, 1e-6f);
}


TEST(ImageProcessingTest, TranslateYPositiveShift)
{
    const Index height = 4;
    const Index width = 1;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    for (Index y = 0; y < height; ++y)
        image(y, 0, 0) = float(y + 1);

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_y(image_map, 1);

    EXPECT_NEAR(image(0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(image(1, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(image(2, 0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(image(3, 0, 0), 3.0f, 1e-6f);
}


TEST(ImageProcessingTest, TranslateYNegativeShift)
{
    const Index height = 4;
    const Index width = 1;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    for (Index y = 0; y < height; ++y)
        image(y, 0, 0) = float(y + 1);

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_y(image_map, -1);

    EXPECT_NEAR(image(0, 0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(image(1, 0, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(image(2, 0, 0), 4.0f, 1e-6f);
    EXPECT_NEAR(image(3, 0, 0), 0.0f, 1e-6f);
}


TEST(ImageProcessingTest, TranslateYShiftBeyondHeightClearsImage)
{
    const Index height = 3;
    const Index width = 2;
    const Index channels = 1;

    Tensor3 image(height, width, channels);
    image.setConstant(4.0f);

    TensorMap3 image_map(image.data(), height, width, channels);
    translate_image_y(image_map, -7);

    for (Index i = 0; i < image.size(); ++i)
        EXPECT_NEAR(image.data()[i], 0.0f, 1e-6f);
}


TEST(ImageProcessingTest, RotateZeroIsIdentity)
{
    const Index height = 3;
    const Index width = 3;
    const Index channels = 2;

    Tensor3 input(height, width, channels);
    for (Index i = 0; i < input.size(); ++i)
        input.data()[i] = float(i + 1);

    Tensor3 output(height, width, channels);
    output.setConstant(-1.0f);

    const TensorMap3 input_map(input.data(), height, width, channels);
    TensorMap3 output_map(output.data(), height, width, channels);

    rotate_image(input_map, output_map, 0.0f);

    for (Index i = 0; i < output.size(); ++i)
        EXPECT_NEAR(output.data()[i], input.data()[i], 1e-6f);
}


TEST(ImageProcessingTest, RotateFullCircleApproximatesInput)
{
    const Index height = 5;
    const Index width = 5;
    const Index channels = 1;

    Tensor3 input(height, width, channels);
    input.setConstant(3.0f);

    Tensor3 output(height, width, channels);
    output.setConstant(0.0f);

    const TensorMap3 input_map(input.data(), height, width, channels);
    TensorMap3 output_map(output.data(), height, width, channels);

    rotate_image(input_map, output_map, 360.0f);

    EXPECT_NEAR(output(2, 2, 0), 3.0f, 1e-4f);
}


TEST(ImageProcessingTest, RotateOutOfBoundsFillsZero)
{
    const Index height = 4;
    const Index width = 4;
    const Index channels = 1;

    Tensor3 input(height, width, channels);
    input.setConstant(5.0f);

    Tensor3 output(height, width, channels);
    output.setConstant(99.0f);

    const TensorMap3 input_map(input.data(), height, width, channels);
    TensorMap3 output_map(output.data(), height, width, channels);

    rotate_image(input_map, output_map, 45.0f);

    EXPECT_NEAR(output(0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(output(2, 2, 0), 5.0f, 1e-6f);
}
