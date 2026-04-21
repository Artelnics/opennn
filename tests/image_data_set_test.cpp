#include "pch.h"

#include "../opennn/image_dataset.h"

using namespace opennn;

TEST(ImageDataset, DefaultConstructor)
{
    ImageDataset image_data_set;

    EXPECT_EQ(image_data_set.get_variables_number(), 0);
    EXPECT_EQ(image_data_set.get_samples_number(), 0);
}


TEST(ImageDataset, GeneralConstructor)
{
    ImageDataset image_data_set(5, { 4, 3, 2 }, { 1 });

    EXPECT_EQ(image_data_set.get_samples_number(), 5);
    // get_image_height() and get_image_width() are not part of the public API
    EXPECT_EQ(image_data_set.get_channels_number(), 2);
    EXPECT_EQ(image_data_set.get_variables_number("Target"), 1);

    // Verify input shape instead
    Shape input_shape = image_data_set.get_input_shape();
    EXPECT_EQ(input_shape.rank(), 3);
    EXPECT_EQ(input_shape[0], 4);  // height
    EXPECT_EQ(input_shape[1], 3);  // width
    EXPECT_EQ(input_shape[2], 2);  // channels
}
