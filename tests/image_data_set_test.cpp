#include "pch.h"

#include "../opennn/image_dataset.h"


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
    EXPECT_EQ(image_data_set.get_image_height(), 4);
    EXPECT_EQ(image_data_set.get_image_width(), 3);
    EXPECT_EQ(image_data_set.get_channels_number(), 2);
    EXPECT_EQ(image_data_set.get_raw_variables_number(Dataset::VariableUse::Target), 1);

}

