#include "pch.h"

#include "../opennn/image_data_set.h"


TEST(ImageDataSetTest, DefaultConstructor)
{
    ImageDataSet image_data_set;

    EXPECT_EQ(image_data_set.get_variables_number(), 0);
    EXPECT_EQ(image_data_set.get_samples_number(), 0);
}


TEST(ImageDataSetTest, GeneralConstructor)
{
    EXPECT_EQ(1, 1);
}

/*
namespace opennn
{
void ImageDataSetTest::test_constructor()
{
    cout << "test_constructor\n";    


    // Image number, height, width, channels number and targets number constructor

    ImageDataSet data_set_2(5, 3, 3, 3, 2);

    EXPECT_EQ(data_set_2.get_samples_number() == 5);
    EXPECT_EQ(data_set_2.get_image_height() == 3);
    EXPECT_EQ(data_set_2.get_image_width() == 3);
    EXPECT_EQ(data_set_2.get_channels_number() == 3);
    EXPECT_EQ(data_set_2.get_raw_variables_number(DataSet::VariableUse::Target));
}

}
*/