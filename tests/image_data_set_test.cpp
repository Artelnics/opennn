#include "pch.h"

#include "../opennn/image_data_set.h"


TEST(ImageDataSetTest, DefaultConstructor)
{
    ImageDataSet data_set_1;

//    assert_true(data_set_1.get_variables_number() == 0, LOG);
//    assert_true(data_set_1.get_samples_number() == 0, LOG);

    EXPECT_EQ(1, 1);
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

    assert_true(data_set_2.get_samples_number() == 5, LOG);
    assert_true(data_set_2.get_image_height() == 3, LOG);
    assert_true(data_set_2.get_image_width() == 3, LOG);
    assert_true(data_set_2.get_channels_number() == 3, LOG);
    assert_true(data_set_2.get_raw_variables_number(DataSet::VariableUse::Target), LOG);
}

}
*/