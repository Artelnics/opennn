//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   S E R I E S   D A T A   S E T   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_data_set_test.h"

namespace opennn
{

ImageDataSetTest::ImageDataSetTest() : UnitTesting()
{
    data_set.set_display(false);
}


ImageDataSetTest::~ImageDataSetTest()
{
}


void ImageDataSetTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    ImageDataSet data_set_1;

    assert_true(data_set_1.get_variables_number() == 0, LOG);
    assert_true(data_set_1.get_samples_number() == 0, LOG);

    // Image number, height, width, channels number and targets number constructor

    ImageDataSet data_set_2(5, 3, 3, 3, 2);

    assert_true(data_set_2.get_samples_number() == 5, LOG);
    assert_true(data_set_2.get_image_height() == 3, LOG);
    assert_true(data_set_2.get_image_width() == 3, LOG);
    assert_true(data_set_2.get_channels_number() == 3, LOG);
    assert_true(data_set_2.get_target_raw_variables_number(), LOG);
}


void ImageDataSetTest::test_destructor()
{
    cout << "test_destructor\n";

    ImageDataSet* data_set = new ImageDataSet();
    delete data_set;
}


void ImageDataSetTest::run_test_case()
{
    cout << "Running image data set test case...\n";

    // Constructor and destructor

    test_constructor();
    test_destructor();

    // Preprocessing

    /*

    // Reading and writing

    test_XML();*/

    cout << "End of image data set test case.\n\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
