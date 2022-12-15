//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer_test.h"


FlattenLayerTest::FlattenLayerTest() : UnitTesting()
{
}


FlattenLayerTest::~FlattenLayerTest()
{
}


void FlattenLayerTest::test_constructor()
{
    cout << "test_constructor\n";

}

void FlattenLayerTest::test_destructor()
{
   cout << "test_destructor\n";

   FlattenLayer* flatten_layer_1 = new FlattenLayer;

   delete flatten_layer_1;

}

void FlattenLayerTest::test_calculate_flatten_outputs()
{    
    const Index image_height = 6;
    const Index image_width = 6;
    const Index image_channels_number= 3;
    const Index images_number = 2;
    const Index pixels_number = image_height * image_width * image_channels_number;

    Tensor<type, 4> inputs(image_height, image_width, image_channels_number, images_number);
    inputs.setRandom();

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions(0) = image_height;
    inputs_dimensions(1) = image_width;
    inputs_dimensions(2) = image_channels_number;
    inputs_dimensions(3) = images_number;

    Tensor<type, 2> outputs(images_number, pixels_number);

    Tensor<Index, 1> outputs_dimensions(images_number);
    outputs_dimensions(0) = images_number;
    outputs_dimensions(1) = pixels_number;

    FlattenLayer flatten_layer(inputs_dimensions);
//    flatten_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    cout << "Inputs: " << endl;
    cout << inputs << endl;

    cout << endl << endl << endl << endl;
    cout << "Inputs dimensions: " << inputs.size() << endl;
    cout << "Outputs dimensions: " << outputs.size() << endl;
    cout << endl << endl << endl << endl;

    cout << "Outputs: " << endl;
    cout << outputs << endl;

    // Test

    assert_true(inputs.size() == outputs.size(), LOG);
}

void FlattenLayerTest::run_test_case()
{
   cout << "Running flatten layer test case...\n";

   // Constructor and destructor

    test_constructor();
    test_destructor();

    // Outputs

    test_calculate_flatten_outputs();

   cout << "End of flatten layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
