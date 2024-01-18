//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   T E S T   C L A S S                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "unscaling_layer_test.h"


UnscalingLayerTest::UnscalingLayerTest() : UnitTesting()
{
}


UnscalingLayerTest::~UnscalingLayerTest()
{
}


void UnscalingLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    UnscalingLayer unscaling_layer_1;

    assert_true(unscaling_layer_1.get_type() == Layer::Type::Unscaling, LOG);
    assert_true(unscaling_layer_1.get_descriptives().size() == 0, LOG);

    // Test

    UnscalingLayer unscaling_layer_2(3);

    assert_true(unscaling_layer_2.get_descriptives().size() == 3, LOG);

    // Test

    descriptives.resize(2);

    UnscalingLayer unscaling_layer_3(descriptives);

    assert_true(unscaling_layer_3.get_descriptives().size() == 2, LOG);
}


void UnscalingLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    UnscalingLayer* unscaling_layer = new UnscalingLayer;
    delete unscaling_layer;
}


void UnscalingLayerTest::run_test_case()
{
    cout << "Running unscaling layer test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    cout << "End of unscaling layer test case.\n\n";
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
