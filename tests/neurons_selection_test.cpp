//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "neurons_selection_test.h"


NeuronsSelectionTest::NeuronsSelectionTest() : UnitTesting()
{
}


NeuronsSelectionTest::~NeuronsSelectionTest()
{
}


void NeuronsSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    GrowingNeurons growing_neurons_1(&training_strategy);

    assert_true(growing_neurons_1.has_training_strategy(), LOG);

    // Test

    GrowingNeurons growing_neurons_2;

    assert_true(!growing_neurons_2.has_training_strategy(), LOG);
}


void NeuronsSelectionTest::test_destructor()
{
    cout << "test_destructor\n";

    GrowingNeurons* growing_neurons = new GrowingNeurons;

    delete growing_neurons;
}


void NeuronsSelectionTest::run_test_case()
{
    cout << "Running neurons selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();

    test_destructor();

    cout << "End of neurons selection algorithm test case.\n\n";
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
