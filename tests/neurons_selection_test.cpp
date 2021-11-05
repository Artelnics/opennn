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

    GrowingNeurons growing_neurons_1(&training_strategy);

    assert_true(growing_neurons_1.has_training_strategy(), LOG);

    GrowingNeurons growing_neurons_2;

    assert_true(!growing_neurons_2.has_training_strategy(), LOG);
}


void NeuronsSelectionTest::test_destructor()
{
    cout << "tes_destructor\n";

    GrowingNeurons* growing_neurons_pointer = new GrowingNeurons;

    delete growing_neurons_pointer;
}


void NeuronsSelectionTest::run_test_case()
{
    cout << "Running neurons selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    cout << "End of neurons selection algorithm test case.\n\n";
}

