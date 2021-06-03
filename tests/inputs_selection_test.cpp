//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "inputs_selection_test.h"


InputsSelectionTest::InputsSelectionTest() : UnitTesting()
{
    training_strategy.set(&neural_network, &data_set);

    growing_inputs.set(&training_strategy);
}


InputsSelectionTest::~InputsSelectionTest()
{
}


void InputsSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    GrowingInputs gi1(&training_strategy);

    assert_true(gi1.has_training_strategy(), LOG);

    GrowingInputs gi2;

    assert_true(!gi2.has_training_strategy(), LOG);
}


void InputsSelectionTest::test_destructor()
{
    cout << "tes_destructor\n";

    GrowingInputs* growing_inputs_pointer = new GrowingInputs;

    delete growing_inputs_pointer;
}


void InputsSelectionTest::test_get_training_strategy_pointer()
{
    cout << "test_get_training_strategy_pointer\n";

    GrowingInputs growing_inputs(&training_strategy);

    assert_true(growing_inputs.get_training_strategy_pointer() != nullptr, LOG);
}


void InputsSelectionTest::test_set()
{
    cout << "test_set\n";

    growing_inputs.set(&training_strategy);

    assert_true(growing_inputs.get_training_strategy_pointer() != nullptr, LOG);
}


void InputsSelectionTest::run_test_case()
{
    cout << "Running inputs selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Get methods

    test_get_training_strategy_pointer();

    // Set methods

    test_set();

    cout << "End of inputs selection algorithm test case.\n\n";
}

