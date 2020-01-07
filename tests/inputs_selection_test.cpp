//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T   S E L E C T I O N   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "inputs_selection_test.h"


InputsSelectionTest::InputsSelectionTest() : UnitTesting()
{
}


InputsSelectionTest::~InputsSelectionTest()
{
}


void InputsSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    NeuralNetwork nn;
    DataSet ds;

    TrainingStrategy training_strategy(&nn, &ds);

    GrowingInputs gi1(&training_strategy);

    assert_true(gi1.has_training_strategy() == true, LOG);

    GrowingInputs gi2;

    assert_true(gi2.has_training_strategy() == false, LOG);

}

void InputsSelectionTest::test_destructor()
{
    cout << "test_destructor\n";

    GrowingInputs* gi = new GrowingInputs;

    delete gi;
}


void InputsSelectionTest::test_get_training_strategy_pointer()
{
    cout << "test_get_training_strategy_pointer\n";

    NeuralNetwork nn;
    DataSet ds;

    TrainingStrategy training_strategy(&nn, &ds);

    GrowingInputs gi(&training_strategy);

    assert_true(gi.get_training_strategy_pointer() != nullptr, LOG);
}


void InputsSelectionTest::test_set_training_strategy_pointer()
{
    cout << "test_set_training_strategy_pointer\n";

    NeuralNetwork nn;
    DataSet ds;

    TrainingStrategy training_strategy(&nn, &ds);

    GrowingInputs growing_inputs;

    growing_inputs.set_training_strategy_pointer(&training_strategy);

    assert_true(growing_inputs.get_training_strategy_pointer() != nullptr, LOG);
}

void InputsSelectionTest::test_set_default()
{
    cout << "test_set_default\n";

}

void InputsSelectionTest::test_set_loss_calculation_method()
{
    cout << "test_set_loss_calculation_method\n";

}


void InputsSelectionTest::test_get_final_loss()
{
    cout << "test_get_final_loss\n";

}

void InputsSelectionTest::test_calculate_losses()
{
    cout << "test_calculate_losses\n";

}

void InputsSelectionTest::test_get_parameters_order()
{
    cout << "test_get_parameters_order\n";

}

// Unit testing methods

void InputsSelectionTest::run_test_case()
{
    cout << "Running inputs selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Get methods

    test_get_training_strategy_pointer();

    // Set methods

    test_set_training_strategy_pointer();

    test_set_default();

    test_get_final_loss();

    test_calculate_losses();

    test_get_parameters_order();

    cout << "End of inputs selection algorithm test case.\n";

}

