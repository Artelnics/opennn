/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T   S E L E C T I O N   A L G O R I T H M   T E S T   C L A S S                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "growing_inputs.h"

#include "inputs_selection_algorithm_test.h"


using namespace OpenNN;


// GENERAL RUCTOR

InputsSelectionAlgorithmTest::InputsSelectionAlgorithmTest() : UnitTesting()
{
}


// DESTRUCTOR

InputsSelectionAlgorithmTest::~InputsSelectionAlgorithmTest()
{
}


// METHODS

// Constructor and destructor methods

void InputsSelectionAlgorithmTest::test_constructor()
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    GrowingInputs gi1(&ts);

    assert_true(gi1.has_training_strategy(), LOG);

    GrowingInputs gi2;

    assert_true(!gi2.has_training_strategy(), LOG);
}

void InputsSelectionAlgorithmTest::test_destructor()
{
    message += "tes_destructor\n";

    GrowingInputs* gi = new GrowingInputs;

    delete gi;
}

// Get methods

void InputsSelectionAlgorithmTest::test_get_training_strategy_pointer()
{
    message += "test_get_training_strategy_pointer\n";

    TrainingStrategy ts;

    GrowingInputs gi(&ts);

    assert_true(gi.get_training_strategy_pointer() != nullptr, LOG);

}


void InputsSelectionAlgorithmTest::test_get_loss_calculation_method()
{
    message += "test_get_loss_calculation_method\n";

    GrowingInputs gi;

    gi.set_loss_calculation_method(InputsSelectionAlgorithm::Minimum);

    assert_true(gi.get_loss_calculation_method() == InputsSelectionAlgorithm::Minimum, LOG);

}

void InputsSelectionAlgorithmTest::test_write_loss_calculation_method()
{
    message += "test_write_loss_calculation_method\n";

    GrowingInputs gi;

    gi.set_loss_calculation_method(InputsSelectionAlgorithm::Minimum);

    assert_true(gi.write_loss_calculation_method() == "Minimum", LOG);
}

// Set methods

void InputsSelectionAlgorithmTest::test_set_training_strategy_pointer()
{
    message += "test_set_training_strategy_pointer\n";

    TrainingStrategy ts;

    GrowingInputs gi;

    gi.set_training_strategy_pointer(&ts);

    assert_true(gi.get_training_strategy_pointer() != nullptr, LOG);
}

void InputsSelectionAlgorithmTest::test_set_default()
{
    message += "test_set_default\n";

}

void InputsSelectionAlgorithmTest::test_set_loss_calculation_method()
{
    message += "test_set_loss_calculation_method\n";

}

// Performances calculation methods

/// @todo

void InputsSelectionAlgorithmTest::test_set_neural_inputs()
{
    message += "test_set_neural_inputs\n";

//    DataSet ds;

//    NeuralNetwork nn(2,6,1);

//    SumSquaredError sse(&nn, &ds);

//    TrainingStrategy ts(&sse);

//    GrowingInputs gi(&ts);

//    Vector<bool> inputs(2);

//    inputs[0] = false;
//    inputs[1] = true;

//    gi.set_neural_inputs(inputs);

//    assert_true(nn.get_inputs_number() == 1, LOG);
}

// Performances calculation methods

void InputsSelectionAlgorithmTest::test_perform_minimum_model_evaluation()
{
    message += "test_perform_minimum_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_perform_maximum_model_evaluation()
{
    message += "test_perform_maximum_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_perform_mean_model_evaluation()
{
    message += "test_perform_mean_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_get_final_losss()
{
    message += "test_get_final_losss\n";

}

void InputsSelectionAlgorithmTest::test_perform_model_evaluation()
{
    message += "test_perform_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_get_parameters_order()
{
    message += "test_get_parameters_order\n";

}

// Unit testing methods

void InputsSelectionAlgorithmTest::run_test_case()
{
    message += "Running order selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
     test_destructor();

    // Get methods

    test_get_training_strategy_pointer();


    test_get_loss_calculation_method();

    test_write_loss_calculation_method();

    // Set methods

    test_set_training_strategy_pointer();

    test_set_default();

    test_set_loss_calculation_method();

    // Performances calculation methods

    test_set_neural_inputs();

    test_perform_minimum_model_evaluation();
    test_perform_maximum_model_evaluation();
    test_perform_mean_model_evaluation();

    test_get_final_losss();

    test_perform_model_evaluation();

    test_get_parameters_order();

    message += "End of order selection algorithm test case.\n";

}

