/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T   S E L E C T I O N   A L G O R I T H M   T E S T   C L A S S                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "growing_inputs.h"

#include "inputs_selection_algorithm_test.h"


using namespace OpenNN;


// GENERAL RUCTOR

InputsSelectionAlgorithmTest::InputsSelectionAlgorithmTest(void) : UnitTesting()
{
}


// DESTRUCTOR

InputsSelectionAlgorithmTest::~InputsSelectionAlgorithmTest(void)
{
}


// METHODS

// Constructor and destructor methods

void InputsSelectionAlgorithmTest::test_constructor(void)
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    GrowingInputs gi1(&ts);

    assert_true(gi1.has_training_strategy(), LOG);

    GrowingInputs gi2;

    assert_true(!gi2.has_training_strategy(), LOG);
}

void InputsSelectionAlgorithmTest::test_destructor(void)
{
    message += "tes_destructor\n";

    GrowingInputs* gi = new GrowingInputs;

    delete gi;
}

// Get methods

void InputsSelectionAlgorithmTest::test_get_training_strategy_pointer(void)
{
    message += "test_get_training_strategy_pointer\n";

    TrainingStrategy ts;

    GrowingInputs gi(&ts);

    assert_true(gi.get_training_strategy_pointer() != NULL, LOG);

}


void InputsSelectionAlgorithmTest::test_get_performance_calculation_method(void)
{
    message += "test_get_performance_calculation_method\n";

    GrowingInputs gi;

    gi.set_performance_calculation_method(InputsSelectionAlgorithm::Minimum);

    assert_true(gi.get_performance_calculation_method() == InputsSelectionAlgorithm::Minimum, LOG);

}

void InputsSelectionAlgorithmTest::test_write_performance_calculation_method(void)
{
    message += "test_write_performance_calculation_method\n";

    GrowingInputs gi;

    gi.set_performance_calculation_method(InputsSelectionAlgorithm::Minimum);

    assert_true(gi.write_performance_calculation_method() == "Minimum", LOG);
}

// Set methods

void InputsSelectionAlgorithmTest::test_set_training_strategy_pointer(void)
{
    message += "test_set_training_strategy_pointer\n";

    TrainingStrategy ts;

    GrowingInputs gi;

    gi.set_training_strategy_pointer(&ts);

    assert_true(gi.get_training_strategy_pointer() != NULL, LOG);
}

void InputsSelectionAlgorithmTest::test_set_default(void)
{
    message += "test_set_default\n";

}

void InputsSelectionAlgorithmTest::test_set_performance_calculation_method(void)
{
    message += "test_set_performance_calculation_method\n";

}

// Performances calculation methods

void InputsSelectionAlgorithmTest::test_set_neural_inputs(void)
{
    message += "test_set_neural_inputs\n";

    NeuralNetwork nn(2,6,1);

    PerformanceFunctional pf(&nn);

    TrainingStrategy ts(&pf);

    GrowingInputs gi(&ts);

    Vector<bool> inputs(2);

    inputs[0] = false;
    inputs[1] = true;

    gi.set_neural_inputs(inputs);

    assert_true(nn.get_inputs_number() == 1, LOG);

}

// Performances calculation methods

void InputsSelectionAlgorithmTest::test_perform_minimum_model_evaluation(void)
{
    message += "test_perform_minimum_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_perform_maximum_model_evaluation(void)
{
    message += "test_perform_maximum_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_perform_mean_model_evaluation(void)
{
    message += "test_perform_mean_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_get_final_performances(void)
{
    message += "test_get_final_performances\n";

}

void InputsSelectionAlgorithmTest::test_perform_model_evaluation(void)
{
    message += "test_perform_model_evaluation\n";

}

void InputsSelectionAlgorithmTest::test_get_parameters_order(void)
{
    message += "test_get_parameters_order\n";

}

// Unit testing methods

void InputsSelectionAlgorithmTest::run_test_case(void)
{
    message += "Running order selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
     test_destructor();

    // Get methods

    test_get_training_strategy_pointer();


    test_get_performance_calculation_method();

    test_write_performance_calculation_method();

    // Set methods

    test_set_training_strategy_pointer();

    test_set_default();

    test_set_performance_calculation_method();

    // Performances calculation methods

    test_set_neural_inputs();

    test_perform_minimum_model_evaluation();
    test_perform_maximum_model_evaluation();
    test_perform_mean_model_evaluation();

    test_get_final_performances();

    test_perform_model_evaluation();

    test_get_parameters_order();

    message += "End of order selection algorithm test case.\n";

}

