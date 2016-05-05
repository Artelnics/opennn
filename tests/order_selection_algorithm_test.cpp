/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D E R   S E L E C T I O N   A L G O R I T H M   T E S T   C L A S S                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "incremental_order.h"

#include "order_selection_algorithm_test.h"


using namespace OpenNN;


// GENERAL RUCTOR

OrderSelectionAlgorithmTest::OrderSelectionAlgorithmTest(void) : UnitTesting()
{
}


// DESTRUCTOR

OrderSelectionAlgorithmTest::~OrderSelectionAlgorithmTest(void)
{
}


// METHODS

// Constructor and destructor methods

void OrderSelectionAlgorithmTest::test_constructor(void)
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    IncrementalOrder io1(&ts);

    assert_true(io1.has_training_strategy(), LOG);

    IncrementalOrder io2;

    assert_true(!io2.has_training_strategy(), LOG);
}

void OrderSelectionAlgorithmTest::test_destructor(void)
{
    message += "tes_destructor\n";

    IncrementalOrder* io = new IncrementalOrder;

    delete io;
}

// Get methods

void OrderSelectionAlgorithmTest::test_get_training_strategy_pointer(void)
{
    message += "test_get_training_strategy_pointer\n";

    TrainingStrategy ts;

    IncrementalOrder io(&ts);

    assert_true(io.get_training_strategy_pointer() != NULL, LOG);

}


void OrderSelectionAlgorithmTest::test_get_performance_calculation_method(void)
{
    message += "test_get_performance_calculation_method\n";

    IncrementalOrder io;

    io.set_performance_calculation_method(OrderSelectionAlgorithm::Minimum);

    assert_true(io.get_performance_calculation_method() == OrderSelectionAlgorithm::Minimum, LOG);

}

void OrderSelectionAlgorithmTest::test_write_performance_calculation_method(void)
{
    message += "test_write_performance_calculation_method\n";

    IncrementalOrder io;

    io.set_performance_calculation_method(OrderSelectionAlgorithm::Minimum);

    assert_true(io.write_performance_calculation_method() == "Minimum", LOG);
}

// Set methods

void OrderSelectionAlgorithmTest::test_set_training_strategy_pointer(void)
{
    message += "test_set_training_strategy_pointer\n";

    TrainingStrategy ts;

    IncrementalOrder io;

    io.set_training_strategy_pointer(&ts);

    assert_true(io.get_training_strategy_pointer() != NULL, LOG);
}

void OrderSelectionAlgorithmTest::test_set_default(void)
{
    message += "test_set_default\n";

}

void OrderSelectionAlgorithmTest::test_set_performance_calculation_method(void)
{
    message += "test_set_performance_calculation_method\n";

}

// Performances calculation methods

void OrderSelectionAlgorithmTest::test_perform_minimum_model_evaluation(void)
{
    message += "test_perform_minimum_model_evaluation\n";

}

void OrderSelectionAlgorithmTest::test_perform_maximum_model_evaluation(void)
{
    message += "test_perform_maximum_model_evaluation\n";

}

void OrderSelectionAlgorithmTest::test_perform_mean_model_evaluation(void)
{
    message += "test_perform_mean_model_evaluation\n";

}

void OrderSelectionAlgorithmTest::test_get_final_performances(void)
{
    message += "test_get_final_performances\n";

}

void OrderSelectionAlgorithmTest::test_perform_model_evaluation(void)
{
    message += "test_perform_model_evaluation\n";

}

void OrderSelectionAlgorithmTest::test_get_parameters_order(void)
{
    message += "test_get_parameters_order\n";

}

// Unit testing methods

void OrderSelectionAlgorithmTest::run_test_case(void)
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

    test_perform_minimum_model_evaluation();
    test_perform_maximum_model_evaluation();
    test_perform_mean_model_evaluation();

    test_get_final_performances();

    test_perform_model_evaluation();

    test_get_parameters_order();

    message += "End of order selection algorithm test case.\n";

}

