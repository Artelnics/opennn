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

    TrainingStrategy training_strategy;

    IncrementalNeurons io1(&training_strategy);

    assert_true(io1.has_training_strategy() == true, LOG);

    IncrementalNeurons io2;

    assert_true(!io2.has_training_strategy() == true, LOG);
}


void NeuronsSelectionTest::test_destructor()
{
    cout << "tes_destructor\n";

    IncrementalNeurons* io = new IncrementalNeurons;

    delete io;
}


void NeuronsSelectionTest::test_get_training_strategy_pointer()
{
    cout << "test_get_training_strategy_pointer\n";

    TrainingStrategy training_strategy;

    IncrementalNeurons io(&training_strategy);

    assert_true(io.get_training_strategy_pointer() != nullptr, LOG);
}


void NeuronsSelectionTest::test_set_training_strategy_pointer()
{
    cout << "test_set_training_strategy_pointer\n";

    NeuralNetwork nn;
    DataSet ds;

    TrainingStrategy training_strategy(&nn, &ds);

    IncrementalNeurons in;

    in.set_training_strategy_pointer(&training_strategy);

    assert_true(in.get_training_strategy_pointer() != nullptr, LOG);
}

void NeuronsSelectionTest::test_set_default()
{
    cout << "test_set_default\n";

}

void NeuronsSelectionTest::test_set_loss_calculation_method()
{
    cout << "test_set_loss_calculation_method\n";

}


void NeuronsSelectionTest::test_get_final_loss()
{
    cout << "test_get_final_loss\n";

}

void NeuronsSelectionTest::test_calculate_losses()
{
    cout << "test_calculate_losses\n";

}

void NeuronsSelectionTest::test_get_parameters_order()
{
    cout << "test_get_parameters_order\n";

}

// Unit testing methods

void NeuronsSelectionTest::run_test_case()
{
    cout << "Running neurons selection algorithm test case...\n";

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

    cout << "End of neurons selection algorithm test case.\n";

}

