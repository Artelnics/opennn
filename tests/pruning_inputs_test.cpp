//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R U N I N G   I N P U T S   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "pruning_inputs_test.h"


PruningInputsTest::PruningInputsTest() : UnitTesting()
{
//    training_strategy.set(&neural_network, &data_set);

//    pruning_inputs.set(&training_strategy);

    training_strategy.set_display(false);

    pruning_inputs.set_display(false);
}


PruningInputsTest::~PruningInputsTest()
{
}


void PruningInputsTest::test_constructor()
{
    cout << "test_constructor\n";

    PruningInputs pruning_inputs_1(&training_strategy);

    assert_true(pruning_inputs_1.has_training_strategy(), LOG);

    PruningInputs pruning_inputs_2;

    assert_true(!pruning_inputs_2.has_training_strategy(), LOG);
}


void PruningInputsTest::test_destructor()
{
    cout << "test_destructor\n";

    PruningInputs* pruning_inputs_pointer = new PruningInputs;

    delete pruning_inputs_pointer;
}


void PruningInputsTest::test_perform_inputs_selection()
{
    cout << "test_perform_inputs_selection\n";

    InputsSelectionResults inputs_selection_results;

    // Test

    data_set.generate_Rosenbrock_data(40,3);

    data_set.split_samples_random();

    neural_network.set(NeuralNetwork::Approximation, {2,6,1});

    inputs_selection_results = pruning_inputs.perform_inputs_selection();

//    assert_true(inputs_selection_results.optimal_inputs_indices[0] == 0, LOG);

    // Test

    data_set.generate_sum_data(40,3);

    neural_network.set(NeuralNetwork::Approximation, {2,6,1});

    inputs_selection_results = pruning_inputs.perform_inputs_selection();

//    assert_true(pir->optimal_inputs_indices[0] == 0, LOG);
}


void PruningInputsTest::run_test_case()
{
    cout << "Running pruning input test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Input selection methods

    test_perform_inputs_selection();

    cout << "End of pruning input test case.\n\n";
}
