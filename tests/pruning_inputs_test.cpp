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
}


PruningInputsTest::~PruningInputsTest()
{
}


void PruningInputsTest::test_constructor()
{
    cout << "test_constructor\n";

    PruningInputs pi1(&training_strategy);

    assert_true(pi1.has_training_strategy(), LOG);

    PruningInputs pi2;

    assert_true(!pi2.has_training_strategy(), LOG);
}


void PruningInputsTest::test_destructor()
{
    cout << "test_destructor\n";

    PruningInputs* pi = new PruningInputs;

    delete pi;
}


void PruningInputsTest::test_perform_inputs_selection()
{
    cout << "test_perform_inputs_selection\n";

//    DataSet data_set;

//    Tensor<type, 2> data;

//    NeuralNetwork neural_network;

//    

//    SumSquaredError sum_squared_error(&neural_network, &data_set);

//    InputsSelectionResults* pir;

    // Test

//    data_set.generate_inputs_selection_data(40,3);

//    data_set.split_samples_random();

//    neural_network.set(NeuralNetwork::Approximation, {2,6,1});

//    TrainingStrategy training_strategy(&neural_network, &data_set);

//    PruningInputs pi(&ts);

//    ts.set_display(false);

//    pi.set_display(false);

//    pi.set_approximation(true);

//    pir = pi.perform_inputs_selection();

//    assert_true(pir->optimal_inputs_indices[0] == 0, LOG);

//    pi.delete_selection_history();
//    pi.delete_parameters_history();
//    pi.delete_loss_history();

    // Test

//    data_set.generate_sum_data(40,3);

//    neural_network.set(NeuralNetwork::Approximation, {2,6,1});

//    ts.set_display(false);

//    pi.set_display(false);

//    pi.set_approximation(false);

//    pir = pi.perform_inputs_selection();

//    assert_true(pir->optimal_inputs_indices[0] == 0, LOG);

//    pi.delete_selection_history();
//    pi.delete_parameters_history();
//    pi.delete_loss_history();
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
