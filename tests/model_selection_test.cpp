//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   T E S T   C L A S S                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "model_selection_test.h"


ModelSelectionTest::ModelSelectionTest() : UnitTesting() 
{
    training_strategy.set(&neural_network, &data_set);

    model_selection.set(&training_strategy);
}


ModelSelectionTest::~ModelSelectionTest() 
{
}


void ModelSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    ModelSelection model_selection_1(&training_strategy);
    assert_true(model_selection_1.has_training_strategy(), LOG);

    ModelSelection model_selection_2;

    assert_true(!model_selection_2.has_training_strategy(), LOG);
}


void ModelSelectionTest::test_destructor()
{
    cout << "test_destructor\n";

    ModelSelection* model_selection_pointer = new ModelSelection;

    delete model_selection_pointer;
}


void ModelSelectionTest::test_get_training_strategy_pointer()
{
    cout << "test_get_training_strategy_pointer\n";

    assert_true(model_selection.get_training_strategy_pointer() != nullptr, LOG);
}


void ModelSelectionTest::test_set()
{
    cout << "test_set_training_strategy_pointer\n";

    model_selection.set(&training_strategy);

    assert_true(model_selection.get_training_strategy_pointer() != nullptr, LOG);
}


void ModelSelectionTest::test_perform_neurons_selection()
{
    cout << "test_perform_neurons_selection\n";

    data_set.generate_sum_data(20,2);

    neural_network.set(NeuralNetwork::Approximation, {1, 2, 1});

    training_strategy.set_display(false);

    model_selection.set_display(false);

    GrowingNeurons* incremental_neurons_pointer = model_selection.get_growing_neurons_pointer();

    incremental_neurons_pointer->set_maximum_selection_failures(2);

    incremental_neurons_pointer->set_display(false);

    NeuronsSelectionResults results;

    results = model_selection.perform_neurons_selection();

    assert_true(model_selection.get_inputs_selection_method() == ModelSelection::GROWING_INPUTS, LOG);
    assert_true(model_selection.get_neurons_selection_method() == ModelSelection::GROWING_NEURONS, LOG);
//    assert_true(results.growing_neurons_results_pointer->optimum_selection_error != 0.0, LOG);
//    assert_true(results.growing_neurons_results_pointer->optimal_neurons_number >= 1 , LOG);
}


void ModelSelectionTest::test_to_XML()
{
    cout << "test_to_XML\n";

    model_selection.save("../data/model_selection.xml");
}


void ModelSelectionTest::test_save()
{
    cout << "test_save\n";

    string file_name = "../data/model_selection1.xml";

    model_selection.save(file_name);
}


void ModelSelectionTest::test_load()
{
    cout << "test_load\n";

    string file_name = "../data/model_selection.xml";
    string file_name2 = "../data/model_selection2.xml";

    model_selection.set_neurons_selection_method(ModelSelection::GROWING_NEURONS);

    // Test

    model_selection.save(file_name);
    model_selection.load(file_name);
    model_selection.save(file_name2);

}


void ModelSelectionTest::run_test_case()
{
    cout << "Running model selection test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Get methods

    test_get_training_strategy_pointer();

    // Set methods

    test_set();

    // Model selection methods

    test_perform_neurons_selection();

    // Serialization methods

    test_save();
    test_load();

    cout << "End of model selection test case.\n\n";
}
