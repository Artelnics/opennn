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
}


ModelSelectionTest::~ModelSelectionTest() 
{
}


void ModelSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    TrainingStrategy training_strategy;

    ModelSelection ms1(&training_strategy);
    assert_true(ms1.has_training_strategy(), LOG);

    ModelSelection ms2;

    assert_true(!ms2.has_training_strategy(), LOG);
}


void ModelSelectionTest::test_destructor()
{
    cout << "test_destructor\n";

    ModelSelection* ms = new ModelSelection;

    delete ms;
}


void ModelSelectionTest::test_get_training_strategy_pointer()
{
    cout << "test_get_training_strategy_pointer\n";

    TrainingStrategy training_strategy;

    ModelSelection ms(&training_strategy);

    assert_true(ms.get_training_strategy_pointer() != nullptr, LOG);
}


void ModelSelectionTest::test_set_training_strategy_pointer()
{
    cout << "test_set_training_strategy_pointer\n";

    NeuralNetwork nn;
    DataSet ds;

    TrainingStrategy training_strategy(&nn, &ds);

    ModelSelection ms;

    ms.set_training_strategy_pointer(&training_strategy);

    assert_true(ms.get_training_strategy_pointer() != nullptr, LOG);
}


void ModelSelectionTest::test_set_default()
{
    cout << "test_set_default\n";
}


void ModelSelectionTest::test_perform_neurons_selection()
{
    cout << "test_perform_neurons_selection\n";

    DataSet ds;

    ds.generate_sum_data(20,2);

    NeuralNetwork nn(NeuralNetwork::Approximation, {1, 2, 1});

    TrainingStrategy ts(&nn, &ds);

    ts.set_display(false);

    ModelSelection model_selection(&ts);

    model_selection.set_display(false);

    IncrementalNeurons* incremental_neurons_pointer = model_selection.get_incremental_neurons_pointer();

    incremental_neurons_pointer->set_maximum_selection_failures(2);

    incremental_neurons_pointer->set_display(false);

    ModelSelection::Results results;

    results = model_selection.perform_neurons_selection();

    assert_true(model_selection.get_inputs_selection_method() == ModelSelection::GROWING_INPUTS, LOG);
    assert_true(model_selection.get_neurons_selection_method() == ModelSelection::INCREMENTAL_NEURONS, LOG);
    assert_true(results.incremental_neurons_results_pointer->final_training_loss != 0.0, LOG);
    assert_true(results.incremental_neurons_results_pointer->final_selection_error != 0.0, LOG);
    assert_true(results.incremental_neurons_results_pointer->optimal_neurons_number >= 1 , LOG);

}


void ModelSelectionTest::test_to_XML()   
{
    cout << "test_to_XML\n";

    ModelSelection ms;

    tinyxml2::XMLDocument* document = ms.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;
}


/// @todo

void ModelSelectionTest::test_from_XML()
{
    cout << "test_from_XML\n";
/*
    ModelSelection ms1;
    ModelSelection ms2;

    ms1.set_neurons_selection_method(ModelSelection::INCREMENTAL_NEURONS);

    tinyxml2::XMLDocument* document = ms1.to_XML();

    ms2.from_XML(*document);

    delete document;

    assert_true(ms2.get_neurons_selection_method() == ModelSelection::INCREMENTAL_NEURONS, LOG);
*/
}


void ModelSelectionTest::test_save()
{
    cout << "test_save\n";

    string file_name = "../data/model_selection.xml";

    ModelSelection ms;

    ms.save(file_name);
}


/// @todo

void ModelSelectionTest::test_load()
{
    cout << "test_load\n";

    string file_name = "../data/model_selection.xml";

    ModelSelection ms;
/*
    ms.set_neurons_selection_method(ModelSelection::INCREMENTAL_NEURONS);

    // Test

    ms.save(file_name);
    ms.load(file_name);
*/
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

    test_set_training_strategy_pointer();

    test_set_default();

    // Model selection methods

    test_perform_neurons_selection();

    // Serialization methods

    test_to_XML();
    test_from_XML();
    test_save();
    test_load();

    cout << "End of model selection test case.\n";
}
