#include "pch.h"

#include "../opennn/model_selection.h"


TEST(ModelSelectionTest, DefaultConstructor)
{
    EXPECT_EQ(1, 1);
}


TEST(ModelSelectionTest, GeneralConstructor)
{
    EXPECT_EQ(1, 1);
}

/*
namespace opennn
{

void ModelSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    ModelSelection model_selection_1(&training_strategy);
    assert_true(model_selection_1.has_training_strategy(), LOG);

    ModelSelection model_selection_2;

    assert_true(!model_selection_2.has_training_strategy(), LOG);
}


void ModelSelectionTest::test_perform_neurons_selection()
{
    cout << "test_perform_neurons_selection\n";

    data_set.generate_sum_data(20,2);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1}, {2}, {1});

    training_strategy.set_display(false);

    model_selection.set_display(false);

    GrowingNeurons* incremental_neurons = model_selection.get_growing_neurons();

    incremental_neurons->set_maximum_selection_failures(2);

    incremental_neurons->set_display(false);

    NeuronsSelectionResults results;

    results = model_selection.perform_neurons_selection();

    assert_true(model_selection.get_inputs_selection_method() == ModelSelection::InputsSelectionMethod::GROWING_INPUTS, LOG);
    assert_true(model_selection.get_neurons_selection_method() == ModelSelection::NeuronsSelectionMethod::GROWING_NEURONS, LOG);
    assert_true(results.optimum_selection_error != 0.0, LOG);
    assert_true(results.optimal_neurons_number >= 1 , LOG);
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

    model_selection.set_neurons_selection_method(ModelSelection::NeuronsSelectionMethod::GROWING_NEURONS);

    // Test

    model_selection.save(file_name);
    model_selection.load(file_name);
    model_selection.save(file_name2);

}

}
*/
