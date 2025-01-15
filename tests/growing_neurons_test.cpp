#include "pch.h"

#include "../opennn/training_strategy.h"
#include "../opennn/growing_neurons.h"


TEST(GrowingNeuronsTest, DefaultConstructor)
{
    GrowingNeurons growing_neurons;

    EXPECT_EQ(growing_neurons.has_training_strategy(), false);
}


TEST(GrowingNeuronsTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;
    
    GrowingNeurons growing_neurons(&training_strategy);

    EXPECT_EQ(growing_neurons.has_training_strategy(), true);
}


TEST(GrowingNeuronsTest, NeuronsSelection)
{


    Tensor<type, 2> data(21, 2);

    data.setValues({{type(-1),type(0)},
                    {type(-0.9),type(0)},
                    {type(-0.9),type(0)},
                    {type(-0.7),type(0)},
                    {type(-0.6),type(0)},
                    {type(-0.5),type(0)},
                    {type(-0.4),type(0)},
                    {type(-0.3),type(0)},
                    {type(-0.2),type(0)},
                    {type(-0.1),type(0)},
                    {type(0),type(0)},
                    {type(0.1),type(0)},
                    {type(0.2),type(0)},
                    {type(0.3),type(0)},
                    {type(0.4),type(0)},
                    {type(0.5),type(0)},
                    {type(0.6),type(0)},
                    {type(0.7),type(0)},
                    {type(0.8),type(0)},
                    {type(0.9),type(0)},
                    {type(1),type(0)}});

    DataSet data_set(21, {1}, {1});
    
    data_set.set_data(data);

    vector<DataSet::VariableUse> uses = { DataSet::VariableUse::Input, DataSet::VariableUse::Target };
    data_set.set_raw_variable_uses(uses);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { 1 }, { 3 }, { 1 });
    neural_network.set_parameters_constant(type(0));

    TrainingStrategy training_strategy;
    GrowingNeurons growing_neurons(&training_strategy);

    NeuronsSelectionResults neuron_selection_results;

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
    training_strategy.set_display(false);

    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons_number(7);
    growing_neurons.set_selection_error_goal(type(1.0e-3f));
    growing_neurons.set_display(false);

    //EXPECT_EQ(neural_network.get_layers_neurons_numbers()[0], 1);
}

/*
void GrowingNeuronsTest::test_perform_neurons_selection()
{
    growing_neurons.set_training_strategy(&training_strategy);

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Tensor<type, 2> data;


    // Test

    samples_number = 21;
    inputs_number = 1;
    targets_number = 1;

    data.resize(samples_number,inputs_number + targets_number);

    data.setValues({{type(-1),type(1)},
                    {type(-0.9), type(-0.9)},
                    {type(-0.9),type(-0.8)},
                    {type(-0.7),type(-0.7)},
                    {type(-0.6),type(-0.6)},
                    {type(-0.5),type(-0.5)},
                    {type(-0.4), type(-0.4)},
                    {type(-0.3),type(-0.3)},
                    {type(-0.2),type(-0.2)},
                    {type(-0.1),type(-0.1)},
                    {type(0),type(0)},
                    {type(0.1),type(0.1)},
                    {type(0.2),type(0.2)},
                    {type(0.3),type(0.3)},
                    {type(0.4),type(0.4)},
                    {type(0.5),type(0.5)},
                    {type(0.6),type(0.6)},
                    {type(0.7),type(0.7)},
                    {type(0.8),type(0.8)},
                    {type(0.9),type(0.9)},
                    {type(1),type(1)}});

    data_set.set_data(data);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {3}, {targets_number});
    neural_network.set_parameters_constant(type(0));

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
    training_strategy.set_display(false);

    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons_number(5);
    growing_neurons.set_selection_error_goal(type(0));
    growing_neurons.set_maximum_selection_failures(1);
    growing_neurons.set_display(false);

    //EXPECT_EQ(neural_network.get_layers_neurons_numbers()[0] == inputs_number);

    neuron_selection_results = growing_neurons.perform_neurons_selection();

    EXPECT_EQ(neuron_selection_results.stopping_condition == NeuronsSelection::StoppingCondition::MaximumNeurons);

}

}
*/
