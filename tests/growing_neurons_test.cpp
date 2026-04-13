#include "pch.h"

#include "../opennn/training_strategy.h"
#include "../opennn/dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/growing_neurons.h"

using namespace opennn;


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
    MatrixR data(21, 2);
    data << -1.0f, 0.0f,
        -0.9f, 0.0f,
        -0.9f, 0.0f,
        -0.7f, 0.0f,
        -0.6f, 0.0f,
        -0.5f, 0.0f,
        -0.4f, 0.0f,
        -0.3f, 0.0f,
        -0.2f, 0.0f,
        -0.1f, 0.0f,
        0.0f, 0.0f,
        0.1f, 0.0f,
        0.2f, 0.0f,
        0.3f, 0.0f,
        0.4f, 0.0f,
        0.5f, 0.0f,
        0.6f, 0.0f,
        0.7f, 0.0f,
        0.8f, 0.0f,
        0.9f, 0.0f,
        1.0f, 0.0f;

    Dataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {3}, {1});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(7);
    growing_neurons.set_display(false);

    NeuronsSelectionResults neuron_selection_results = growing_neurons.perform_neurons_selection();

    EXPECT_GE(neuron_selection_results.optimal_neurons_number, 1);
}
TEST(GrowingNeuronsTest, PerformNeuronsSelection)
{
    MatrixR data(21, 2);
    data << -1.0f,  1.0f,
        -0.9f, -0.9f,
        -0.9f, -0.8f,
        -0.7f, -0.7f,
        -0.6f, -0.6f,
        -0.5f, -0.5f,
        -0.4f, -0.4f,
        -0.3f, -0.3f,
        -0.2f, -0.2f,
        -0.1f, -0.1f,
        0.0f,  0.0f,
        0.1f,  0.1f,
        0.2f,  0.2f,
        0.3f,  0.3f,
        0.4f,  0.4f,
        0.5f,  0.5f,
        0.6f,  0.6f,
        0.7f,  0.7f,
        0.8f,  0.8f,
        0.9f,  0.9f,
        1.0f,  1.0f;

    Dataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {3}, {1});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(5);
    growing_neurons.set_display(false);

    NeuronsSelectionResults results = growing_neurons.perform_neurons_selection();

    EXPECT_EQ(results.stopping_condition, GrowingNeurons::StoppingCondition::MaximumNeurons);
}


TEST(GrowingNeuronsTest, StopByTime)
{
    MatrixR data(21, 2);
    data << -1.0f,  1.0f,
        -0.9f, -0.9f,
        -0.8f, -0.8f,
        -0.7f, -0.7f,
        -0.6f, -0.6f,
        -0.5f, -0.5f,
        -0.4f, -0.4f,
        -0.3f, -0.3f,
        -0.2f, -0.2f,
        -0.1f, -0.1f,
        0.0f,  0.0f,
        0.1f,  0.1f,
        0.2f,  0.2f,
        0.3f,  0.3f,
        0.4f,  0.4f,
        0.5f,  0.5f,
        0.6f,  0.6f,
        0.7f,  0.7f,
        0.8f,  0.8f,
        0.9f,  0.9f,
        1.0f,  1.0f;

    Dataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_maximum_neurons(100);
    growing_neurons.set_maximum_time(type(0));
    growing_neurons.set_display(false);

    NeuronsSelectionResults results = growing_neurons.perform_neurons_selection();

    EXPECT_EQ(results.stopping_condition, GrowingNeurons::StoppingCondition::MaximumTime);
}


TEST(GrowingNeuronsTest, OptimalNeuronsFound)
{
    // Relación no lineal: output = input^2
    const Index samples = 40;
    MatrixR data(samples, 2);
    for(Index i = 0; i < samples; i++)
    {
        type x = type(i) / samples * 2 - 1;
        data(i, 0) = x;
        data(i, 1) = x * x;
    }

    Dataset dataset(samples, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(5);
    growing_neurons.set_display(false);

    NeuronsSelectionResults results = growing_neurons.perform_neurons_selection();

    EXPECT_GE(results.optimal_neurons_number, 1);
    EXPECT_LE(results.optimal_neurons_number, 5);
}


TEST(GrowingNeuronsTest, NeuronsIncrement)
{
    MatrixR data(21, 2);
    for(Index i = 0; i < 21; i++)
    {
        type x = type(i) / 20 * 2 - 1;
        data(i, 0) = x;
        data(i, 1) = x;
    }

    Dataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_neurons_increment(2);
    growing_neurons.set_maximum_neurons(7);
    growing_neurons.set_display(false);

    NeuronsSelectionResults results = growing_neurons.perform_neurons_selection();

    // Con incremento de 2 y minimo 1: prueba 1, 3, 5, 7
    EXPECT_EQ(results.optimal_neurons_number % 2, 1); // siempre impar
}

