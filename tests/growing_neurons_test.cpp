#include "pch.h"

#include "opennn/training_strategy.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/growing_neurons.h"
#include "opennn/dense_layer.h"
#include "opennn/normalization_layer_3d.h"

using namespace opennn;

TEST(GrowingNeuronsTest, DefaultConstructor)
{
    GrowingNeurons growing_neurons;
}

// Selection hands the last trainable layer a rank-1 shape. A layer that cannot
// take one used to keep its previous shape and say nothing, so selection ran its
// whole loop against a network it never changed and reported the result as if
// the neuron count had varied. It must refuse instead.
TEST(GrowingNeuronsTest, RefusesALastLayerThatCannotTakeANeuronCount)
{
    const Index samples = 8;

    TabularDataset dataset(samples, {1}, {1});
    MatrixR data(samples, 2);
    for (Index i = 0; i < samples; i++)
    {
        const type x = type(i) / type(samples);
        data(i, 0) = x;
        data(i, 1) = x;
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{2}, "Linear"));
    neural_network.add_layer(make_unique<Normalization3d>(Shape{2, 2}));
    neural_network.compile();

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_display(false);
    growing_neurons.set_maximum_neurons(3);

    try
    {
        growing_neurons.perform_neurons_selection();
        FAIL() << "selection accepted a layer that cannot take a neuron count";
    }
    catch (const exception& error)
    {
        // Assert the reason, not merely that something threw: this test is only
        // worth anything if it is the rank guard that stopped it.
        EXPECT_NE(string(error.what()).find("does not accept a rank-1 input shape"),
                  string::npos)
            << "threw, but not for the reason under test: " << error.what();
    }
}

// The rank a layer declares has to be the rank it actually enforces, or the
// question selection asks is worthless.
TEST(GrowingNeuronsTest, DeclaredInputRanksMatchWhatLayersAccept)
{
    const opennn::Dense dense(Shape{1}, Shape{2}, "Linear");
    EXPECT_TRUE(dense.accepts_input_rank(1));
    EXPECT_TRUE(dense.accepts_input_rank(2));
    EXPECT_FALSE(dense.accepts_input_rank(3));

    const Normalization3d normalization(Shape{2, 2});
    EXPECT_FALSE(normalization.accepts_input_rank(1));
    EXPECT_TRUE(normalization.accepts_input_rank(2));
}

TEST(GrowingNeuronsTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;

    GrowingNeurons growing_neurons(&training_strategy);
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

    TabularDataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {3}, {1});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(7);
    growing_neurons.set_display(false);

    NeuronsSelectionResult neuron_selection_results = growing_neurons.perform_neurons_selection();

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

    TabularDataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {3}, {1});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(5);
    growing_neurons.set_display(false);

    NeuronsSelectionResult results = growing_neurons.perform_neurons_selection();

    ASSERT_TRUE(results.stopping_condition);
    EXPECT_EQ(*results.stopping_condition, GrowingNeurons::StoppingCondition::MaximumNeurons);
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

    TabularDataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_maximum_neurons(100);
    growing_neurons.set_maximum_time(type(0));
    growing_neurons.set_display(false);

    NeuronsSelectionResult results = growing_neurons.perform_neurons_selection();

    ASSERT_TRUE(results.stopping_condition);
    EXPECT_EQ(*results.stopping_condition, GrowingNeurons::StoppingCondition::MaximumTime);
}

TEST(GrowingNeuronsTest, OptimalNeuronsFound)
{

    const Index samples = 40;
    MatrixR data(samples, 2);
    for(Index i = 0; i < samples; i++)
    {
        type x = type(i) / samples * 2 - 1;
        data(i, 0) = x;
        data(i, 1) = x * x;
    }

    TabularDataset dataset(samples, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(5);
    growing_neurons.set_display(false);

    NeuronsSelectionResult results = growing_neurons.perform_neurons_selection();

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

    TabularDataset dataset(21, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_neurons_increment(2);
    growing_neurons.set_maximum_neurons(7);
    growing_neurons.set_display(false);

    NeuronsSelectionResult results = growing_neurons.perform_neurons_selection();

    EXPECT_EQ(results.optimal_neurons_number % 2, 1);
}

TEST(GrowingNeuronsTest, CrossValidationKeepsPersistentRoles)
{

    const Index samples = 40;

    TabularDataset dataset(samples, {1}, {1});
    MatrixR data(samples, 2);
    for (Index i = 0; i < samples; i++)
    {
        const type x = type(i) / type(samples);
        data(i, 0) = x;
        data(i, 1) = x;
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    const vector<SampleRole> roles_before = dataset.get_sample_roles();

    ApproximationNetwork neural_network({1}, {3}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingNeurons growing_neurons(&training_strategy);
    growing_neurons.set_display(false);
    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons(4);
    growing_neurons.set_folds_number(3);

    NeuronsSelectionResult results = growing_neurons.perform_neurons_selection();

    EXPECT_GE(results.optimal_neurons_number, 1);

    EXPECT_TRUE(dataset.get_sample_roles() == roles_before);
}
