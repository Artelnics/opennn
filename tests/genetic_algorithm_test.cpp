#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/genetic_algorithm.h"

using namespace opennn;


TEST(GeneticAlgorithmTest, DefaultConstructor)
{
    GeneticAlgorithm genetic_algorithm;

    EXPECT_EQ(genetic_algorithm.has_training_strategy(), false);
}


TEST(GeneticAlgorithmTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    EXPECT_EQ(genetic_algorithm.has_training_strategy(), true);
}


TEST(GeneticAlgorithmTest, InputSelection)
{
    const Index inputs_number = 3;
    const Index samples_number = 30;

    Dataset dataset(samples_number, {inputs_number}, {1});

    // Create data where feature 0 is correlated with target, features 1-2 are noise
    MatrixR data(samples_number, inputs_number + 1);
    for(Index i = 0; i < samples_number; i++)
    {
        data(i, 0) = type(i) / type(samples_number);              // correlated with target
        data(i, 1) = type(10.0);                                   // constant noise
        data(i, 2) = type(10.0);                                   // constant noise
        data(i, 3) = type(i) / type(samples_number) + type(0.01); // target ~ feature 0
    }
    dataset.set_data(data);

    // GA requires a validation set to rank individuals
    dataset.split_samples_random(type(0.7), type(0.15), type(0.15));

    ApproximationNetwork neural_network(dataset.get_input_shape(), {2}, {1});

    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
    training_strategy.get_optimization_algorithm()->set_display(false);
    training_strategy.get_optimization_algorithm()->set_maximum_epochs(10);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(6);
    genetic_algorithm.set_maximum_epochs(3);

    InputsSelectionResults results = genetic_algorithm.perform_input_selection();

    // Should complete without crashing and produce valid results
    EXPECT_GE(results.get_epochs_number(), 1);
    EXPECT_GE(results.optimum_validation_error, type(0));
}


TEST(GeneticAlgorithmTest, RequiresValidation)
{
    const Index samples_number = 20;

    Dataset dataset(samples_number, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");  // No validation

    ApproximationNetwork neural_network({2}, {2}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_display(false);

    EXPECT_THROW(genetic_algorithm.perform_input_selection(), runtime_error);
}
