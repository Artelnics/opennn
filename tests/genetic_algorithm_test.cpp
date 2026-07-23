#include "pch.h"

#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/genetic_algorithm.h"
#include "opennn/random_utilities.h"

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
    set_seed(0);

    const Index inputs_number = 3;
    const Index samples_number = 30;

    TabularDataset dataset(samples_number, {inputs_number}, {1});

    MatrixR data(samples_number, inputs_number + 1);
    for(Index i = 0; i < samples_number; i++)
    {
        data(i, 0) = type(i) / type(samples_number);
        data(i, 1) = type(10.0);
        data(i, 2) = type(10.0);
        data(i, 3) = type(i) / type(samples_number) + type(0.01);
    }
    dataset.set_data(data);

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

    InputsSelectionResult results = genetic_algorithm.perform_input_selection();

    EXPECT_GE(results.get_epochs_number(), 1);
    EXPECT_GE(results.optimum_validation_error, type(0));

    ASSERT_EQ(results.optimal_inputs.size(), inputs_number);
    EXPECT_TRUE(results.optimal_inputs(0));
}


TEST(GeneticAlgorithmTest, SelectsParsimoniousSubset)
{
    set_seed(0);

    const Index inputs_number = 40;
    const Index samples_number = 200;

    TabularDataset dataset(samples_number, {inputs_number}, {1});

    MatrixR data(samples_number, inputs_number + 1);
    for (Index i = 0; i < samples_number; i++)
    {
        const type signal = type(i) / type(samples_number);
        for (Index j = 0; j < inputs_number; j++)
        {
            const unsigned h = (unsigned(i) * 2654435761u) ^ (unsigned(j + 1) * 40503u);
            data(i, j) = type(h % 1000u) / type(1000);
        }
        data(i, 0) = signal;
        data(i, inputs_number) = signal;
    }
    dataset.set_data(data);
    dataset.split_samples_random(type(0.7), type(0.15), type(0.15));

    ApproximationNetwork neural_network(dataset.get_input_shape(), {2}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
    training_strategy.get_optimization_algorithm()->set_display(false);
    training_strategy.get_optimization_algorithm()->set_maximum_epochs(10);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(20);
    genetic_algorithm.set_maximum_epochs(5);

    const InputsSelectionResult results = genetic_algorithm.perform_input_selection();

    const Index selected_count = results.optimal_inputs.count();

    EXPECT_LT(selected_count, inputs_number);
    EXPECT_LT(selected_count, Index(30));
    EXPECT_TRUE(results.optimal_inputs(0));
}


TEST(GeneticAlgorithmTest, CrossValidationKeepsPersistentRoles)
{
    set_seed(0);

    const Index inputs_number = 3;
    const Index samples_number = 40;

    TabularDataset dataset(samples_number, {inputs_number}, {1});
    MatrixR data(samples_number, inputs_number + 1);
    for (Index i = 0; i < samples_number; i++)
    {
        data(i, 0) = type(i) / type(samples_number);
        data(i, 1) = type(10.0);
        data(i, 2) = type(10.0);
        data(i, 3) = type(i) / type(samples_number) + type(0.01);
    }
    dataset.set_data(data);
    dataset.split_samples_random(type(0.7), type(0.15), type(0.15));

    const vector<SampleRole> roles_before = dataset.get_sample_roles();

    ApproximationNetwork neural_network(dataset.get_input_shape(), {2}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
    training_strategy.get_optimization_algorithm()->set_display(false);
    training_strategy.get_optimization_algorithm()->set_maximum_epochs(10);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(6);
    genetic_algorithm.set_maximum_epochs(3);
    genetic_algorithm.set_folds_number(3);

    const InputsSelectionResult results = genetic_algorithm.perform_input_selection();

    EXPECT_GE(results.get_epochs_number(), 1);
    EXPECT_GE(results.optimum_validation_error, type(0));
    EXPECT_TRUE(results.optimal_inputs(0));

    EXPECT_TRUE(dataset.get_sample_roles() == roles_before);
}


TEST(GeneticAlgorithmTest, RequiresValidation)
{
    const Index samples_number = 20;

    TabularDataset dataset(samples_number, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({2}, {2}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_display(false);

    EXPECT_THROW(genetic_algorithm.perform_input_selection(), runtime_error);
}


TEST(GeneticAlgorithmTest, CrossValidationDoesNotRequirePersistentValidation)
{
    set_seed(0);

    const Index samples_number = 40;

    TabularDataset dataset(samples_number, {3}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network(dataset.get_input_shape(), {2}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
    training_strategy.get_optimization_algorithm()->set_display(false);
    training_strategy.get_optimization_algorithm()->set_maximum_epochs(10);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(6);
    genetic_algorithm.set_maximum_epochs(3);
    genetic_algorithm.set_folds_number(3);

    const vector<SampleRole> roles_before = dataset.get_sample_roles();

    InputsSelectionResult results;
    EXPECT_NO_THROW(results = genetic_algorithm.perform_input_selection());
    EXPECT_GE(results.get_epochs_number(), 1);

    EXPECT_TRUE(dataset.get_sample_roles() == roles_before);
}
