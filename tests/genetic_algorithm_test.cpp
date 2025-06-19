#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/training_strategy.h"
#include "../opennn/genetic_algorithm.h"

using namespace opennn;

TEST(GeneticAlgorithmTest, DefaultConstructor)
{
    GeneticAlgorithm genetic_algorithm;

    EXPECT_EQ(genetic_algorithm.has_training_strategy(), false);
    EXPECT_EQ(genetic_algorithm.get_population().dimension(0), 0);
}


TEST(GeneticAlgorithmTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;
    
    GeneticAlgorithm genetic_algorithm(&training_strategy);

    EXPECT_EQ(genetic_algorithm.has_training_strategy(), true);
}


TEST(GeneticAlgorithmTest, InitializePopulation)
{
    Index individuals_number = 8;
    Index inputs_number = 3;
    Index targets_number = 1;
    Index neurons_number = 1;
    Index samples_number = 10;

    Dataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_display(false);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    Tensor <bool, 2> population;
    Tensor <bool, 1> individual;
    Tensor <bool, 1> gene;

    genetic_algorithm.set_individuals_number(individuals_number);
    genetic_algorithm.initialize_population();

    population = genetic_algorithm.get_population();
    gene = population.chip(0, 1);
    individual = population.chip(1, 0);

    EXPECT_EQ(population.dimension(0), individuals_number);
    EXPECT_EQ(population.dimension(1), inputs_number);
    EXPECT_EQ(gene.size(), individuals_number);
    EXPECT_EQ(individual.size(), inputs_number);
}


TEST(GeneticAlgorithmTest, FitnessAssignment)
{
    Index samples_number = 10;
    Index inputs_number = 3;
    Index targets_number = 1;
    Index neurons_number = 2;
    Index individuals_number = 4;

    Dataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_display(false);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_individuals_number(individuals_number);
    genetic_algorithm.perform_fitness_assignment();

    Tensor <type, 1> fitness = genetic_algorithm.get_fitness();

    //EXPECT_EQ(maximal_index(fitness), 3);
    //EXPECT_EQ(minimal_index(fitness), 0);
}


TEST(GeneticAlgorithmTest, Selection)
{
    Index samples_number = 10;
    Index inputs_number = 3;
    Index targets_number = 1;
    Index neurons_number = 1;
    Index individuals_number = 4;

    Dataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_display(false);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    genetic_algorithm.set_individuals_number(individuals_number);

    genetic_algorithm.perform_fitness_assignment();
    Tensor <type, 1> fitness = genetic_algorithm.get_fitness();

    genetic_algorithm.initialize_population();

    Tensor <bool, 2> population = genetic_algorithm.get_population();

    genetic_algorithm.perform_selection();
    Tensor <bool, 1> selection = genetic_algorithm.get_selection();

    genetic_algorithm.set_elitism_size(0);

    EXPECT_EQ(selection(0), true);
    EXPECT_EQ(selection(1), true);
    EXPECT_EQ(selection(2), true);
    EXPECT_EQ(selection(3), true);

    EXPECT_GE(count(selection.data(), selection.data() + selection.size(), 1), 4);
    EXPECT_GE(count(selection.data() + 1, selection.data() + selection.size(), 1), 3);

    //Elitism = 1

    genetic_algorithm.set_elitism_size(1);

    EXPECT_EQ(selection(0), true);
    EXPECT_EQ(selection(1), true);
    EXPECT_EQ(selection(2), true);
    EXPECT_EQ(selection(3), true);

    EXPECT_GE(count(selection.data(), selection.data() + selection.size(), 1), 4);
    EXPECT_GE(count(selection.data() + 1, selection.data() + selection.size(), 1), 3);
}


TEST(GeneticAlgorithmTest, Crossover)
{
    Index samples_number = 10;
    Index inputs_number = 5;
    Index targets_number = 1;
    Index neurons_number = 1;
    Index individuals_number = 4;

    Dataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_display(false);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    genetic_algorithm.set_individuals_number(individuals_number);

    Tensor <bool, 2> population(4, 4);
    population.setValues({{true, false, false, false},
                          {true, false, false, true},
                          {true, false, true, false},
                          {true, false, true, true}});

    genetic_algorithm.set_population(population);

    genetic_algorithm.perform_fitness_assignment();

    genetic_algorithm.perform_selection();

    genetic_algorithm.perform_crossover();

    Tensor <bool, 2> crossover_population = genetic_algorithm.get_population();

    for (Index i = 0; i < individuals_number; i++)
    {
        EXPECT_EQ(crossover_population(i, 0), true);
        EXPECT_EQ(crossover_population(i, 1), true);
        EXPECT_EQ(crossover_population(i, 2), false);
    }
}


TEST(GeneticAlgorithmTest, Mutation)
{
    Index samples_number = 10;
    Index inputs_number = 5;
    Index targets_number = 1;
    Index neurons_number = 1;
    Index individuals_number = 4;

    Dataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_display(false);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    //Test no mutation

    genetic_algorithm.set_individuals_number(individuals_number);

    Tensor <bool, 2> population(4, 4);
    population.setValues({{true, false, true, false},
                          {false, true, true, false},
                          {true, false, false, true},
                          {false, true, false, true}});

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_mutation_rate(0);

    genetic_algorithm.perform_mutation();

    Tensor <bool, 2> mutated_population = genetic_algorithm.get_population();

    EXPECT_EQ(are_equal(mutated_population, population), true);

    //Test mutation

    individuals_number = 10;

    genetic_algorithm.set_individuals_number(individuals_number);

    population.resize(10, 10);
    population.setRandom();

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_mutation_rate(0.5);

    genetic_algorithm.perform_mutation();

    mutated_population = genetic_algorithm.get_population();

    Index mutated_genes = 0;
    Tensor <bool, 1> individual;
    Tensor <bool, 1> mutated_individual;

    for (Index i = 0; i < population.dimension(0); i++)
    {
        individual = population.chip(i, 0);

        mutated_individual = mutated_population.chip(i, 0);

        for (Index j = 0; j < individuals_number; j++)
        {
            if (individual(j) != mutated_individual(j))
            {
                mutated_genes++;
            }
        }
    }

    //EXPECT_GE(mutated_genes, 25);
}

/*
TEST(GeneticAlgorithmTest, InputSelection)
{
    Tensor<type, 2> data;

    InputsSelectionResults input_selection_results;

    // Test 1

    data.resize(20,4);

    for(Index i = 0; i < 20; i++)
    {
        data(i,0) = type(i);
        data(i,1) = type(10.0);
        data(i,2) = type(10.0);
        data(i,3) = type(i);
    }

    dataset.set_data(data);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2}, {6}, {1});

    genetic_algorithm.set_display(false);

    genetic_algorithm.set_individuals_number(6);

    genetic_algorithm.set_selection_error_goal(1);

    input_selection_results = genetic_algorithm.perform_input_selection();

    EXPECT_EQ(input_selection_results.stopping_condition == InputsSelection::StoppingCondition::SelectionErrorGoal);
    EXPECT_EQ(input_selection_results.selection_error_history(0) <= 1);


    // Test 2

    Index j = -10;

    for(Index i = 0; i < 10; i++)
    {
        data(i,0) = type(j);
        data(i,1) = type(rand());
        data(i,2) = type(1);
        j++;
    }

    for(Index i = 10; i < 20; i++)
    {
        data(i,0) = type(i);
        data(i,1) = type(rand());
        data(i,2) = type(0);
    }

    dataset.set_data(data);
    dataset.set_default_raw_variables_uses();

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(4);
    genetic_algorithm.set_selection_error_goal(type(0));
    genetic_algorithm.set_maximum_epochs_number(1);

    input_selection_results = genetic_algorithm.perform_input_selection();

    EXPECT_EQ(genetic_algorithm.get_maximum_iterations_number() == 1);
    EXPECT_EQ(genetic_algorithm.get_selection_error_goal() < 1);

    // Test 3

    data.resize(10,6);

    for(Index i = 0; i < 10; i++)
    {
        data(i,0) = type(rand());
        data(i,1) = type(rand());
        data(i,2) = type(rand());
        data(i,3) = type(rand());
        data(i,4) = type(10);
        data(i,5) = data(i,0) + data(i,1) + data(i,2) + data(i,3);
    }

    dataset.set_data(data);
    dataset.set_default_raw_variables_uses();

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(4);
    genetic_algorithm.set_selection_error_goal(type(0.01));
    genetic_algorithm.set_maximum_epochs_number(10);

    input_selection_results = genetic_algorithm.perform_input_selection();

    EXPECT_EQ(input_selection_results.get_epochs_number() <= 100);

}
*/
