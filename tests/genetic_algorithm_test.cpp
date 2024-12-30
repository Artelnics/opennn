#include "pch.h"

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


TEST(GeneticAlgorithmTest, InitializePopulation)
{
    DataSet data_set(1, { 1 }, { 1 });

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { 1 }, { 1 }, { 1 });

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

//    genetic_algorithm.set_individuals_number(10);
//    genetic_algorithm.initialize_population();

    const Tensor<bool, 2>& population = genetic_algorithm.get_population();
//    const Tensor<bool, 1> gene = population.chip(0, 1);
//    const Tensor<bool, 1> individual = population.chip(1, 0);

//    EXPECT_EQ(population.dimension(0), individuals_number);
//    EXPECT_EQ(population.dimension(1), inputs_number);
//    EXPECT_EQ(gene.size(), individuals_number);
//    EXPECT_EQ(individual.size(), inputs_number);
}


/*
void GeneticAlgorithmTest::test_initialize_population()
{
    Tensor<bool, 2> population;
    Tensor<bool, 1> individual;
    Tensor<bool, 1> gene;

    Index individuals_number = 8;

    // Test

    Index inputs_number = 3;
    Index outputs_number = 1;
    Index samples_number = 10;
    Index hidden_neurons_number = 1;

    Tensor<type,2> data(samples_number, inputs_number + outputs_number);
    data.setRandom();
    data_set.set_data(data);

    genetic_algorithm.set_individuals_number(individuals_number);

    genetic_algorithm.initialize_population();

    population = genetic_algorithm.get_population();
    gene = population.chip(0,1);
    individual = population.chip(1,0);

    EXPECT_EQ(population.dimension(0), individuals_number);
    EXPECT_EQ(population.dimension(1), inputs_number);
    EXPECT_EQ(gene.size(), individuals_number);
    EXPECT_EQ(individual.size(), inputs_number);

    // Test

    inputs_number = 10;
    outputs_number = 3;
    samples_number = 15;
    hidden_neurons_number = 1;

    data.resize(samples_number, inputs_number + outputs_number);
    data.setRandom();
    data_set.set_data(data);

    Tensor<Index, 1> input_variables_indices(inputs_number);
    Tensor<Index, 1> target_variables_indices(outputs_number);

    input_variables_indices.setValues({0,1,2,3,4,5,6,7,8,9});
    target_variables_indices.setValues({10,11,12});

    data_set.set_input_target_raw_variables_indices(input_variables_indices,target_variables_indices);

    genetic_algorithm.set_individuals_number(individuals_number);

    genetic_algorithm.initialize_population();

    population = genetic_algorithm.get_population();
    gene = population.chip(0,1);
    individual = population.chip(1,0);

    EXPECT_EQ(population.dimension(0), individuals_number);
    EXPECT_EQ(population.dimension(1), inputs_number);

    EXPECT_EQ(gene.size(), individuals_number);
    EXPECT_EQ(individual.size(), inputs_number);
}
*/


TEST(GeneticAlgorithmTest, FitnessAssignment)
{
/*
    DataSet data_set;

    Tensor<type, 1> selection_errors;

    Tensor<type, 1> fitness;

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {3}, {2}, {1});

    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm;

    genetic_algorithm.set_individuals_number(4);

    selection_errors.resize(4);
    selection_errors.setValues({type(1), type(2), type(3), type(4)});

//    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.perform_fitness_assignment();

    fitness = genetic_algorithm.get_fitness();

    EXPECT_EQ(maximal_index(fitness), 3);
    EXPECT_EQ(minimal_index(fitness), 0);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {3}, {2}, {1});

    genetic_algorithm.set_individuals_number(4);

    selection_errors.resize(4);

    selection_errors(0) = type(4);
    selection_errors(1) = type(3);
    selection_errors(2) = type(2);
    selection_errors(3) = type(1);

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.perform_fitness_assignment();

    fitness = genetic_algorithm.get_fitness();

    EXPECT_EQ(maximal_index(fitness), 0);
    EXPECT_EQ(minimal_index(fitness), 3);
*/
}


TEST(GeneticAlgorithmTest, Selection)
{
/*
    Tensor<bool, 2> population;

    Tensor<bool, 1> selection;

    Tensor<type, 1> selection_errors;

    Tensor<type, 1> fitness;

    // Test 1

    genetic_algorithm.set_individuals_number(4);

    fitness.resize(4);
    fitness.setValues({type(1), type(2), type(3), type(4)});

    genetic_algorithm.set_fitness(fitness);

    selection_errors.resize(4);
    selection_errors.setValues({type(0.4), type(0.3), type(0.2), type(0.1)});

    genetic_algorithm.initialize_population();

    genetic_algorithm.set_selection_errors(selection_errors);

    population = genetic_algorithm.get_population();

    genetic_algorithm.perform_selection();

    selection = genetic_algorithm.get_selection();

    genetic_algorithm.set_elitism_size(0);

    EXPECT_EQ(selection(0) == 0 || selection(0) == 1,LOG);
    EXPECT_EQ(selection(1) == 0 || selection(1) == 1,LOG);
    EXPECT_EQ(selection(2) == 0 || selection(2) == 1,LOG);
    EXPECT_EQ(selection(3) == 0 || selection(3) == 1,LOG);

    EXPECT_EQ( count(selection.data(), selection.data() + selection.size(), 1)  == 2,LOG);

    EXPECT_EQ( count(selection.data() + 1, selection.data() + selection.size(), 1)  >= 1,LOG);

    // 4 individuals with elitism size = 1

    genetic_algorithm.set_individuals_number(4);

    fitness.resize(4);
    fitness.setValues({type(1), type(2), type(3), type(4)});

    genetic_algorithm.set_fitness(fitness);

    selection_errors.resize(4);
    selection_errors.setValues({type(0.4), type(0.3), type(0.2), type(0.1)});

    genetic_algorithm.initialize_population();

    genetic_algorithm.set_selection_errors(selection_errors);

    population = genetic_algorithm.get_population();

    genetic_algorithm.set_elitism_size(1);

    genetic_algorithm.perform_selection();

    EXPECT_EQ(selection(0) == 1,LOG);
    EXPECT_EQ(selection(1) == 0 || selection(1) == 1,LOG);
    EXPECT_EQ(selection(2) == 0 || selection(2) == 1,LOG);
    EXPECT_EQ(selection(3) == 0 || selection(3) == 1,LOG);

    EXPECT_EQ( count(selection.data(), selection.data() + selection.size(), 1)  == 2,LOG);
    EXPECT_EQ( count(selection.data() + 1, selection.data() + selection.size(), 1)  >= 1,LOG);

    // 10 individuals without elitism

    genetic_algorithm.set_individuals_number(8);

    fitness.resize(8);
    fitness.setValues({type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8),});

    genetic_algorithm.set_fitness(fitness);

    selection_errors.resize(8);
    selection_errors.setValues({type(0.8),type(0.7),type(0.6),type(0.5),type(0.4),type(0.3),type(0.2),type(0.1) });

    genetic_algorithm.initialize_population_random();

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.set_elitism_size(2);

    population = genetic_algorithm.get_population();

    for(Index i = 0; i < 10; i++)
    {
        genetic_algorithm.perform_selection();

        selection = genetic_algorithm.get_selection();

        EXPECT_EQ( count(selection.data(), selection.data() + selection.size(), 1)  == 4);
    }
*/
}


TEST(GeneticAlgorithmTest, Crossover)
{
/*
    Tensor<type, 2> data(10,5);
    data.setRandom();
    data_set.set_data(data);

    Tensor<bool, 2> population;
    Tensor<bool, 2> crossover_population;
    Tensor<bool, 1> individual;

    Tensor<type, 1> fitness(4);

    Tensor<type, 1> selection_errors(4);

    // Test

    genetic_algorithm.set_individuals_number(4);

    population.resize(4, 4);

    population.setValues({{true,false,false,false},
                          {true,false,false,true},
                          {true,false,true,false},
                          {true,false,true,true}});

    genetic_algorithm.set_population(population);

    selection_errors.setValues({type(0.4), type(0.3), type(0.2), type(0.1)});

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.perform_fitness_assignment();

    genetic_algorithm.perform_selection();

    genetic_algorithm.perform_crossover();

    crossover_population = genetic_algorithm.get_population();

    for(Index i = 0; i<4; i++)
    {
       EXPECT_EQ(crossover_population(i,0) == 1);
       EXPECT_EQ(crossover_population(i,1) == 0);
    }
*/
}


TEST(GeneticAlgorithmTest, Mutation)
{
/*
    Tensor<type, 2> data(10,5);
    data.setRandom();
    data_set.set_data(data);

    Tensor<bool, 2> population;
    Tensor<bool, 1> individual;
    Tensor<bool, 2> mutated_population;
    Tensor <bool, 1> mutated_individual;

    // Test 1

    genetic_algorithm.set_individuals_number(4);

    population.resize(4,4);
    population.setValues({{true, false, true, false},
                          {false, true, true, false},
                          {true, false, false, true},
                          {false, true, false, true}});

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_mutation_rate(type(0));

    genetic_algorithm.perform_mutation();

    mutated_population = genetic_algorithm.get_population();

    EXPECT_EQ(are_equal(population, mutated_population));

    // Test 2

    genetic_algorithm.set_individuals_number(10);

    population.resize(10,10);
    population.setRandom();

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_mutation_rate(type(0.5));


    genetic_algorithm.perform_mutation();

    mutated_population = genetic_algorithm.get_population();

    Index mutated_genes = 0;

    for(Index i = 0; i < population.dimension(0); i++)
    {
        individual=population.chip(i, 0);

        mutated_individual=mutated_population.chip(i, 0);

        for(Index j = 0; j<10; j++)
        {
            if(individual(j) != mutated_individual(j)) mutated_genes++;
        }
    }

    EXPECT_EQ( mutated_genes >= 25);
*/
}


TEST(GeneticAlgorithmTest, InputSelection)
{
/*
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

    data_set.set_data(data);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2}, {6}, {1});

    genetic_algorithm.set_display(false);

    genetic_algorithm.set_individuals_number(6);

    genetic_algorithm.set_selection_error_goal(1);

    input_selection_results = genetic_algorithm.perform_inputs_selection();

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

    data_set.set_data(data);
    data_set.set_default_raw_variables_uses();

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(4);
    genetic_algorithm.set_selection_error_goal(type(0));
    genetic_algorithm.set_maximum_epochs_number(1);

    input_selection_results = genetic_algorithm.perform_inputs_selection();

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

    data_set.set_data(data);
    data_set.set_default_raw_variables_uses();

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(4);
    genetic_algorithm.set_selection_error_goal(type(0.01));
    genetic_algorithm.set_maximum_epochs_number(10);

    input_selection_results = genetic_algorithm.perform_inputs_selection();

    EXPECT_EQ(input_selection_results.get_epochs_number() <= 100);
*/
}
