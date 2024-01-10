//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "genetic_algorithm_test.h"


GeneticAlgorithmTest::GeneticAlgorithmTest() : UnitTesting()
{
    training_strategy.set(&neural_network, &data_set);
    genetic_algorithm.set(&training_strategy);
}


GeneticAlgorithmTest::~GeneticAlgorithmTest()
{
}


void GeneticAlgorithmTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    GeneticAlgorithm genetic_algorithm_1(&training_strategy);
    assert_true(genetic_algorithm_1.has_training_strategy(), LOG);

    // Test

    GeneticAlgorithm genetic_algorithm_2;
    assert_true(!genetic_algorithm_2.has_training_strategy(), LOG);
}


void GeneticAlgorithmTest::test_destructor()
{
    cout << "test_destructor\n";

    GeneticAlgorithm* genetic_algorithm_pointer = new GeneticAlgorithm;
    delete genetic_algorithm_pointer;
}


void GeneticAlgorithmTest::test_initialize_population()
{
    cout << "test_initialize_population\n";

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
    data_set.set(data);

    genetic_algorithm.set_individuals_number(individuals_number);

    genetic_algorithm.initialize_population();

    population = genetic_algorithm.get_population();
    gene = population.chip(0,1);
    individual = population.chip(1,0);

    assert_true(population.dimension(0) == individuals_number, LOG);
    assert_true(population.dimension(1) == inputs_number, LOG);
    assert_true(gene.size() == individuals_number, LOG);
    assert_true(individual.size() == inputs_number, LOG);

    // Test

    inputs_number = 10;
    outputs_number = 3;
    samples_number = 15;
    hidden_neurons_number = 1;

    data.resize(samples_number, inputs_number + outputs_number);
    data.setRandom();
    data_set.set(data);

    Tensor<Index, 1> input_variables_indices(inputs_number);
    Tensor<Index, 1> target_variables_indices(outputs_number);

    input_variables_indices.setValues({0,1,2,3,4,5,6,7,8,9});
    target_variables_indices.setValues({10,11,12});

    data_set.set_input_target_columns(input_variables_indices,target_variables_indices);

    genetic_algorithm.set_individuals_number(individuals_number);

    genetic_algorithm.initialize_population();

    population = genetic_algorithm.get_population();
    gene = population.chip(0,1);
    individual = population.chip(1,0);

    assert_true(population.dimension(0) == individuals_number, LOG);
    assert_true(population.dimension(1) == inputs_number, LOG);

    assert_true(gene.size() == individuals_number, LOG);
    assert_true(individual.size() == inputs_number, LOG);
}



void GeneticAlgorithmTest::test_perform_fitness_assignment()
{
    cout << "test_calculate_fitness\n";

    Tensor<type, 1> selection_errors;

    Tensor<type, 1> fitness;

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {3,2,1});

    genetic_algorithm.set_individuals_number(4);

    selection_errors.resize(4);

    selection_errors(0) = type(1);
    selection_errors(1) = type(2);
    selection_errors(2) = type(3);
    selection_errors(3) = type(4);

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.perform_fitness_assignment();

    fitness = genetic_algorithm.get_fitness();

    assert_true(maximal_index(fitness) == 3, LOG);
    assert_true(minimal_index(fitness) == 0, LOG);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {3,2,1});

    genetic_algorithm.set_individuals_number(4);

    selection_errors.resize(4);

    selection_errors(0) = type(4);
    selection_errors(1) = type(3);
    selection_errors(2) = type(2);
    selection_errors(3) = type(1);

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.perform_fitness_assignment();

    fitness = genetic_algorithm.get_fitness();

    assert_true(maximal_index(fitness) == 0, LOG);
    assert_true(minimal_index(fitness) == 3, LOG);

}


void GeneticAlgorithmTest::test_perform_selection()
{
    cout << "test_perform_selection\n";

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
    selection_errors.setValues({0.4,0.3,0.2,0.1});

    genetic_algorithm.initialize_population();

    genetic_algorithm.set_selection_errors(selection_errors);

    population = genetic_algorithm.get_population();

    genetic_algorithm.perform_selection();

    selection = genetic_algorithm.get_selection();

    genetic_algorithm.set_elitism_size(0);

    assert_true(selection(0) == 0 || selection(0) == 1,LOG);
    assert_true(selection(1) == 0 || selection(1) == 1,LOG);
    assert_true(selection(2) == 0 || selection(2) == 1,LOG);
    assert_true(selection(3) == 0 || selection(3) == 1,LOG);

    assert_true( count(selection.data(), selection.data() + selection.size(), 1)  == 2,LOG);

    assert_true( count(selection.data() + 1, selection.data() + selection.size(), 1)  >= 1,LOG);

    // 4 individuals with elitism size = 1

    genetic_algorithm.set_individuals_number(4);

    fitness.resize(4);
    fitness.setValues({type(1), type(2), type(3), type(4)});

    genetic_algorithm.set_fitness(fitness);

    selection_errors.resize(4);
    selection_errors.setValues({0.4,0.3,0.2,0.1});

    genetic_algorithm.initialize_population();

    genetic_algorithm.set_selection_errors(selection_errors);

    population = genetic_algorithm.get_population();

    genetic_algorithm.set_elitism_size(1);

    genetic_algorithm.perform_selection();

    assert_true(selection(0) == 1,LOG);
    assert_true(selection(1) == 0 || selection(1) == 1,LOG);
    assert_true(selection(2) == 0 || selection(2) == 1,LOG);
    assert_true(selection(3) == 0 || selection(3) == 1,LOG);

    assert_true( count(selection.data(), selection.data() + selection.size(), 1)  == 2,LOG);
    assert_true( count(selection.data() + 1, selection.data() + selection.size(), 1)  >= 1,LOG);

    // 10 individuals without elitism

    genetic_algorithm.set_individuals_number(8);

    fitness.resize(8);
    fitness.setValues({type(1), type(2), type(3), type(4), type(5), type(6), type(7), type(8),});

    genetic_algorithm.set_fitness(fitness);

    selection_errors.resize(8);
    selection_errors.setValues({0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1});

    genetic_algorithm.initialize_population_random();

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.set_elitism_size(2);

    population = genetic_algorithm.get_population();

    for(Index i = 0; i < 10; i++)
    {
        genetic_algorithm.perform_selection();

        selection = genetic_algorithm.get_selection();

        assert_true( count(selection.data(), selection.data() + selection.size(), 1)  == 4, LOG);
    }
}


void GeneticAlgorithmTest::test_perform_crossover()
{
        cout << "test_perform_crossover\n";

        Tensor<type, 2> data(10,5);
        data.setRandom();
        data_set.set(data);

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

        selection_errors.setValues({0.4,0.3,0.2,0.1});

        genetic_algorithm.set_selection_errors(selection_errors);

        genetic_algorithm.perform_fitness_assignment();

        genetic_algorithm.perform_selection();

        genetic_algorithm.perform_crossover();

        crossover_population = genetic_algorithm.get_population();

        for(Index i = 0; i<4; i++)
        {
           assert_true(crossover_population(i,0) == 1, LOG);
           assert_true(crossover_population(i,1) == 0, LOG);
        }

}


void GeneticAlgorithmTest::test_perform_mutation()
{
    cout << "test_perform_mutation\n";

    Tensor<type, 2> data(10,5);
    data.setRandom();
    data_set.set(data);

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

    assert_true(are_equal(population, mutated_population), LOG);

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

    assert_true( mutated_genes >= 25, LOG);
}


void GeneticAlgorithmTest::test_perform_inputs_selection()
{
    cout << "test_perform_inputs_selection\n";

    Tensor<type, 2> data;

    InputsSelectionResults inputs_selection_results;

    // Test 1

    data.resize(20,4);

    for(Index i = 0; i < 20; i++)
    {
        data(i,0) = static_cast<type>(i);
        data(i,1) = type(10.0);
        data(i,2) = type(10.0);
        data(i,3) = static_cast<type>(i);
    }

    data_set.set(data);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2,6,1});

    genetic_algorithm.set_display(false);

    genetic_algorithm.set_individuals_number(6);

    genetic_algorithm.set_selection_error_goal(1);

    inputs_selection_results = genetic_algorithm.perform_inputs_selection();

    assert_true(inputs_selection_results.stopping_condition == InputsSelection::StoppingCondition::SelectionErrorGoal, LOG);
    assert_true(inputs_selection_results.selection_error_history(0) <= 1, LOG);


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
        data(i,2) = type(0.0);
    }

    data_set.set(data);
    data_set.set_default_columns_uses();

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(4);
    genetic_algorithm.set_selection_error_goal(type(0.0));
    genetic_algorithm.set_maximum_epochs_number(1);

    inputs_selection_results = genetic_algorithm.perform_inputs_selection();

    assert_true(genetic_algorithm.get_maximum_iterations_number() == 1, LOG);
    assert_true(genetic_algorithm.get_selection_error_goal() < 1, LOG);

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
    data_set.set_default_columns_uses();

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(4);
    genetic_algorithm.set_selection_error_goal(type(0.01));
    genetic_algorithm.set_maximum_epochs_number(10);

    inputs_selection_results = genetic_algorithm.perform_inputs_selection();

    assert_true(inputs_selection_results.get_epochs_number() <= 100, LOG);

}


void GeneticAlgorithmTest::run_test_case()
{
    cout << "Running genetic algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Population methods

    test_initialize_population();

    test_perform_fitness_assignment();

    // Selection methods

    test_perform_selection();

    // Crossover methods

    test_perform_crossover();

    // Mutation methods

    test_perform_mutation();

    // Order selection methods

    test_perform_inputs_selection();

    cout << "End of genetic algorithm test case.\n\n";
}
