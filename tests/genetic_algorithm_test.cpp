//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   T E S T   C L A S S   H E A D E R 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "genetic_algorithm_test.h"


GeneticAlgorithmTest::GeneticAlgorithmTest() : UnitTesting()
{
}


GeneticAlgorithmTest::~GeneticAlgorithmTest()
{
}


void GeneticAlgorithmTest::test_constructor()
{
    cout << "test_constructor\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TrainingStrategy training_strategy(&neural_network, &data_set);

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

    GeneticAlgorithm* genetic_algorithm = new GeneticAlgorithm;

    delete genetic_algorithm;
}


void GeneticAlgorithmTest::test_set_default()
{
    cout << "test_set_default\n";
}


void GeneticAlgorithmTest::test_initialize_population()
{
    cout << "test_initialize_population\n";

    DataSet data_set;

    NeuralNetwork neural_network;
    Tensor<Index, 1> architecture;

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    Tensor<bool, 2> population;
    Tensor<bool, 1> gene;

    // Test

    architecture.resize(3);
    architecture.setValues({3,2,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    genetic_algorithm.set_individuals_number(10);

    genetic_algorithm.initialize_population();

    population = genetic_algorithm.get_population();
    gene = population.chip(0,1);

    assert_true(population.size() == 10, LOG);
    assert_true(gene.size() == 3, LOG);
}


/// @todo

void GeneticAlgorithmTest::test_calculate_fitness()
{
    cout << "test_calculate_fitness\n";

    DataSet data_set;

    NeuralNetwork neural_network;
    Tensor<Index, 1> architecture;

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    Tensor<type, 1> selection_errors;

    Tensor<type, 1> fitness;

    // Test

    architecture.resize(3);
    architecture.setValues({3,2,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    genetic_algorithm.set_individuals_number(4);

    selection_errors.resize(4);

    selection_errors(0) = 1;
    selection_errors(1) = 2;
    selection_errors(2) = 3;
    selection_errors(3) = 4;

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.perform_fitness_assignment();

    fitness = genetic_algorithm.get_fitness();

    assert_true(maximal_index(fitness) == 0, LOG);
    assert_true(minimal_index(fitness) == 3, LOG);
}


void GeneticAlgorithmTest::test_perform_selection()
{
    cout << "test_perform_selection\n";

    DataSet data_set;

    NeuralNetwork neural_network;
    Tensor<Index, 1> architecture;

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    Tensor<bool, 2> population;

    Tensor<bool, 1> selection;

    Tensor<type, 1> selection_errors;
    Tensor<type, 1> fitness;

    // Test

    architecture.resize(3);
    architecture.setValues({3,2,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    genetic_algorithm.set_individuals_number(4);

    fitness.resize(4);
    fitness[0] = 1;
    fitness[1] = 2;
    fitness[2] = 3;
    fitness[3] = 4;

    genetic_algorithm.set_fitness(fitness);

    selection_errors.resize(4);
    selection_errors(0) = static_cast<type>(0.4);
    selection_errors(1) = static_cast<type>(0.3);
    selection_errors(2) = static_cast<type>(0.2);
    selection_errors(3) = static_cast<type>(0.1);

    genetic_algorithm.initialize_population();

    genetic_algorithm.set_selection_errors(selection_errors);

    genetic_algorithm.set_elitism_size(2);

    population = genetic_algorithm.get_population();

    genetic_algorithm.perform_selection();

    selection = genetic_algorithm.get_selection();

//    assert_true(selected_population[0] == population[3], LOG);
//    assert_true(selected_population[1] == population[2], LOG);

}


void GeneticAlgorithmTest::test_perform_crossover()
{
    cout << "test_perform_crossover\n";

    DataSet data_set;

    Tensor<Index, 1> architecture(3);
    architecture.setValues({2,2,1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    Tensor<bool, 2> population;
    Tensor<bool, 1> individual;

    Tensor<type, 1> fitness;

    // Test

    genetic_algorithm.set_individuals_number(4);

    population.resize(4, 4);
    individual.resize(2);
    individual[0] = true;
    individual[1] = true;
//    population[0] = individual[0];
//    population[1] = individual[1];

    genetic_algorithm.set_population(population);

    fitness.resize(4);
    fitness[0] = 1;
    fitness[1] = 2;
    fitness[2] = 3;
    fitness[3] = 4;

    genetic_algorithm.set_fitness(fitness);

//    loss(0,0) = 0.0; loss(0,1) = static_cast<type>(0.4);
//    loss(1,0) = 0.0; loss(1,1) = static_cast<type>(0.3);
//    loss(2,0) = 0.0; loss(2,1) = static_cast<type>(0.2);
//    loss(3,0) = 0.0; loss(3,1) = static_cast<type>(0.1);




//    genetic_algorithm.set_loss(loss);

    genetic_algorithm.set_elitism_size(2);

    genetic_algorithm.perform_selection();

    genetic_algorithm.perform_crossover();

//    crossover_population = genetic_algorithm.get_population();

//    assert_true(crossover_population(2,1), LOG);

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_fitness(fitness);

//    genetic_algorithm.set_loss(loss);

    genetic_algorithm.set_elitism_size(2);

    genetic_algorithm.perform_selection();

    genetic_algorithm.perform_crossover();

//    crossover_population = genetic_algorithm.get_population();

//    assert_true(crossover_population(2,1), LOG);

}


void GeneticAlgorithmTest::test_perform_mutation()
{
    cout << "test_perform_mutation\n";

    DataSet data_set;

    NeuralNetwork neural_network;
    Tensor<Index, 1> architecture;

    SumSquaredError sum_squared_error(&neural_network, &data_set);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    Tensor<bool, 2> population(4,1);
//    Tensor<bool, 1> individual(1);

    Tensor<bool, 2> mutated_population;

    // Test

    architecture.resize(3);
    architecture.setValues({1,2,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    genetic_algorithm.set_individuals_number(4);

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_mutation_rate(1);

    genetic_algorithm.perform_mutation();

    mutated_population = genetic_algorithm.get_population();

    population.resize(4,1);

    genetic_algorithm.set_population(population);

    genetic_algorithm.set_mutation_rate(0);

    genetic_algorithm.perform_mutation();

    mutated_population = genetic_algorithm.get_population();

    assert_true(mutated_population(0,0) == 1, LOG);
    assert_true(mutated_population(1,0) == 1, LOG);
    assert_true(mutated_population(2,0) == 0, LOG);
    assert_true(mutated_population(3,0) == 0, LOG);

}


void GeneticAlgorithmTest::test_perform_inputs_selection()
{
    cout << "test_perform_inputs_selection\n";

    DataSet data_set;

    Tensor<type, 2> data;

    NeuralNetwork neural_network;
    Tensor<Index, 1> architecture;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GeneticAlgorithm genetic_algorithm(&training_strategy);

    InputsSelectionResults inputs_selection_results;

    // Test

    data.resize(20,3);

    for(Index i = 0; i < 20; i++)
    {
        data(i,0) = static_cast<type>(i);
        data(i,1) = 10.0;
        data(i,2) = static_cast<type>(i);
    }

    data_set.set(data);

    architecture.resize(3);
    architecture.setValues({2,6,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    genetic_algorithm.set_display(false);

    genetic_algorithm.set_individuals_number(10);

    genetic_algorithm.set_selection_error_goal(1);

    inputs_selection_results = genetic_algorithm.perform_inputs_selection();

//    assert_true(ga_results->selection_error < 1, LOG);
    assert_true(inputs_selection_results.stopping_condition == InputsSelection::SelectionErrorGoal, LOG);

//    genetic_algorithm.delete_selection_history();
//    genetic_algorithm.delete_parameters_history();
//    genetic_algorithm.delete_loss_history();

    // Test

    Index j = -10;

    for(Index i = 0; i < 10; i++)
    {
        data(i,0) = (type)j;
        data(i,1) = rand();
        data(i,2) = 1.0;
        j+=1;
    }
    for(Index i = 10; i < 20; i++)
    {
        data(i,0) = (type)i;
        data(i,1) = rand();
        data(i,2) = 0.0;
    }

    data_set.set(data);

//    data_set.generate_inputs_selection_data(20,3);

    data_set.set_default_columns_uses();

    neural_network.set(NeuralNetwork::Approximation, architecture);

    genetic_algorithm.set_individuals_number(10);

    genetic_algorithm.set_selection_error_goal(0.0);
    genetic_algorithm.set_maximum_iterations_number(1);

    inputs_selection_results = genetic_algorithm.perform_inputs_selection();

//    assert_true(genetic_algorithm.iterations_number == 1, LOG);
//    assert_true(genetic_algorithm.selection_error < 1, LOG);
//    assert_true(genetic_algorithm.stopping_condition == InputsSelection::SelectionErrorGoal, LOG);

}


void GeneticAlgorithmTest::test_to_XML()
{
    cout << "test_to_XML\n";

    GeneticAlgorithm genetic_algorithm;

//    tinyxml2::XMLDocument* document = genetic_algorithm.to_XML();
//    assert_true(document != nullptr, LOG);

//    delete document;
}


void GeneticAlgorithmTest::test_from_XML()
{
    cout << "test_from_XML\n";

    GeneticAlgorithm genetic_algorithm;

//    tinyxml2::XMLDocument* document = genetic_algorithm.to_XML();
//    genetic_algorithm.from_XML(*document);

//    delete document;
}


void GeneticAlgorithmTest::run_test_case()
{
    cout << "Running genetic algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set_default();

    // Population methods

    test_initialize_population();

    test_calculate_fitness();

    // Selection methods

    test_perform_selection();

    // Crossover methods

    test_perform_crossover();

    // Mutation methods

    test_perform_mutation();

    // Order selection methods

    test_perform_inputs_selection();

    // Serialization methods

    test_to_XML();

    test_from_XML();

    cout << "End of genetic algorithm test case.\n\n";
}
