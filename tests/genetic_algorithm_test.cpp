/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G E N E T I C   A L G O R I T H M   T E S T   C L A S S   H E A D E R                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "genetic_algorithm_test.h"

#include "genetic_algorithm.h"


using namespace OpenNN;


// CONSTRUCTOR

GeneticAlgorithmTest::GeneticAlgorithmTest() : UnitTesting()
{
}


// DESTRUCTOR

GeneticAlgorithmTest::~GeneticAlgorithmTest()
{
}

// METHODS

// Constructor and destructor methods

void GeneticAlgorithmTest::test_constructor()
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    GeneticAlgorithm ga1(&ts);

    assert_true(ga1.has_training_strategy(), LOG);

    GeneticAlgorithm ga2;

    assert_true(!ga2.has_training_strategy(), LOG);
}

void GeneticAlgorithmTest::test_destructor()
{
    message += "test_destructor\n";

    GeneticAlgorithm* ga = new GeneticAlgorithm;

    delete ga;
}

// Set methods

void GeneticAlgorithmTest::test_set_default()
{
    message += "test_set_default\n";
}

// Population methods

void GeneticAlgorithmTest::test_initialize_population()
{
    message += "test_initialize_population\n";

    DataSet ds;

    NeuralNetwork nn(3,2,1);

    SumSquaredError sse(&nn,&ds);

    TrainingStrategy ts(&sse);

    GeneticAlgorithm ga(&ts);

    Vector< Vector<bool> > population;

    ga.set_population_size(10);

    ga.set_inicialization_method(GeneticAlgorithm::Random);

    ga.initialize_population();

    population = ga.get_population();

    assert_true(population.size() == 10, LOG);
    assert_true(population[0].size() == 3, LOG);

}

void GeneticAlgorithmTest::test_calculate_fitness()
{
    message += "test_calculate_fitness\n";

    DataSet ds;

    NeuralNetwork nn(3,2,1);

    SumSquaredError sse(&nn,&ds);

    TrainingStrategy ts(&sse);

    GeneticAlgorithm ga(&ts);

    Matrix<double> loss(4,2);

    Vector<double> fitness;

    ga.set_population_size(4);

    loss(0,1) = 1;
    loss(1,1) = 2;
    loss(2,1) = 3;
    loss(3,1) = 4;

    ga.set_loss(loss);

    ga.set_fitness_assignment_method(GeneticAlgorithm::RankBased);

    ga.calculate_fitness();

    fitness = ga.get_fitness();

    assert_true(fitness.calculate_maximal_index() == 0, LOG);
    assert_true(fitness.calculate_minimal_index() == 3, LOG);

    ga.set_fitness_assignment_method(GeneticAlgorithm::ObjectiveBased);

    ga.calculate_fitness();

    fitness = ga.get_fitness();

    assert_true(fitness.calculate_maximal_index() == 0, LOG);
    assert_true(fitness.calculate_minimal_index() == 3, LOG);
}


void GeneticAlgorithmTest::test_perform_selection()
{
    message += "test_perform_selection\n";

    DataSet ds;

    NeuralNetwork nn(3,2,1);

    SumSquaredError sse(&nn,&ds);

    TrainingStrategy ts(&sse);

    GeneticAlgorithm ga(&ts);

    Vector< Vector<bool> > population;

    Vector< Vector<bool> > selected_population;

    Vector<double> fitness(4);

    Matrix<double> loss(4,2);

    ga.set_population_size(4);

    fitness[0] = 1;
    fitness[1] = 2;
    fitness[2] = 3;
    fitness[3] = 4;

    loss(0,0) = 0.0; loss(0,1) = 0.4;
    loss(1,0) = 0.0; loss(1,1) = 0.3;
    loss(2,0) = 0.0; loss(2,1) = 0.2;
    loss(3,0) = 0.0; loss(3,1) = 0.1;

    ga.set_inicialization_method(GeneticAlgorithm::Random);

    ga.initialize_population();

    ga.set_fitness(fitness);

    ga.set_loss(loss);

    ga.set_elitism_size(2);

    population = ga.get_population();

    ga.perform_selection();

    selected_population = ga.get_population();

    assert_true(selected_population[0] == population[3], LOG);
    assert_true(selected_population[1] == population[2], LOG);
}

// Crossover methods

void GeneticAlgorithmTest::test_perform_crossover()
{
    message += "test_perform_crossover\n";

    DataSet ds;

    NeuralNetwork nn(2,2,1);

    SumSquaredError sse(&nn,&ds);

    TrainingStrategy ts(&sse);

    GeneticAlgorithm ga(&ts);

    Vector< Vector<bool> > population(4);
    Vector<bool> individual(2);

    Vector< Vector<bool> > crossover_population;

    Vector<double> fitness(4);

    Matrix<double> loss(4,2);

    individual[0] = true; individual[1] = true;
    population[0] = individual;
    population[1] = individual;

    individual[0] = false; individual[1] = true;
    population[2] = individual;
    population[3] = individual;

    fitness[0] = 1;
    fitness[1] = 2;
    fitness[2] = 3;
    fitness[3] = 4;

    loss(0,0) = 0.0; loss(0,1) = 0.4;
    loss(1,0) = 0.0; loss(1,1) = 0.3;
    loss(2,0) = 0.0; loss(2,1) = 0.2;
    loss(3,0) = 0.0; loss(3,1) = 0.1;

    ga.set_population_size(4);

    ga.set_population(population);

    ga.set_fitness(fitness);

    ga.set_loss(loss);

    ga.set_elitism_size(2);

    ga.perform_selection();

    ga.set_crossover_method(GeneticAlgorithm::Uniform);

    ga.perform_crossover();

    crossover_population = ga.get_population();

    assert_true(crossover_population[2][1] == true, LOG);

    ga.set_population(population);

    ga.set_fitness(fitness);

    ga.set_loss(loss);

    ga.set_elitism_size(2);

    ga.perform_selection();

    ga.set_crossover_method(GeneticAlgorithm::OnePoint);

    ga.set_crossover_first_point(1);

    ga.perform_crossover();

    crossover_population = ga.get_population();

    assert_true(crossover_population[2][1] == true, LOG);
}

// Mutation methods

void GeneticAlgorithmTest::test_perform_mutation()
{
    message += "test_perform_mutation\n";

    DataSet ds;

    NeuralNetwork nn(1,2,1);

    SumSquaredError sse(&nn,&ds);

    TrainingStrategy ts(&sse);

    GeneticAlgorithm ga(&ts);

    Vector< Vector<bool> > population(4);
    Vector<bool> individual(1);

    Vector< Vector<bool> > mutated_population;

    individual[0] = 1;
    population[0] = individual;
    population[1] = individual;

    individual[0] = 0;
    population[2] = individual;
    population[3] = individual;

    ga.set_population_size(4);

    ga.set_population(population);

    ga.set_mutation_rate(1);

    ga.perform_mutation();

    mutated_population = ga.get_population();

    assert_true(mutated_population[0][0] == 1, LOG);
    assert_true(mutated_population[1][0] == 1, LOG);
    assert_true(mutated_population[2][0] == 1, LOG);
    assert_true(mutated_population[3][0] == 1, LOG);

    ga.set_population(population);

    ga.set_mutation_rate(0);

    ga.perform_mutation();

    mutated_population = ga.get_population();

    assert_true(mutated_population[0][0] == 1, LOG);
    assert_true(mutated_population[1][0] == 1, LOG);
    assert_true(mutated_population[2][0] == 0, LOG);
    assert_true(mutated_population[3][0] == 0, LOG);

}

// Order selection methods


// @todo

void GeneticAlgorithmTest::test_perform_order_selection()
{
//    message += "test_perform_order_selection\n";

//    DataSet ds;

//    Matrix<double> data;

//    NeuralNetwork nn;

//    SumSquaredError sse(&nn,&ds);

//    TrainingStrategy ts(&sse);

//    GeneticAlgorithm ga(&ts);

//    GeneticAlgorithm::GeneticAlgorithmResults* ga_results;

//    // Test

//    data.set(20,3);

//    for (size_t i = 0; i < 20; i++)
//    {
//        data(i,0) = (double)i;
//        data(i,1) = 10.0;
//        data(i,2) = (double)i;
//    }

//    ds.set(data);

//    nn.set(2,6,1);

//    ts.set_display(false);

//    ga.set_display(false);

//    ga.set_approximation(true);

//    ga.set_population_size(10);

//    ga.set_selection_error_goal(1);

//    ga_results = ga.perform_inputs_selection();

//    assert_true(ga_results->final_selection_error < 1, LOG);
//    assert_true(ga_results->stopping_condition == InputsSelectionAlgorithm::SelectionLossGoal, LOG);

//    ga.delete_selection_history();
//    ga.delete_parameters_history();
//    ga.delete_loss_history();

//    // Test

//    size_t j = -10;

//    for (size_t i = 0; i < 10; i++)
//    {
//        data(i,0) = (double)j;
//        data(i,1) = rand();
//        data(i,2) = 1.0;
//        j+=1;
//    }
//    for (size_t i = 10; i < 20; i++)
//    {
//        data(i,0) = (double)i;
//        data(i,1) = rand();
//        data(i,2) = 0.0;
//    }

//    ds.set(data);

//    nn.set(2,6,1);

//    ts.set_display(false);

//    ga.set_display(false);

//    ga.set_approximation(false);

//    ga.set_population_size(10);

//    ga.set_selection_error_goal(0.0);
//    ga.set_maximum_iterations_number(1);

//    ga_results = ga.perform_inputs_selection();

//    assert_true(ga_results->iterations_number == 1, LOG);
//    assert_true(ga_results->stopping_condition == InputsSelectionAlgorithm::MaximumIterations, LOG);

//    ga.delete_selection_history();
//    ga.delete_parameters_history();
//    ga.delete_loss_history();

}

// Serialization methods

void GeneticAlgorithmTest::test_to_XML()
{
    message += "test_to_XML\n";

    GeneticAlgorithm ga;

    tinyxml2::XMLDocument* document = ga.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;
}

void GeneticAlgorithmTest::test_from_XML()
{
    message += "test_from_XML\n";

    GeneticAlgorithm ga;

    tinyxml2::XMLDocument* document = ga.to_XML();
    ga.from_XML(*document);

    delete document;
}

// Unit testing methods

void GeneticAlgorithmTest::run_test_case()
{
    message += "Running genetic algorithm test case...\n";

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

    test_perform_order_selection();

    // Serialization methods

    test_to_XML();

    test_from_XML();

    message += "End of genetic algorithm test case.\n";
}
