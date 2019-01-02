/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E V O L U T I O N A R Y   A L G O R I T H M   T E S T   C L A S S                                          */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includees

#include "evolutionary_algorithm_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR 

EvolutionaryAlgorithmTest::EvolutionaryAlgorithmTest() : UnitTesting()
{
}


// DESTRUCTOR

EvolutionaryAlgorithmTest::~EvolutionaryAlgorithmTest()
{
}


// METHODS

void EvolutionaryAlgorithmTest::test_constructor()
{
   message += "test_constructor\n"; 

   // Default constructor

   EvolutionaryAlgorithm ea1; 
   assert_true(ea1.has_loss_index() == false, LOG);
/*
   // Loss index constructor

   LossIndex pf2;

   EvolutionaryAlgorithm ea2(&pf2); 
   assert_true(ea2.has_loss_index() == true, LOG);

   DataSet ds;
   NeuralNetwork nn3(1, 1);

   LossIndex mof3(&nn3, &ds);

   EvolutionaryAlgorithm ea3(&mof3); 
   assert_true(ea3.has_loss_index() == true, LOG);
*/
}


void EvolutionaryAlgorithmTest::test_destructor()
{
    message += "test_destructor\n";
}


void EvolutionaryAlgorithmTest::test_get_population_size()
{
   message += "test_get_population_size\n";

   EvolutionaryAlgorithm ea;

   size_t population_size = ea.get_population_size();

   assert_true(population_size == 0, LOG);
}


void EvolutionaryAlgorithmTest::test_get_population()
{
   message += "test_get_population\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn,&ds);
   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   Matrix<double> population = ea.get_population();

   size_t rows_number = population.get_rows_number();
   size_t columns_number = population.get_columns_number();

   assert_true(rows_number == 4, LOG);
   assert_true(columns_number == 1, LOG);
}


void EvolutionaryAlgorithmTest::test_get_loss()
{
   message += "test_get_loss\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   Vector<double> evaluation = ea.get_loss();

   size_t size = evaluation.size();

   assert_true(size == 4, LOG);

}


void EvolutionaryAlgorithmTest::test_get_fitness()
{
   message += "test_get_fitnesss\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   Vector<double> fitness = ea.get_fitness();

   size_t size = fitness.size();

   assert_true(size == 4, LOG);

}


void EvolutionaryAlgorithmTest::test_get_selection()
{
   message += "test_get_selection\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn,&ds);
   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   Vector<bool> selection = ea.get_selection();

   size_t size = selection.size();

   assert_true(size == 4, LOG);

}


void EvolutionaryAlgorithmTest::test_get_selective_pressure()
{
   message += "test_get_selective_pressure\n";

   EvolutionaryAlgorithm ea;

   ea.set_fitness_assignment_method(EvolutionaryAlgorithm::LinearRanking);

   ea.set_selective_pressure(1.0);

   assert_true(ea.get_selective_pressure() == 1.0, LOG);
}


void EvolutionaryAlgorithmTest::test_get_recombination_size()
{
   message += "test_get_recombination_size\n";

   EvolutionaryAlgorithm ea;

   ea.set_recombination_size(0.0);

   assert_true(ea.get_recombination_size() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_get_mutation_rate()
{
   message += "test_get_mutation_rate\n";

   EvolutionaryAlgorithm ea;

   ea.set_mutation_rate(0.0);

   assert_true(ea.get_mutation_rate() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_get_mutation_range()
{
   message += "test_get_mutation_range\n";

   EvolutionaryAlgorithm ea;

   ea.set_mutation_range(0.0);

   assert_true(ea.get_mutation_range() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_get_maximum_generations_number()
{
   message += "test_get_maximum_generations_number\n";

   EvolutionaryAlgorithm ea;

   ea.set_maximum_generations_number(1);

   assert_true(ea.get_maximum_generations_number() == 1, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_population_history()
{
   message += "test_get_reserve_population_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_population_history(true);

   assert_true(ea.get_reserve_population_history() == true, LOG);

   ea.set_reserve_population_history(false);

   assert_true(ea.get_reserve_population_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_mean_norm_history()
{
   message += "test_get_reserve_mean_norm_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_mean_norm_history(true);

   assert_true(ea.get_reserve_mean_norm_history() == true, LOG);

   ea.set_reserve_mean_norm_history(false);

   assert_true(ea.get_reserve_mean_norm_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_standard_deviation_norm_history()
{
   message += "test_get_reserve_standard_deviation_norm_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_standard_deviation_norm_history(true);

   assert_true(ea.get_reserve_standard_deviation_norm_history() == true, LOG);

   ea.set_reserve_standard_deviation_norm_history(false);

   assert_true(ea.get_reserve_standard_deviation_norm_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_best_norm_history()
{
   message += "test_get_reserve_best_norm_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_best_norm_history(true);

   assert_true(ea.get_reserve_best_norm_history() == true, LOG);

   ea.set_reserve_best_norm_history(false);

   assert_true(ea.get_reserve_best_norm_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_mean_loss_history()
{
   message += "test_get_reserve_mean_loss_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_mean_loss_history(true);

   assert_true(ea.get_reserve_mean_loss_history() == true, LOG);

   ea.set_reserve_mean_loss_history(false);

   assert_true(ea.get_reserve_mean_loss_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_standard_deviation_loss_history()
{
   message += "test_get_reserve_standard_deviation_loss_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_standard_deviation_loss_history(true);

   assert_true(ea.get_reserve_standard_deviation_loss_history() == true, LOG);

   ea.set_reserve_standard_deviation_loss_history(false);

   assert_true(ea.get_reserve_standard_deviation_loss_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_reserve_best_loss_history()
{
   message += "test_get_reserve_best_loss_history\n";

   EvolutionaryAlgorithm ea;

   ea.set_reserve_best_loss_history(true);

   assert_true(ea.get_reserve_best_loss_history() == true, LOG);

   ea.set_reserve_best_loss_history(false);

   assert_true(ea.get_reserve_best_loss_history() == false, LOG);
}


void EvolutionaryAlgorithmTest::test_get_fitness_assignment_method()
{
   message += "test_get_fitness_assignment_method\n";

   EvolutionaryAlgorithm ea;

   ea.set_fitness_assignment_method(EvolutionaryAlgorithm::LinearRanking);
  
   assert_true(ea.get_fitness_assignment_method() == EvolutionaryAlgorithm::LinearRanking, LOG);
}


void EvolutionaryAlgorithmTest::test_get_selection_method()
{
   message += "test_get_selection_method\n";

   EvolutionaryAlgorithm ea;

   ea.set_selection_method(EvolutionaryAlgorithm::RouletteWheel);
  
   assert_true(ea.get_selection_method() == EvolutionaryAlgorithm::RouletteWheel, LOG);

}


void EvolutionaryAlgorithmTest::test_get_recombination_method()
{
   message += "test_get_recombination_method\n";

   EvolutionaryAlgorithm ea;

   ea.set_recombination_method(EvolutionaryAlgorithm::Line);
  
   assert_true(ea.get_recombination_method() == EvolutionaryAlgorithm::Line, LOG);

   ea.set_recombination_method(EvolutionaryAlgorithm::Intermediate);
  
   assert_true(ea.get_recombination_method() == EvolutionaryAlgorithm::Intermediate, LOG);
}


void EvolutionaryAlgorithmTest::test_get_mutation_method()
{
   message += "test_get_mutation_method\n";

   EvolutionaryAlgorithm ea;

   // Test

   ea.set_mutation_method(EvolutionaryAlgorithm::Normal);
  
   assert_true(ea.get_mutation_method() == EvolutionaryAlgorithm::Normal, LOG);

   // Test

   ea.set_mutation_method(EvolutionaryAlgorithm::Uniform);
  
   assert_true(ea.get_mutation_method() == EvolutionaryAlgorithm::Uniform, LOG);
}


void EvolutionaryAlgorithmTest::test_set()
{
   message += "test_set\n";

}


void EvolutionaryAlgorithmTest::test_set_default()
{
   message += "test_set_default\n";

   DataSet ds;
   NeuralNetwork nn;
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   // Test
   
   ea.set_default();
   assert_true(ea.get_population_size() == 0, LOG);

   // Test

   nn.set(1, 1);
   ea.set_population_size(4);
   assert_true(ea.get_population_size() == 4, LOG);
}


void EvolutionaryAlgorithmTest::test_set_population_size()
{
   message += "test_set_population_size\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);
   
   ea.set_population_size(4);

   assert_true(ea.get_population_size() == 4, LOG);

}


void EvolutionaryAlgorithmTest::test_set_population()
{
   message += "test_set_population\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   size_t parameters_number = nn.get_parameters_number();
   size_t population_size = ea.get_population_size();

   Matrix<double> population(population_size, parameters_number, 0.0);   

   ea.set_population(population);

   assert_true(ea.get_population() == 0.0, LOG);

}


void EvolutionaryAlgorithmTest::test_set_loss()
{
   message += "test_set_loss\n";

   EvolutionaryAlgorithm ea;

   Vector<double> evaluation;

   ea.set_loss(evaluation);

   assert_true(ea.get_loss() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_fitness()
{
   message += "test_set_fitness\n";

   EvolutionaryAlgorithm ea;

   Vector<double> fitness;

   ea.set_fitness(fitness);

   assert_true(ea.get_fitness() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_selection()
{
   message += "test_set_selection\n";

   EvolutionaryAlgorithm ea;

   Vector<double> selection;

   ea.set_fitness(selection);

   assert_true(ea.get_selection() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_selective_pressure()
{
   message += "test_set_selective_pressure\n";

   EvolutionaryAlgorithm ea;

   ea.set_selective_pressure(1.0);

   assert_true(ea.get_selective_pressure() == 1.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_recombination_size()
{
   message += "test_set_recombination_size\n";

   EvolutionaryAlgorithm ea;

   ea.set_recombination_size(0.0);

   assert_true(ea.get_recombination_size() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_mutation_rate()
{
   message += "test_set_mutation_rate\n";

   EvolutionaryAlgorithm ea;

   ea.set_mutation_rate(0.0);

   assert_true(ea.get_mutation_rate() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_mutation_range()
{
   message += "test_set_mutation_range\n";

   EvolutionaryAlgorithm ea;

   ea.set_mutation_range(0.0);

   assert_true(ea.get_mutation_range() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_maximum_generations_number()
{
   message += "test_set_maximum_generations_number\n";

   EvolutionaryAlgorithm ea;

   ea.set_maximum_generations_number(1);

   assert_true(ea.get_maximum_generations_number() == 1, LOG);
}


void EvolutionaryAlgorithmTest::test_set_mean_loss_goal()
{
   message += "test_set_mean_loss_goal\n";

   EvolutionaryAlgorithm ea;

   ea.set_mean_loss_goal(1.0);

   assert_true(ea.get_mean_loss_goal() == 1.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_standard_deviation_loss_goal()
{
   message += "test_set_standard_deviation_loss_goal\n";

   EvolutionaryAlgorithm ea;

   ea.set_standard_deviation_loss_goal(1.0);

   assert_true(ea.get_standard_deviation_loss_goal() == 1.0, LOG);
}


void EvolutionaryAlgorithmTest::test_set_fitness_assignment_method()
{
   message += "test_set_fitness_assignment_method\n";
}


void EvolutionaryAlgorithmTest::test_set_selection_method()
{
   message += "test_set_selection_method\n";
}


void EvolutionaryAlgorithmTest::test_set_recombination_method()
{
   message += "test_set_recombination_method\n";
}


void EvolutionaryAlgorithmTest::test_set_mutation_method()
{
   message += "test_set_mutation_method\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_population_history()
{
   message += "test_set_reserve_population_history\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_mean_norm_history()
{
   message += "test_set_reserve_mean_norm_history\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_standard_deviation_norm_history()
{
   message += "test_set_reserve_standard_deviation_norm_history\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_best_norm_history()
{
   message += "test_set_reserve_best_norm_history\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_mean_loss_history()
{
   message += "test_set_reserve_mean_loss_history\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_standard_deviation_loss_history()
{
   message += "test_set_reserve_standard_deviation_loss_history\n";
}


void EvolutionaryAlgorithmTest::test_set_reserve_best_loss_history()
{
   message += "test_set_reserve_best_loss_history\n";
}


void EvolutionaryAlgorithmTest::test_get_individual()
{
   message += "test_get_individual\n";

   DataSet ds;
   NeuralNetwork nn(1, 1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   // Test

   ea.set_population_size(4);

   ea.initialize_population(0.0);

   assert_true(ea.get_individual(0) == 0.0, LOG);

}


void EvolutionaryAlgorithmTest::test_set_individual()
{
   message += "test_set_individual\n";

   DataSet ds;
   NeuralNetwork nn(1, 1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   Vector<double> individual(2, 0.0);

   // Test

   ea.set_population_size(4);

   ea.set_individual(0, individual);

   assert_true(ea.get_individual(0) == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_randomize_population_uniform()
{
   message += "test_randomize_population_uniform\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   // Test

   ea.set_population_size(4);

   ea.randomize_population_uniform();

   Matrix<double> population = ea.get_population();

   assert_true(population >=-1  && population <= 1.0, LOG);
}


void EvolutionaryAlgorithmTest::test_randomize_population_normal()
{
   message += "test_randomize_population_normal\n";
}


void EvolutionaryAlgorithmTest::test_calculate_population_norm()
{
   message += "test_calculate_population_norm\n";

   DataSet ds;
   NeuralNetwork nn(1);
   SumSquaredError sse(&nn, &ds);
   EvolutionaryAlgorithm ea(&sse);

   // Test

   ea.set_population_size(4);

   ea.initialize_population(0.0);

   assert_true(ea.calculate_population_norm() == 0.0, LOG);
}


void EvolutionaryAlgorithmTest::test_evaluate_population()
{
   message += "test_evaluate_population\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(1, 1, 1);
   ds.initialize_data(0.0);
/*
   SumSquaredError sse(&nn, &ds);
   sse.set_loss_method(LossIndex::SUM_SQUARED_ERROR);

   EvolutionaryAlgorithm ea(&sse);

   // Test

   ea.set_population_size(4);

   ea.initialize_population(0.0);

   ea.evaluate_population();

   assert_true(ea.get_loss() == 0.0, LOG);
*/
}


void EvolutionaryAlgorithmTest::test_perform_linear_ranking_fitness_assignment()
{
   message += "test_perform_linear_ranking_fitness_assignment\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(3, 1, 1);
   ds.randomize_data_normal();

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   size_t population_size;

   double selective_pressure;

   Vector<double> loss;

   Vector<double> fitness;

   // Test

   ea.set_population_size(4);

   selective_pressure = ea.get_selective_pressure();
   population_size = ea.get_population_size();

   ea.randomize_population_normal();

   ea.evaluate_population();

   ea.perform_linear_ranking_fitness_assignment();

   fitness = ea.get_fitness();

   assert_true(fitness >= 0.0, LOG);
   assert_true(fitness <= (population_size-1.0)*selective_pressure, LOG);

   assert_true(fitness.calculate_minimum() == 0.0, LOG);
   assert_true(fitness.calculate_maximum() == (population_size-1.0)*selective_pressure, LOG);

   // Test

   ea.set_population_size(4);

   ea.randomize_population_normal();

   ea.evaluate_population();

   loss = ea.get_loss();

   ea.perform_linear_ranking_fitness_assignment();

   fitness = ea.get_fitness();

   assert_true(loss.calculate_minimal_index() == fitness.calculate_maximal_index(), LOG);
   assert_true(loss.calculate_maximal_index() == fitness.calculate_minimal_index(), LOG);

   // Test

   ea.initialize_population(0.0);

   population_size = ea.get_population_size();

   ea.evaluate_population();

   ea.perform_linear_ranking_fitness_assignment();

   fitness = ea.get_fitness();

   assert_true(fitness.size() == population_size, LOG);

   assert_true(fitness.calculate_maximum() == (population_size-1.0)*selective_pressure, LOG);

   //assert_true(fitness.calculate_standard_deviation() <= 1.0e-3, LOG);

}


void EvolutionaryAlgorithmTest::test_perform_roulette_wheel_selection()
{
   message += "test_perform_roulette_wheel_selection\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   size_t population_size = ea.get_population_size();

   size_t best_individual_index;

   Vector<bool> selection;

   ea.evaluate_population();

   ea.perform_fitness_assignment();

   // Test

   ea.set_population_size(4);
   population_size = ea.get_population_size();

   ea.set_elitism_size(population_size/2);

   ea.perform_roulette_wheel_selection();

   selection = ea.get_selection();

   assert_true(selection.count_equal_to(true) == population_size/2, LOG);

   best_individual_index = ea.calculate_best_individual_index();

   assert_true(selection[best_individual_index] == true, LOG);

   // Test

   ea.set_elitism_size(0);

   ea.perform_roulette_wheel_selection();

   selection = ea.get_selection();

   assert_true(selection.count_equal_to(true) == population_size/2, LOG);

   // Test

   ea.set_elitism_size(1);

   ea.perform_roulette_wheel_selection();

   selection = ea.get_selection();

   assert_true(selection.count_equal_to(true) == population_size/2, LOG);

   best_individual_index = ea.calculate_best_individual_index();

   assert_true(selection[best_individual_index] == true, LOG);

   // Test

   ea.initialize_random();

   population_size = ea.get_population_size();

   ea.initialize_population(1.0);

   ea.evaluate_population();

   ea.perform_fitness_assignment();

   ea.perform_roulette_wheel_selection();

   selection = ea.get_selection();

   assert_true(selection.count_equal_to(true) == population_size/2, LOG);
}


void EvolutionaryAlgorithmTest::test_perform_intermediate_recombination()
{
   message += "test_perform_intermediate_recombination\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   size_t population_size = ea.get_population_size();

   ea.evaluate_population();

   ea.perform_fitness_assignment();

   ea.perform_selection();

   // Test

   ea.perform_intermediate_recombination();

   assert_true(ea.get_population_size() == population_size, LOG);

}


void EvolutionaryAlgorithmTest::test_perform_line_recombination()
{
   message += "test_perform_line_recombination\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   size_t population_size = ea.get_population_size();

   ea.evaluate_population();

   ea.perform_fitness_assignment();

   ea.perform_selection();

   // Test

   ea.perform_line_recombination();

   assert_true(ea.get_population_size() == population_size, LOG);
}


void EvolutionaryAlgorithmTest::test_perform_normal_mutation()
{
   message += "test_perform_normal_mutation\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   size_t population_size = ea.get_population_size();

   ea.evaluate_population();

   ea.perform_fitness_assignment();

   ea.perform_selection();

   ea.perform_recombination();

   // Test

   ea.perform_normal_mutation();

   assert_true(ea.get_population_size() == population_size, LOG);
}


void EvolutionaryAlgorithmTest::test_perform_uniform_mutation()
{
   message += "test_perform_uniform_mutation\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal();

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);

   size_t population_size = ea.get_population_size();

   ea.evaluate_population();

   ea.perform_fitness_assignment();

   ea.perform_selection();

   ea.perform_recombination();

   // Test

   ea.perform_uniform_mutation();

   assert_true(ea.get_population_size() == population_size, LOG);
}


void EvolutionaryAlgorithmTest::test_perform_training()
{
   message += "test_perform_training\n";

   NeuralNetwork nn(1, 1);

   DataSet ds(2, 1, 1);
   ds.randomize_data_normal(0.0, 1.0e-3);

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   ea.set_population_size(4);
/*
   double old_loss = sse.calculate_loss();

   ea.set_display(false);
   ea.set_maximum_generations_number(10);
   ea.perform_training();

   double loss = sse.calculate_loss();

   assert_true(loss <= old_loss, LOG);

   // Performance goal

   ea.randomize_population_normal(0.0, 1.0e-3);

   double best_loss_goal = 1000.0;

   ea.set_best_loss_goal(best_loss_goal);
   ea.set_mean_loss_goal(0.0);
   ea.set_standard_deviation_loss_goal(0.0);
   ea.set_maximum_generations_number(10);
   ea.set_maximum_time(10.0);

   ea.perform_training();

   loss = sse.calculate_loss();

   assert_true(loss < best_loss_goal, LOG);

   // Mean loss goal
   
   ea.randomize_population_normal(0.0, 1.0e-3);

   double mean_loss_goal = 1000.0;

   ea.set_best_loss_goal(0.0);
   ea.set_mean_loss_goal(mean_loss_goal);
   ea.set_standard_deviation_loss_goal(0.0);
   ea.set_maximum_generations_number(10);
   ea.set_maximum_time(100.0);

   ea.perform_training();

   double mean_loss = ea.calculate_mean_loss();

   assert_true(mean_loss < mean_loss_goal, LOG);

   // Standard deviation loss goal

   ea.randomize_population_normal(0.0, 1.0e-3);

   double standard_deviation_loss_goal = 1000.0;

   ea.set_best_loss_goal(0.0);
   ea.set_mean_loss_goal(0.0);
   ea.set_standard_deviation_loss_goal(standard_deviation_loss_goal);
   ea.set_maximum_generations_number(10);
   ea.set_maximum_time(10.0);

   ea.perform_training();

   double standard_deviation_loss = ea.calculate_standard_deviation_loss();

   assert_true(standard_deviation_loss < standard_deviation_loss_goal, LOG);
*/
}


void EvolutionaryAlgorithmTest::test_to_XML()
{
   message += "test_to_XML\n";

   EvolutionaryAlgorithm ea;
   tinyxml2::XMLDocument* document = ea.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;
}


// @todo Check test

void EvolutionaryAlgorithmTest::test_from_XML()
{
   message += "test_from_XML\n";

   DataSet ds;

   NeuralNetwork nn;

   SumSquaredError sse(&nn, &ds);

   EvolutionaryAlgorithm ea(&sse);

   EvolutionaryAlgorithm ea1;
   EvolutionaryAlgorithm ea2;

   tinyxml2::XMLDocument* document;

   Matrix<double> population;

    // Test

   document = ea1.to_XML();

   ea2.from_XML(*document);

   delete document;

   assert_true(ea1 == ea2, LOG);

    // Test

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   nn.set(1, 1);

   ea.set_population_size(4);
   ea.set_elitism_size(0);

   ea.randomize_population_normal();
   population = ea.get_population();

   document = ea.to_XML();

   ea.initialize_population(0.0);

   ea.from_XML(*document);

   delete document;

   assert_true((ea.get_population() - population).calculate_absolute_value() < 1.0e-3, LOG);
}


void EvolutionaryAlgorithmTest::test_set_reserve_all_training_history()
{
   message += "test_set_reserve_all_training_history\n";

   EvolutionaryAlgorithm ea;
   ea.set_reserve_all_training_history(true);
}


void EvolutionaryAlgorithmTest::run_test_case()
{
   message += "Running evolutionary algorithm test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Population methods

   test_get_population_size();

   test_get_population();

   test_get_loss();
   test_get_fitness();
   test_get_selection();

   // Training operators

   test_get_fitness_assignment_method();
   test_get_selection_method();
   test_get_recombination_method();
   test_get_mutation_method();

   // Training

   test_get_selective_pressure();
   test_get_recombination_size();
   test_get_mutation_rate();
   test_get_mutation_range();

   test_get_maximum_generations_number();

   test_get_reserve_population_history();
   test_get_reserve_mean_norm_history();
   test_get_reserve_standard_deviation_norm_history();
   test_get_reserve_best_norm_history();
   test_get_reserve_mean_loss_history();
   test_get_reserve_standard_deviation_loss_history();
   test_get_reserve_best_loss_history();

   // Set methods

   test_set();
   test_set_default();

   test_set_population_size();

   test_set_population();

   test_set_loss();
   test_set_fitness();
   test_set_selection();

   test_set_selective_pressure();
   test_set_recombination_size();

   test_set_mutation_rate();
   test_set_mutation_range();

   test_set_maximum_generations_number();
   test_set_mean_loss_goal();
   test_set_standard_deviation_loss_goal();

   test_set_fitness_assignment_method();
   test_set_selection_method();
   test_set_recombination_method();
   test_set_mutation_method();

   test_set_reserve_population_history();
   test_set_reserve_mean_norm_history();
   test_set_reserve_standard_deviation_norm_history();
   test_set_reserve_best_norm_history();
   test_set_reserve_mean_loss_history();
   test_set_reserve_standard_deviation_loss_history();
   test_set_reserve_best_loss_history();

   test_set_reserve_all_training_history();

   // Population methods

   test_get_individual();

   test_set_individual();

   test_randomize_population_uniform();
   test_randomize_population_normal();

   test_calculate_population_norm();

   // Population evaluation methods

   test_evaluate_population();

   // Fitness assignment methods

   test_perform_linear_ranking_fitness_assignment();

   // Selection methods

   test_perform_roulette_wheel_selection();

   // Recombination methods

   test_perform_intermediate_recombination();
   test_perform_line_recombination();

   // Mutation methods

   test_perform_normal_mutation();
   test_perform_uniform_mutation();

   // Training methods

   test_perform_training();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of evolutionary algorithm test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
