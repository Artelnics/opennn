//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   T E S T   C L A S S   H E A D E R 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef GENETICALGORITHMTEST_H
#define GENETICALGORITHMTEST_H

// Unit testing includes

#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/unit_testing.h"
#include "../opennn/genetic_algorithm.h"

namespace opennn
{

class GeneticAlgorithmTest : public UnitTesting
{

public:

   // CONSTRUCTOR

   explicit GeneticAlgorithmTest();  

   virtual ~GeneticAlgorithmTest();

    and destructor

   void test_constructor();
   void test_destructor();

   // Population

   void test_initialize_population();

   // Fitness assignment

   void test_perform_fitness_assignment();

   // Selection

   void test_perform_selection();

   // Crossover

   void test_perform_crossover();

   // Mutation

   void test_perform_mutation();

   // Inputs selection

   void test_perform_inputs_selection();

   // Unit testing

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   GeneticAlgorithm genetic_algorithm;

};

}

#endif
