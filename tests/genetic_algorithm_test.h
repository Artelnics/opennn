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

#include "unit_testing.h"

class GeneticAlgorithmTest : public UnitTesting
{

public:

   // CONSTRUCTOR

   explicit GeneticAlgorithmTest();  

   virtual ~GeneticAlgorithmTest(); 

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Population methods

   void test_initialize_population();

   // Fitness assignment methods

   void test_perform_fitness_assignment();

   // Selection methods

   void test_perform_selection();

   // Crossover methods

   void test_perform_crossover();

   // Mutation methods

   void test_perform_mutation();

   // Inputs selection methods

   void test_perform_inputs_selection();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   GeneticAlgorithm genetic_algorithm;

};


#endif
