//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R U N I N G   I N P U T S   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef PRUNINGINPUTSTEST_H
#define PRUNINGINPUTSTEST_H

// Unit testing includes

#include "unit_testing.h"

class PruningInputsTest : public UnitTesting
{

public:

   // CONSTRUCTOR

   explicit PruningInputsTest();

   virtual ~PruningInputsTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Input selection methods

   void test_perform_inputs_selection();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   PruningInputs pruning_inputs;

};


#endif
