//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef GROWINGNEURONSTEST_H
#define GROWINGNEURONSTEST_H

// Unit testing includes

#include "unit_testing.h"

class GrowingNeuronsTest : public UnitTesting
{


public:

   explicit GrowingNeuronsTest();

   virtual ~GrowingNeuronsTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Order selection methods

   void test_perform_neurons_selection();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   GrowingNeurons growing_neurons;

};


#endif
