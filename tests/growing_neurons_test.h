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

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/growing_neurons.h"

namespace opennn
{

class GrowingNeuronsTest : public UnitTesting
{

public:

   explicit GrowingNeuronsTest();

   virtual ~GrowingNeuronsTest();

   // Constructor and destructor

   void test_constructor();

   void test_destructor();

   // Order selection

   void test_perform_neurons_selection();

   // Unit testing

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   GrowingNeurons growing_neurons;

};

}

#endif
