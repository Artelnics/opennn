//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef INPUTSELECTIONALGORITHMTEST_H
#define INPUTSELECTIONALGORITHMTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class InputsSelectionTest : public UnitTesting
{

public:

   // CONSTRUCTOR

   explicit InputsSelectionTest();  

   virtual ~InputsSelectionTest();

   // Constructor and destructor methods

   void test_constructor();

   void test_destructor();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   GrowingInputs growing_inputs;

};

#endif
