//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef GROWINGINPUTSTEST_H
#define GROWINGINPUTSTEST_H

// Unit testing includes

#include "unit_testing.h"

class GrowingInputsTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit GrowingInputsTest();

   virtual ~GrowingInputsTest();

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

   GrowingInputs growing_inputs;
};


#endif
