//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T   S E L E C T I O N   A L G O R I T H M   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef INPUTSELECTIONALGORITHMTEST_H
#define INPUTSELECTIONALGORITHMTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class InputsSelectionTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit InputsSelectionTest();  

   virtual ~InputsSelectionTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_training_strategy_pointer();

   void test_get_loss_calculation_method();

   void test_write_loss_calculation_method();

   // Set methods

   void test_set_training_strategy_pointer();

   void test_set_default();

   void test_set_loss_calculation_method();

   // Loss calculation methods

   void test_get_final_loss();

   void test_calculate_losses();

   void test_get_parameters_order();

   // Unit testing methods

   void run_test_case();

};

#endif
