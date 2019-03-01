/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D E R   S E L E C T I O N   A L G O R I T H M   T E S T   C L A S S   H E A D E R                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __ORDERSELECTIONALGORITHMTEST_H__
#define __ORDERSELECTIONALGORITHMTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class OrderSelectionAlgorithmTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit OrderSelectionAlgorithmTest();


   // DESTRUCTOR

   virtual ~OrderSelectionAlgorithmTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_training_strategy_pointer();


   void test_get_loss_calculation_method();

   void test_write_loss_calculation_method() ;

   // Set methods

   void test_set_training_strategy_pointer();

   void test_set_default();

   void test_set_loss_calculation_method();

   // Performances calculation methods

   void test_perform_minimum_model_evaluation();
   void test_perform_maximum_model_evaluation();
   void test_perform_mean_model_evaluation();

   void test_get_final_losss();

   void test_perform_model_evaluation();

   void test_get_parameters_order();

   // Unit testing methods

   void run_test_case();

};


#endif
