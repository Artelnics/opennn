/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D E R   S E L E C T I O N   A L G O R I T H M   T E S T   C L A S S   H E A D E R                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
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

   explicit OrderSelectionAlgorithmTest(void);


   // DESTRUCTOR

   virtual ~OrderSelectionAlgorithmTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Get methods

   void test_get_training_strategy_pointer(void);


   void test_get_performance_calculation_method(void);

   void test_write_performance_calculation_method(void) ;

   // Set methods

   void test_set_training_strategy_pointer(void);

   void test_set_default(void);

   void test_set_performance_calculation_method(void);

   // Performances calculation methods

   void test_perform_minimum_model_evaluation(void);
   void test_perform_maximum_model_evaluation(void);
   void test_perform_mean_model_evaluation(void);

   void test_get_final_performances(void);

   void test_perform_model_evaluation(void);

   void test_get_parameters_order(void);

   // Unit testing methods

   void run_test_case(void);

};


#endif
