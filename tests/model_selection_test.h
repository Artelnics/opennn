/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: An Open Source Neural Networks C++ Library                                                         */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   T E S T   C L A S S   H E A D E R                                          */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MODELSELECTIONTEST_H__
#define __MODELSELECTIONTEST_H__

#include "unit_testing.h"

using namespace OpenNN;


class ModelSelectionTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit ModelSelectionTest();


   // DESTRUCTOR

   virtual ~ModelSelectionTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_training_strategy_pointer();

   // Set methods

   void test_set_training_strategy_pointer();

   void test_set_default();

   // Model selection methods

   void test_perform_order_selection();

   // Serialization methods

   void test_to_XML();   
   void test_from_XML();
   void test_save();
   void test_load();

   // Unit testing methods

   void run_test_case();
};


#endif
