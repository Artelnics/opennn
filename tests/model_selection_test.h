/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: An Open Source Neural Networks C++ Library                                                         */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   T E S T   C L A S S   H E A D E R                                          */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
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

   explicit ModelSelectionTest(void);


   // DESTRUCTOR

   virtual ~ModelSelectionTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Get methods

   void test_get_training_strategy_pointer(void);

   // Set methods

   void test_set_training_strategy_pointer(void);

   void test_set_default(void);

   // Model selection methods

   void test_perform_order_selection(void);

   // Serialization methods

   void test_to_XML(void);   
   void test_from_XML(void);
   void test_save(void);
   void test_load(void);

   // Unit testing methods

   void run_test_case(void);
};


#endif
