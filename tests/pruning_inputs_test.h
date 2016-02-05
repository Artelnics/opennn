/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R U N I N G   I N P U T S   T E S T   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PRUNINGINPUTSTEST_H__
#define __PRUNINGINPUTSTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class PruningInputsTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit PruningInputsTest(void);


   // DESTRUCTOR

   virtual ~PruningInputsTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Set methods

   void test_set_default(void);

   // Input selection methods

   void test_perform_inputs_selection(void);

   // Serialization methods

   void test_to_XML(void);

   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);

};


#endif
