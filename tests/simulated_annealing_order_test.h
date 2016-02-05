/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S I M U L A T E D   A N N E A L I N G   O R D E R   T E S T   C L A S S   H E A D E R                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __SIMULATEDANNEALINGORDERTEST_H__
#define __SIMULATEDANNEALINGORDERTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class SimulatedAnnealingOrderTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit SimulatedAnnealingOrderTest(void);


   // DESTRUCTOR

   virtual ~SimulatedAnnealingOrderTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Set methods

   void test_set_default(void);

   // Order selection methods

   void test_perform_order_selection(void);

   // Serialization methods

   void test_to_XML(void);

   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);

};


#endif
