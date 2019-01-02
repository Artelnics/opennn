/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S I M U L A T E D   A N N E A L I N G   O R D E R   T E S T   C L A S S   H E A D E R                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
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

   explicit SimulatedAnnealingOrderTest();


   // DESTRUCTOR

   virtual ~SimulatedAnnealingOrderTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Set methods

   void test_set_default();

   // Order selection methods

   void test_perform_order_selection();

   // Serialization methods

   void test_to_XML();

   void test_from_XML();

   // Unit testing methods

   void run_test_case();

};


#endif
