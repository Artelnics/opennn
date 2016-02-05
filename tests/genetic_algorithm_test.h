/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G E N E T I C   A L G O R I T H M   T E S T   C L A S S   H E A D E R                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __GENETICALGORITHMTEST_H__
#define __GENETICALGORITHMTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class GeneticAlgorithmTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit GeneticAlgorithmTest(void);


   // DESTRUCTOR

   virtual ~GeneticAlgorithmTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Set methods

   void test_set_default(void);

   // Population methods

   void test_initialize_population(void);

   void test_calculate_fitness(void);

   // Selection methods

   void test_perform_selection(void);

   // Crossover methods

   void test_perform_crossover(void);

   // Mutation methods

   void test_perform_mutation(void);

   // Order selection methods

   void test_perform_order_selection(void);

   // Serialization methods

   void test_to_XML(void);

   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);

};


#endif
