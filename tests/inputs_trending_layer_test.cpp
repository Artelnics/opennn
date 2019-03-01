/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   T R E N D I N G   L A Y E R   T E S T   C L A S S                                            */
/*                                                                                                              */
/*   Patricia Garcia                                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   patriciagarcia@artelnics.com                                                                               */
/*                                                                                                              */
/****************************************************************************************************************/

#include "inputs_trending_layer_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

InputsTrendingLayerTest::InputsTrendingLayerTest() : UnitTesting()
{
}

// DESTRUCTOR

InputsTrendingLayerTest::~InputsTrendingLayerTest()
{
}

// METHODS

void InputsTrendingLayerTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   InputsTrendingLayer itl1;

   assert_true(itl1.get_inputs_trending_neurons_number() == 0, LOG);

   // Number of neurons constructor

   InputsTrendingLayer itl2(3);

   assert_true(itl2.get_inputs_trending_neurons_number() == 3, LOG);

   // Copy constructor

   InputsTrendingLayer itl3(itl2);

   assert_true(itl3.get_inputs_trending_neurons_number() == 3, LOG);


}


void InputsTrendingLayerTest::test_destructor()
{
   message += "test_destructor\n";
}


void InputsTrendingLayerTest::run_test_case()
{
    // Constructor and destructor methods

    test_constructor();
    test_destructor();
}
