/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   T R E N D I N G   L A Y E R   T E S T   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Patricia Garcia                                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   patriciagarcia@artelnics.com                                                                               */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef INPUTS_TRENDING_LAYER_TEST_H
#define INPUTS_TRENDING_LAYER_TEST_H

#include "unit_testing.h"

using namespace OpenNN;

class InputsTrendingLayerTest : public UnitTesting
{
    #define STRING(x) #x
    #define TOSTRING(x) STRING(x)
    #define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

    public:

    // GENERAL CONSTRUCTOR

    explicit InputsTrendingLayerTest();

    // DESTRUCTOR

    virtual ~InputsTrendingLayerTest();

    // METHODS

    // Constructor and destructor methods

    void test_constructor();
    void test_destructor();

    // Unit testing methods

    void run_test_case();
};

#endif // INPUTS_TRENDING_LAYER_TEST_H
