/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O U T P U T S   T R E N D I N G   L A Y E R   T E S T   C L A S S   H E A D E R                            */
/*                                                                                                              */
/*   Patricia Garcia                                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   patriciagarcia@artelnics.com                                                                               */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef OUTPUTS_TRENDING_LAYER_TEST_H
#define OUTPUTS_TRENDING_LAYER_TEST_H

#include "unit_testing.h"

using namespace OpenNN;

class OutputsTrendingLayerTest : public UnitTesting
{
    #define STRING(x) #x
    #define TOSTRING(x) STRING(x)
    #define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

    public:

    // GENERAL CONSTRUCTOR

    explicit OutputsTrendingLayerTest();

    // DESTRUCTOR

    virtual ~OutputsTrendingLayerTest();

    // Unit testing methods

    void run_test_case();
};

#endif // OUTPUTS_TRENDING_LAYER_TEST_H
