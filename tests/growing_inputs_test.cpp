/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R O W I N G   I N P U T S   T E S T   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "growing_inputs_test.h"


using namespace OpenNN;


// CONSTRUCTOR

GrowingInputsTest::GrowingInputsTest(void) : UnitTesting()
{
}


// DESTRUCTOR

GrowingInputsTest::~GrowingInputsTest(void)
{
}

// METHODS

// Constructor and destructor methods

void GrowingInputsTest::test_constructor(void)
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    GrowingInputs gi1(&ts);

    assert_true(gi1.has_training_strategy(), LOG);

    GrowingInputs gi2;

    assert_true(!gi2.has_training_strategy(), LOG);
}

void GrowingInputsTest::test_destructor(void)
{
    message += "test_destructor\n";

    GrowingInputs* gi = new GrowingInputs;

    delete gi;
}

// Set methods

void GrowingInputsTest::test_set_default(void)
{
    message += "test_set_default\n";
}

// Input selection methods

void GrowingInputsTest::test_perform_inputs_selection(void)
{
    message += "test_perform_inputs_selection\n";

    DataSet ds;

    Matrix<double> data;

    NeuralNetwork nn;

    PerformanceFunctional pf(&nn,&ds);

    TrainingStrategy ts(&pf);

    GrowingInputs gi(&ts);

    GrowingInputs::GrowingInputsResults* gir;

    // Test

    data.set(20,3);

    for (size_t i = 0; i < 20; i++)
    {
        data(i,0) = (double)i;
        data(i,1) = 10.0;
        data(i,2) = (double)i;
    }

    ds.set(data);

    //ds.get_instances_pointer()->split_random_indices();

    nn.set(2,6,1);

    ts.set_display(false);

    gi.set_display(false);

    gi.set_function_regression(true);

    gir = gi.perform_inputs_selection();

    assert_true(gir->optimal_inputs[0] == 1, LOG);

    gi.delete_selection_history();
    gi.delete_parameters_history();
    gi.delete_performance_history();

    // Test

    size_t j = -10;

    for (size_t i = 0; i < 10; i++)
    {
        data(i,0) = (double)i;
        data(i,1) = 10.0;
        data(i,2) = 1.0;
        j+=1;
    }
    for (size_t i = 10; i < 20; i++)
    {
        data(i,0) = (double)i;
        data(i,1) = 10.0;
        data(i,2) = 0.0;
    }

    ds.set(data);

    nn.set(2,6,1);

    ts.set_display(false);

    gi.set_display(false);

    gi.set_function_regression(false);

    gir = gi.perform_inputs_selection();

    assert_true(gir->optimal_inputs[0] == 1, LOG);

    gi.delete_selection_history();
    gi.delete_parameters_history();
    gi.delete_performance_history();
}

// Serialization methods

void GrowingInputsTest::test_to_XML(void)
{
    message += "test_to_XML\n";

    GrowingInputs gi;

    tinyxml2::XMLDocument* document = gi.to_XML();
    assert_true(document != NULL, LOG);

    delete document;
}

void GrowingInputsTest::test_from_XML(void)
{
    message += "test_from_XML\n";

    GrowingInputs gi;

    tinyxml2::XMLDocument* document = gi.to_XML();
    gi.from_XML(*document);

    delete document;
}

// Unit testing methods

void GrowingInputsTest::run_test_case(void)
{
    message += "Running growing input test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set_default();

    // Input selection methods

    test_perform_inputs_selection();

    // Serialization methods

    test_to_XML();

    test_from_XML();

    message += "End of growing input test case.\n";
}
