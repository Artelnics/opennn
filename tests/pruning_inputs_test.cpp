/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R U N I N G   I N P U T S   T E S T   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "pruning_inputs_test.h"


using namespace OpenNN;


// CONSTRUCTOR

PruningInputsTest::PruningInputsTest() : UnitTesting()
{
}


// DESTRUCTOR

PruningInputsTest::~PruningInputsTest()
{
}

// METHODS

// Constructor and destructor methods

void PruningInputsTest::test_constructor()
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    PruningInputs pi1(&ts);

    assert_true(pi1.has_training_strategy(), LOG);

    PruningInputs pi2;

    assert_true(!pi2.has_training_strategy(), LOG);
}

void PruningInputsTest::test_destructor()
{
    message += "test_destructor\n";

    PruningInputs* pi = new PruningInputs;

    delete pi;
}

// Set methods

void PruningInputsTest::test_set_default()
{
    message += "test_set_default\n";
}

// Input selection methods


// @todo

void PruningInputsTest::test_perform_inputs_selection()
{
//    message += "test_perform_inputs_selection\n";

//    DataSet ds;

//    Matrix<double> data;

//    NeuralNetwork nn;

//    SumSquaredError sse(&nn,&ds);

//    TrainingStrategy ts(&sse);

//    PruningInputs pi(&ts);

//    PruningInputs::PruningInputsResults* pir;

//    // Test

//    data.set(20,3);

//    for (size_t i = 0; i < 20; i++)
//    {
//        data(i,0) = (double)i;
//        data(i,1) = 10.0;
//        data(i,2) = (double)i;
//    }

//    ds.set(data);

//    nn.set(2,6,1);

//    ts.set_display(false);

//    pi.set_display(false);

//    pi.set_approximation(true);

//    pir = pi.perform_inputs_selection();

//    assert_true(pir->optimal_inputs[0] == 1, LOG);

//    pi.delete_selection_history();
//    pi.delete_parameters_history();
//    pi.delete_loss_history();

//    // Test

//    size_t j = -10;

//    for (size_t i = 0; i < 10; i++)
//    {
//        data(i,0) = (double)i;
//        data(i,1) = rand();
//        data(i,2) = 1.0;
//        j+=1;
//    }
//    for (size_t i = 10; i < 20; i++)
//    {
//        data(i,0) = (double)i;
//        data(i,1) = rand();
//        data(i,2) = 0.0;
//    }

//    ds.set(data);

//    nn.set(2,6,1);

//    ts.set_display(false);

//    pi.set_display(false);

//    pi.set_approximation(false);

//    pir = pi.perform_inputs_selection();

//    assert_true(pir->optimal_inputs[0] == 1, LOG);

//    pi.delete_selection_history();
//    pi.delete_parameters_history();
//    pi.delete_loss_history();

}

// Serialization methods

void PruningInputsTest::test_to_XML()
{
    message += "test_to_XML\n";

    PruningInputs pi;

    tinyxml2::XMLDocument* document = pi.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;
}

void PruningInputsTest::test_from_XML()
{
    message += "test_from_XML\n";

    PruningInputs pi;

    tinyxml2::XMLDocument* document = pi.to_XML();
    pi.from_XML(*document);

    delete document;
}

// Unit testing methods

void PruningInputsTest::run_test_case()
{
    message += "Running pruning input test case...\n";

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

    message += "End of pruning input test case.\n";
}
