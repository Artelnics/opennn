/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N C R E M E N T A L   O R D E R   T E S T   C L A S S   H E A D E R                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "incremental_order_test.h"


using namespace OpenNN;


// CONSTRUCTOR

IncrementalOrderTest::IncrementalOrderTest() : UnitTesting()
{
}


// DESTRUCTOR

IncrementalOrderTest::~IncrementalOrderTest()
{
}


// METHODS

// Constructor and destructor methods

void IncrementalOrderTest::test_constructor()
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    IncrementalOrder io1(&ts);

    assert_true(io1.has_training_strategy(), LOG);

    IncrementalOrder io2;

    assert_true(!io2.has_training_strategy(), LOG);
}

void IncrementalOrderTest::test_destructor()
{
    message += "test_destructor\n";

    IncrementalOrder* io = new IncrementalOrder;

    delete io;

}

// Set methods

void IncrementalOrderTest::test_set_default()
{
    message += "test_set_default\n";

}

// Order selection methods

void IncrementalOrderTest::test_perform_order_selection()
{
    message += "test_perform_order_selection\n";

    string str;
    Matrix<double> data;

    Vector<Instances::Use> uses;

    NeuralNetwork nn;

    DataSet ds;

    SumSquaredError sse(&nn, &ds);

    TrainingStrategy ts(&sse);

    IncrementalOrder io(&ts);

    IncrementalOrder::IncrementalOrderResults* results = nullptr;

    // Test

    str =
            "-1 0\n"
            "-0.9 0\n"
            "-0.8 0\n"
            "-0.7 0\n"
            "-0.6 0\n"
            "-0.5 0\n"
            "-0.4 0\n"
            "-0.3 0\n"
            "-0.2 0\n"
            "-0.1 0\n"
            "0.0 0\n"
            "0.1 0\n"
            "0.2 0\n"
            "0.3 0\n"
            "0.4 0\n"
            "0.5 0\n"
            "0.6 0\n"
            "0.7 0\n"
            "0.8 0\n"
            "0.9 0\n"
            "1 0\n";

    data.parse(str);
    ds.set(data);

    uses.set(21,Instances::Training);
    for (size_t i = 0; i < 10; i++)
        uses[2*i+1] = Instances::Selection;

    ds.get_instances_pointer()->set_uses(uses);

    nn.set(1,3,1);
    nn.initialize_parameters(0.0);

    ts.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

    ts.set_training_method(TrainingStrategy::QUASI_NEWTON_METHOD);

    ts.set_display(false);

    io.set_trials_number(1);
    io.set_maximum_order(7);
    io.set_selection_error_goal(1.0e-3);
    io.set_display(false);

    results = io.perform_order_selection();

    assert_true(nn.get_multilayer_perceptron_pointer()->get_layers_perceptrons_numbers()[0] == 1, LOG);
    assert_true(results->stopping_condition ==
                OrderSelectionAlgorithm::SelectionErrorGoal, LOG);

    // Test

    str =
            "-1 -1\n"
            "-0.9 -0.9\n"
            "-0.8 -0.8\n"
            "-0.7 -0.7\n"
            "-0.6 -0.6\n"
            "-0.5 -0.5\n"
            "-0.4 -0.4\n"
            "-0.3 -0.3\n"
            "-0.2 -0.2\n"
            "-0.1 -0.1\n"
            "0.0 0.0\n"
            "0.1 0.1\n"
            "0.2 0.2\n"
            "0.3 0.3\n"
            "0.4 0.4\n"
            "0.5 0.5\n"
            "0.6 0.6\n"
            "0.7 0.7\n"
            "0.8 0.8\n"
            "0.9 0.9\n"
            "1 1\n";

    data.parse(str);
    ds.set(data);

    uses.set(21,Instances::Training);
    for (size_t i = 0; i < 10; i++)
        uses[2*i+1] = Instances::Selection;

    ds.get_instances_pointer()->set_uses(uses);

    nn.set(1,3,1);
    nn.initialize_parameters(0.0);

    ts.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

    ts.set_training_method(TrainingStrategy::QUASI_NEWTON_METHOD);

    ts.set_display(false);

    io.set_trials_number(1);
    io.set_maximum_order(7);
    io.set_selection_error_goal(0.0);
    io.set_maximum_selection_failures(1);
    io.set_display(false);

    results = io.perform_order_selection();

    assert_true(nn.get_multilayer_perceptron_pointer()->get_layers_perceptrons_numbers()[0] == 1, LOG);
    assert_true(results->stopping_condition ==
                OrderSelectionAlgorithm::MaximumSelectionFailures, LOG);

}

// Serialization methods

void IncrementalOrderTest::test_to_XML()
{
    message += "test_to_XML\n";

    IncrementalOrder io;

    tinyxml2::XMLDocument* document = io.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;

}

void IncrementalOrderTest::test_from_XML()
{
    message += "test_from_XML\n";

    IncrementalOrder io;

    tinyxml2::XMLDocument* document = io.to_XML();
    io.from_XML(*document);

    delete document;

}

// Unit testing methods

void IncrementalOrderTest::run_test_case()
{
    message += "Running incremental order test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set_default();

    // Order selection methods

    test_perform_order_selection();

    // Serialization methods

    test_to_XML();

    test_from_XML();

    message += "End of incremental order test case.\n";

}
