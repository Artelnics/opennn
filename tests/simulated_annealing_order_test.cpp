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


// Unit testing includes

#include "simulated_annealing_order_test.h"


using namespace OpenNN;


// CONSTRUCTOR

SimulatedAnnealingOrderTest::SimulatedAnnealingOrderTest(void) : UnitTesting()
{
}


// DESTRUCTOR

SimulatedAnnealingOrderTest::~SimulatedAnnealingOrderTest(void)
{
}


// METHODS

// Constructor and destructor methods

void SimulatedAnnealingOrderTest::test_constructor(void)
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    SimulatedAnnealingOrder sa1(&ts);

    assert_true(sa1.has_training_strategy(), LOG);

    SimulatedAnnealingOrder sa2;

    assert_true(!sa2.has_training_strategy(), LOG);
}

void SimulatedAnnealingOrderTest::test_destructor(void)
{
    message += "test_destructor\n";

    SimulatedAnnealingOrder* sa = new SimulatedAnnealingOrder;

    delete sa;

}

// Set methods

void SimulatedAnnealingOrderTest::test_set_default(void)
{
    message += "test_set_default\n";

}

// Order selection methods

void SimulatedAnnealingOrderTest::test_perform_order_selection(void)
{
    message += "test_perform_order_selection\n";

    std::string str;
    Matrix<double> data;

    Vector<Instances::Use> uses;

    NeuralNetwork nn;

    DataSet ds;

    PerformanceFunctional pf(&nn, &ds);

    TrainingStrategy ts(&pf);

    SimulatedAnnealingOrder sa(&ts);

    SimulatedAnnealingOrder::SimulatedAnnealingOrderResults* results;

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
    for (size_t i = 0; i < 11; i++)
        uses[2*i+1] = Instances::Selection;

    ds.get_instances_pointer()->set_uses(uses);

    nn.set(1,3,1);
    nn.initialize_parameters(0.0);

    pf.set_error_type(PerformanceFunctional::SUM_SQUARED_ERROR);

    ts.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);

    ts.set_display(false);

    sa.set_trials_number(1);
    sa.set_maximum_order(7);
    sa.set_selection_performance_goal(1.0);
    sa.set_minimum_temperature(0.0);
    sa.set_display(false);

    results = sa.perform_order_selection();

    assert_true(results->stopping_condition ==
                OrderSelectionAlgorithm::SelectionPerformanceGoal, LOG);

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
    for (size_t i = 0; i < 11; i++)
        uses[2*i+1] = Instances::Selection;

    ds.get_instances_pointer()->set_uses(uses);

    nn.set(1,3,1);
    nn.initialize_parameters(0.0);

    pf.set_error_type(PerformanceFunctional::SUM_SQUARED_ERROR);

    ts.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);

    ts.set_display(false);

    sa.set_trials_number(1);
    sa.set_maximum_order(7);
    sa.set_selection_performance_goal(0.0);
    sa.set_minimum_temperature(0.0);
    sa.set_display(false);

    results = sa.perform_order_selection();

    assert_true(results->stopping_condition ==
                OrderSelectionAlgorithm::MaximumIterations, LOG);

}

// Serialization methods

void SimulatedAnnealingOrderTest::test_to_XML(void)
{
    message += "test_to_XML\n";

    SimulatedAnnealingOrder sa;

    tinyxml2::XMLDocument* document = sa.to_XML();
    assert_true(document != NULL, LOG);

    delete document;

}

void SimulatedAnnealingOrderTest::test_from_XML(void)
{
    message += "test_from_XML\n";

    SimulatedAnnealingOrder sa;

    tinyxml2::XMLDocument* document = sa.to_XML();
    sa.from_XML(*document);

    delete document;

}

// Unit testing methods

void SimulatedAnnealingOrderTest::run_test_case(void)
{
    message += "Running simulated annealing order test case...\n";

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

    message += "End of simulated annealing order test case.\n";

}
