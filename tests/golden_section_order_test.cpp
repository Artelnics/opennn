/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G O L D E N   S E C T I O N   O R D E R   T E S T   C L A S S   H E A D E R                                */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "golden_section_order_test.h"


using namespace OpenNN;


// CONSTRUCTOR

GoldenSectionOrderTest::GoldenSectionOrderTest() : UnitTesting()
{
}


// DESTRUCTOR

GoldenSectionOrderTest::~GoldenSectionOrderTest()
{
}


// METHODS

// Constructor and destructor methods

void GoldenSectionOrderTest::test_constructor()
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    GoldenSectionOrder gs1(&ts);

    assert_true(gs1.has_training_strategy(), LOG);

    GoldenSectionOrder gs2;

    assert_true(!gs2.has_training_strategy(), LOG);
}

void GoldenSectionOrderTest::test_destructor()
{
    message += "test_destructor\n";

    GoldenSectionOrder* gs = new GoldenSectionOrder;

    delete gs;

}

// Set methods

void GoldenSectionOrderTest::test_set_default()
{
    message += "test_set_default\n";

}

// Order selection methods


// @todo

void GoldenSectionOrderTest::test_perform_order_selection()
{
//    message += "test_perform_order_selection\n";

//    string str;
//    Matrix<double> data;

//    Vector<Instances::Use> uses;

//    NeuralNetwork nn;

//    DataSet ds;

//    SumSquaredError sse(&nn, &ds);

//    TrainingStrategy ts(&sse);

//    GoldenSectionOrder gs(&ts);

//    GoldenSectionOrder::GoldenSectionOrderResults* results;

//    // Test

//    str =
//            "-1 0\n"
//            "-0.9 0\n"
//            "-0.8 0\n"
//            "-0.7 0\n"
//            "-0.6 0\n"
//            "-0.5 0\n"
//            "-0.4 0\n"
//            "-0.3 0\n"
//            "-0.2 0\n"
//            "-0.1 0\n"
//            "0.0 0\n"
//            "0.1 0\n"
//            "0.2 0\n"
//            "0.3 0\n"
//            "0.4 0\n"
//            "0.5 0\n"
//            "0.6 0\n"
//            "0.7 0\n"
//            "0.8 0\n"
//            "0.9 0\n"
//            "1 0\n";

//    data.parse(str);
//    ds.set(data);

//    uses.set(21,Instances::Training);
//    for (size_t i = 0; i < 10; i++)
//        uses[2*i+1] = Instances::Selection;

//    ds.get_instances_pointer()->set_uses(uses);

//    nn.set(1,3,1);
//    nn.initialize_parameters(0.0);

//    sse.set_loss_method(LossIndex::SUM_SQUARED_ERROR);

//    ts.set_training_method(TrainingStrategy::QUASI_NEWTON_METHOD);

//    ts.set_display(false);

//    gs.set_trials_number(1);
//    gs.set_maximum_order(7);
//    gs.set_selection_error_goal(1.0e-3);
//    gs.set_display(false);

//    results = gs.perform_order_selection();

//    assert_true(nn.get_multilayer_perceptron_pointer()->get_layers_perceptrons_numbers()[0] == 1, LOG);
//    assert_true(results->stopping_condition ==
//                OrderSelectionAlgorithm::SelectionLossGoal, LOG);

//    // Test

//    str =
//            "-1 -1\n"
//            "-0.9 -0.9\n"
//            "-0.8 -0.8\n"
//            "-0.7 -0.7\n"
//            "-0.6 -0.6\n"
//            "-0.5 -0.5\n"
//            "-0.4 -0.4\n"
//            "-0.3 -0.3\n"
//            "-0.2 -0.2\n"
//            "-0.1 -0.1\n"
//            "0.0 0.0\n"
//            "0.1 0.1\n"
//            "0.2 0.2\n"
//            "0.3 0.3\n"
//            "0.4 0.4\n"
//            "0.5 0.5\n"
//            "0.6 0.6\n"
//            "0.7 0.7\n"
//            "0.8 0.8\n"
//            "0.9 0.9\n"
//            "1 1\n";

//    data.parse(str);
//    ds.set(data);

//    uses.set(21,Instances::Training);
//    for (size_t i = 0; i < 10; i++)
//        uses[2*i+1] = Instances::Selection;

//    ds.get_instances_pointer()->set_uses(uses);

//    nn.set(1,3,1);
//    nn.initialize_parameters(0.0);

//    sse.set_loss_method(LossIndex::SUM_SQUARED_ERROR);

//    ts.set_training_method(TrainingStrategy::QUASI_NEWTON_METHOD);

//    ts.set_display(false);

//    gs.set_trials_number(1);
//    gs.set_maximum_order(7);
//    gs.set_selection_error_goal(0.0);
//    gs.set_display(false);

//    results = gs.perform_order_selection();

//    assert_true(nn.get_multilayer_perceptron_pointer()->get_layers_perceptrons_numbers()[0] == 1, LOG);


}

// Serialization methods

void GoldenSectionOrderTest::test_to_XML()
{
    message += "test_to_XML\n";

    GoldenSectionOrder gs;

    tinyxml2::XMLDocument* document = gs.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;

}

void GoldenSectionOrderTest::test_from_XML()
{
    message += "test_from_XML\n";

    GoldenSectionOrder gs;

    tinyxml2::XMLDocument* document = gs.to_XML();
    gs.from_XML(*document);

    delete document;

}

// Unit testing methods

void GoldenSectionOrderTest::run_test_case()
{
    message += "Running golden section order test case...\n";

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

    message += "End of golden section order test case.\n";

}
