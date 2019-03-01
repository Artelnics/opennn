/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   T E S T   C L A S S                                                        */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "model_selection_test.h"
#include "incremental_order_test.h"

using namespace OpenNN;


ModelSelectionTest::ModelSelectionTest() : UnitTesting() 
{
}


ModelSelectionTest::~ModelSelectionTest() 
{
}


void ModelSelectionTest::test_constructor()
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    ModelSelection ms1(&ts);
    assert_true(ms1.has_training_strategy(), LOG);

    ModelSelection ms2;

    assert_true(!ms2.has_training_strategy(), LOG);

}


void ModelSelectionTest::test_destructor()
{
    message += "test_destructor\n";

    ModelSelection* ms = new ModelSelection;

    delete ms;
}


void ModelSelectionTest::test_get_training_strategy_pointer()
{
    message += "test_get_optimization_algorithm_pointer\n";

    TrainingStrategy ts;

    ModelSelection ms(&ts);

    assert_true(ms.get_training_strategy_pointer() != nullptr, LOG);
}

void ModelSelectionTest::test_set_training_strategy_pointer()
{
    message += "test_set_training_strategy_pointer\n";

    TrainingStrategy ts;

    ModelSelection ms;

    ms.set_training_strategy_pointer(&ts);

    assert_true(ms.get_training_strategy_pointer() != nullptr, LOG);
}


void ModelSelectionTest::test_set_default()
{
    message += "test_set_default\n";
}

void ModelSelectionTest::test_perform_order_selection()
{
    message += "test_order_selection\n";

    IncrementalOrderTest iot;

    iot.test_perform_order_selection();
}


void ModelSelectionTest::test_to_XML()   
{
    message += "test_to_XML\n";

    ModelSelection ms;

    tinyxml2::XMLDocument* document = ms.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;
}

void ModelSelectionTest::test_from_XML()
{
    message += "test_from_XML\n";

    ModelSelection ms1;
    ModelSelection ms2;

    ms1.set_order_selection_method(ModelSelection::INCREMENTAL_ORDER);

    tinyxml2::XMLDocument* document = ms1.to_XML();

    ms2.from_XML(*document);

    delete document;

    assert_true(ms2.get_order_selection_method() == ModelSelection::INCREMENTAL_ORDER, LOG);
}

void ModelSelectionTest::test_save()
{
    message += "test_save\n";

    string file_name = "../data/model_selection.xml";

    ModelSelection ms;

    ms.save(file_name);
}

void ModelSelectionTest::test_load()
{
    message += "test_load\n";

    string file_name = "../data/model_selection.xml";

    ModelSelection ms;

    ms.set_order_selection_method(ModelSelection::INCREMENTAL_ORDER);

    // Test

    ms.save(file_name);
    ms.load(file_name);

}


void ModelSelectionTest::run_test_case()
{
    message += "Running model selection test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Get methods

    test_get_training_strategy_pointer();

    // Set methods

    test_set_training_strategy_pointer();

    test_set_default();

    // Model selection methods

    test_perform_order_selection();

    // Serialization methods

    test_to_XML();
    test_from_XML();
    test_save();
    test_load();

    message += "End of model selection test case.\n";
}
