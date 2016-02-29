/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   T E S T   C L A S S                                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "model_selection_test.h"
#include "incremental_order_test.h"

using namespace OpenNN;


ModelSelectionTest::ModelSelectionTest(void) : UnitTesting() 
{
}


ModelSelectionTest::~ModelSelectionTest(void) 
{
}


void ModelSelectionTest::test_constructor(void)
{
    message += "test_constructor\n";

    TrainingStrategy ts;

    ModelSelection ms1(&ts);
    assert_true(ms1.has_training_strategy(), LOG);

    ModelSelection ms2;

    assert_true(!ms2.has_training_strategy(), LOG);

}


void ModelSelectionTest::test_destructor(void)
{
    message += "test_destructor\n";

    ModelSelection* ms = new ModelSelection;

    delete ms;
}


void ModelSelectionTest::test_get_training_strategy_pointer(void)
{
    message += "test_get_training_algorithm_pointer\n";

    TrainingStrategy ts;

    ModelSelection ms(&ts);

    assert_true(ms.get_training_strategy_pointer() != NULL, LOG);
}

void ModelSelectionTest::test_set_training_strategy_pointer(void)
{
    message += "test_set_training_strategy_pointer\n";

    TrainingStrategy ts;

    ModelSelection ms;

    ms.set_training_strategy_pointer(&ts);

    assert_true(ms.get_training_strategy_pointer() != NULL, LOG);
}


void ModelSelectionTest::test_set_default(void)
{
    message += "test_set_default\n";
}

void ModelSelectionTest::test_perform_order_selection(void)
{
    message += "test_order_selection\n";

    IncrementalOrderTest iot;

    iot.test_perform_order_selection();
}


void ModelSelectionTest::test_to_XML(void)   
{
    message += "test_to_XML\n";

    ModelSelection ms;

    tinyxml2::XMLDocument* document = ms.to_XML();
    assert_true(document != NULL, LOG);

    delete document;
}

void ModelSelectionTest::test_from_XML(void)
{
    message += "test_from_XML\n";

    ModelSelection ms1;
    ModelSelection ms2;

    ms1.set_order_selection_type(ModelSelection::INCREMENTAL_ORDER);

    tinyxml2::XMLDocument* document = ms1.to_XML();

    ms2.from_XML(*document);

    delete document;

    assert_true(ms2.get_order_selection_type() == ModelSelection::INCREMENTAL_ORDER, LOG);
}

void ModelSelectionTest::test_save(void)
{
    message += "test_save\n";

    std::string file_name = "../data/model_selection.xml";

    ModelSelection ms;

    ms.save(file_name);
}

void ModelSelectionTest::test_load(void)
{
    message += "test_load\n";

    std::string file_name = "../data/model_selection.xml";

    ModelSelection ms;

    ms.set_order_selection_type(ModelSelection::INCREMENTAL_ORDER);

    // Test

    ms.save(file_name);
    ms.load(file_name);

}


void ModelSelectionTest::run_test_case(void)
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
