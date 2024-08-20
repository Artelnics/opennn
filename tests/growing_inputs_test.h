//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef GROWINGINPUTSTEST_H
#define GROWINGINPUTSTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/growing_inputs.h"

namespace opennn
{

class GrowingInputsTest : public UnitTesting
{

public:

    // CONSTRUCTOR

    explicit GrowingInputsTest();

    virtual ~GrowingInputsTest();

    // Constructor and destructor

    void test_constructor();

    void test_destructor();

    // Input selection

    void test_perform_inputs_selection();

    // Unit testing

    void run_test_case();

private:

    Tensor<type, 2> data;

    DataSet data_set;

    NeuralNetwork neural_network;

    TrainingStrategy training_strategy;

    GrowingInputs growing_inputs;
};

}

#endif
