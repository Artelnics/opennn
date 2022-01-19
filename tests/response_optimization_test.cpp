//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   T E S T   C L A S S                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization_test.h"


ResponseOptimizationTest::ResponseOptimizationTest() : UnitTesting()
{
    generate_neural_network();
}


ResponseOptimizationTest::~ResponseOptimizationTest()
{
}


void ResponseOptimizationTest::test_constructor()
{
    cout << "test_constructor\n";

    ResponseOptimization response_optimization_1(&neural_network);

    ResponseOptimization response_optimization_2;
}


void ResponseOptimizationTest::test_destructor()
{
    cout << "test_destructor\n";

    ResponseOptimization* response_optimization_1 = new ResponseOptimization;

    delete response_optimization_1;
}


// Set methods

void ResponseOptimizationTest::test_set_evaluations_number(){

    cout << "test_set_evaluations_number\n";

    ResponseOptimization response_optimization(&neural_network);

    response_optimization.set_evaluations_number(5);

    assert_true(response_optimization.get_evaluations_number() == 5, LOG);

};


void ResponseOptimizationTest::test_set_input_condition(){

    cout << "test_set_input_condition\n";

    ResponseOptimization response_optimization(&neural_network);

    // Index input

    conditions_values.resize(2);
    response_optimization.set_input_condition(0, ResponseOptimization::Condition::Between,conditions_values);
    conditions_values.resize(1);
    response_optimization.set_input_condition(1, ResponseOptimization::Condition::EqualTo,conditions_values);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::EqualTo, LOG);

    conditions_values.resize(1);
    response_optimization.set_input_condition(0, ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    response_optimization.set_input_condition(1, ResponseOptimization::Condition::LessEqualTo,conditions_values);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::LessEqualTo, LOG);

    response_optimization.set_input_condition(0, ResponseOptimization::Condition::Minimum);
    response_optimization.set_input_condition(1, ResponseOptimization::Condition::Maximum);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::Minimum, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::Maximum, LOG);

    // Name input

    conditions_values.resize(2);
    response_optimization.set_input_condition("x", ResponseOptimization::Condition::Between,conditions_values);
    conditions_values.resize(1);
    response_optimization.set_input_condition("y", ResponseOptimization::Condition::EqualTo,conditions_values);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::EqualTo, LOG);

    conditions_values.resize(1);
    response_optimization.set_input_condition("x", ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    response_optimization.set_input_condition("y", ResponseOptimization::Condition::LessEqualTo,conditions_values);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::LessEqualTo, LOG);

    response_optimization.set_input_condition("x", ResponseOptimization::Condition::Minimum);
    response_optimization.set_input_condition("y", ResponseOptimization::Condition::Maximum);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::Minimum, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::Maximum, LOG);

};


void ResponseOptimizationTest::test_set_output_condition(){

    cout << "test_set_output_condition\n";

    ResponseOptimization response_optimization(&neural_network);

    // Index input

    conditions_values.resize(2);
    response_optimization.set_output_condition(0, ResponseOptimization::Condition::Between,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Between, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition(0, ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition(0, ResponseOptimization::Condition::LessEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::LessEqualTo, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition(0, ResponseOptimization::Condition::Maximum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Maximum, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition(0, ResponseOptimization::Condition::Minimum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Minimum, LOG);

    // Name input

    conditions_values.resize(2);
    response_optimization.set_output_condition("z", ResponseOptimization::Condition::Between,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Between, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition("z", ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition("z", ResponseOptimization::Condition::LessEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::LessEqualTo, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition("z", ResponseOptimization::Condition::Maximum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Maximum, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition("z", ResponseOptimization::Condition::Minimum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Minimum, LOG);

};


void ResponseOptimizationTest::test_set_inputs_outputs_conditions(){

    cout << "test_set_inputs_outputs_conditions\n";

    ResponseOptimization response_optimization(&neural_network);

    Tensor<string,1> names(3);
    names.setValues({"x","y","z"});
    Tensor<string,1> conditions(3);

    conditions.setValues({"Between","EqualTo","GreaterEqualTo"});
    conditions_values.resize(2);
    conditions_values.setConstant(1);
    response_optimization.set_inputs_outputs_conditions(names,conditions,conditions_values);
    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);

    conditions.setValues({"LessEqualTo","Maximum","Minimum"});
    conditions_values.resize(2);
    conditions_values.setConstant(1);
    response_optimization.set_inputs_outputs_conditions(names,conditions,conditions_values);
    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::LessEqualTo, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::Maximum, LOG);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Minimum, LOG);

};


void ResponseOptimizationTest::test_get_conditions(){

    cout << "test_get_conditions\n";

    ResponseOptimization response_optimization(&neural_network);

    Tensor<string,1> conditions_names(3);

    conditions_names.setValues({"Between","Maximize","Minimize"});
    assert_true(response_optimization.get_conditions(conditions_names)(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(1) == ResponseOptimization::Condition::Maximum, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(2) == ResponseOptimization::Condition::Minimum, LOG);

    conditions_names.setValues({"EqualTo","GreaterEqualTo","LessEqualTo"});
    assert_true(response_optimization.get_conditions(conditions_names)(0) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(1) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(2) == ResponseOptimization::Condition::LessEqualTo, LOG);


};


void ResponseOptimizationTest::test_get_values_conditions(){

    cout << "test_get_conditions\n";

    ResponseOptimization response_optimization(&neural_network);

    Tensor<ResponseOptimization::Condition,1> conditions(6);
    conditions.setValues({ResponseOptimization::Condition::Between,
                         ResponseOptimization::Condition::EqualTo,
                         ResponseOptimization::Condition::GreaterEqualTo,
                         ResponseOptimization::Condition::LessEqualTo,
                         ResponseOptimization::Condition::Maximum,
                         ResponseOptimization::Condition::Minimum});
    conditions_values.resize(5);

    assert_true(response_optimization.get_values_conditions(conditions,conditions_values).size() == 6, LOG);
    assert_true(response_optimization.get_values_conditions(conditions,conditions_values)(0).size() == 2, LOG);
    assert_true(response_optimization.get_values_conditions(conditions,conditions_values)(1).size() == 1, LOG);
    assert_true(response_optimization.get_values_conditions(conditions,conditions_values)(2).size() == 1, LOG);
    assert_true(response_optimization.get_values_conditions(conditions,conditions_values)(3).size() == 1, LOG);
    assert_true(response_optimization.get_values_conditions(conditions,conditions_values)(4).size() == 0, LOG);
    assert_true(response_optimization.get_values_conditions(conditions,conditions_values)(5).size() == 0, LOG);


};


void ResponseOptimizationTest::test_calculate_inputs(){

};


void ResponseOptimizationTest::test_calculate_envelope(){

};


void ResponseOptimizationTest::test_perform_optimization(){

};


void ResponseOptimizationTest::run_test_case()
{
    cout << "Running response optimization test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Get methods

    test_set_evaluations_number();

    test_set_input_condition();
    test_set_output_condition();
    test_set_inputs_outputs_conditions();

    test_get_conditions();
    test_get_values_conditions();

    // Performance methods

    test_calculate_inputs();
    test_calculate_envelope();

    test_perform_optimization();

    cout << "End of model selection test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
