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
    generate_neural_networks();
}


ResponseOptimizationTest::~ResponseOptimizationTest()
{
}


void ResponseOptimizationTest::test_constructor()
{
    cout << "test_constructor\n";

    ResponseOptimization response_optimization_1(&neural_network);

    assert_true(response_optimization_1.get_inputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization_1.get_outputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);

    ResponseOptimization response_optimization_2(&neural_network_2);
    assert_true(response_optimization_2.get_inputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization_2.get_inputs_conditions()(1) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization_2.get_outputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization_2.get_outputs_conditions()(1) == ResponseOptimization::Condition::None, LOG);

    ResponseOptimization response_optimization_3;
}


void ResponseOptimizationTest::test_destructor()
{
    cout << "test_destructor\n";

    ResponseOptimization* response_optimization_1 = new ResponseOptimization;

    delete response_optimization_1;
}


// Set methods

void ResponseOptimizationTest::test_set()
{
    cout << "test_set\n";

    response_optimization.set(&neural_network);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);

    response_optimization.set(&neural_network_2);

    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::None, LOG);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::None, LOG);
}


void ResponseOptimizationTest::test_set_evaluations_number()
{
    cout << "test_set_evaluations_number\n";

    response_optimization.set(&neural_network);

    response_optimization.set_evaluations_number(5);

    assert_true(response_optimization.get_evaluations_number() == 5, LOG);

};


void ResponseOptimizationTest::test_set_input_condition()
{
    cout << "test_set_input_condition\n";

    response_optimization.set(&neural_network);

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


void ResponseOptimizationTest::test_set_output_condition()
{
    cout << "test_set_output_condition\n";

    response_optimization.set(&neural_network_2);

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

    conditions_values.resize(2);
    response_optimization.set_output_condition(1, ResponseOptimization::Condition::Between,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Between, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition(1, ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::GreaterEqualTo, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition(1, ResponseOptimization::Condition::LessEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::LessEqualTo, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition(1, ResponseOptimization::Condition::Maximum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Maximum, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition(1, ResponseOptimization::Condition::Minimum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Minimum, LOG);

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

    conditions_values.resize(2);
    response_optimization.set_output_condition("t", ResponseOptimization::Condition::Between,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Between, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition("t", ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::GreaterEqualTo, LOG);

    conditions_values.resize(1);
    response_optimization.set_output_condition("t", ResponseOptimization::Condition::LessEqualTo,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::LessEqualTo, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition("t", ResponseOptimization::Condition::Maximum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Maximum, LOG);

    conditions_values.resize(0);
    response_optimization.set_output_condition("t", ResponseOptimization::Condition::Minimum,conditions_values);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Minimum, LOG);

};


void ResponseOptimizationTest::test_set_inputs_outputs_conditions()
{
    cout << "test_set_inputs_outputs_conditions\n";

    ResponseOptimization response_optimization(&neural_network,&data_set);

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

    // Multiple outputs

    response_optimization.set(&neural_network_2);

    names.resize(4);
    names.setValues({"x","y","z","t"});
    conditions.resize(4);

    conditions.setValues({"Between","EqualTo","GreaterEqualTo","LessEqualTo"});
    conditions_values.resize(5);
    conditions_values.setConstant(1);
    response_optimization.set_inputs_outputs_conditions(names,conditions,conditions_values);
    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::LessEqualTo, LOG);


    conditions.setValues({"GreaterEqualTo","LessEqualTo","Maximum","Minimum"});
    conditions_values.resize(2);
    conditions_values.setConstant(1);
    response_optimization.set_inputs_outputs_conditions(names,conditions,conditions_values);
    assert_true(response_optimization.get_inputs_conditions()(0) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_inputs_conditions()(1) == ResponseOptimization::Condition::LessEqualTo, LOG);
    assert_true(response_optimization.get_outputs_conditions()(0) == ResponseOptimization::Condition::Maximum, LOG);
    assert_true(response_optimization.get_outputs_conditions()(1) == ResponseOptimization::Condition::Minimum, LOG);


};


void ResponseOptimizationTest::test_get_conditions()
{
    cout << "test_get_conditions\n";

    response_optimization.set(&neural_network);

    Tensor<string,1> conditions_names(3);

    conditions_names.setValues({"Between","Maximize","Minimize"});
    assert_true(response_optimization.get_conditions(conditions_names)(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(1) == ResponseOptimization::Condition::Maximum, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(2) == ResponseOptimization::Condition::Minimum, LOG);

    conditions_names.setValues({"EqualTo","GreaterEqualTo","LessEqualTo"});
    assert_true(response_optimization.get_conditions(conditions_names)(0) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(1) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(2) == ResponseOptimization::Condition::LessEqualTo, LOG);

    // Multiple outputs

    response_optimization.set(&neural_network_2);

    conditions_names.resize(4);

    conditions_names.setValues({"Between","EqualTo","Maximize","Minimize"});
    assert_true(response_optimization.get_conditions(conditions_names)(0) == ResponseOptimization::Condition::Between, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(1) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(2) == ResponseOptimization::Condition::Maximum, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(3) == ResponseOptimization::Condition::Minimum, LOG);

    conditions_names.setValues({"EqualTo","EqualTo","GreaterEqualTo","LessEqualTo"});
    assert_true(response_optimization.get_conditions(conditions_names)(0) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(1) == ResponseOptimization::Condition::EqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(2) == ResponseOptimization::Condition::GreaterEqualTo, LOG);
    assert_true(response_optimization.get_conditions(conditions_names)(3) == ResponseOptimization::Condition::LessEqualTo, LOG);


};


void ResponseOptimizationTest::test_get_values_conditions()
{
    cout << "test_get_values_conditions\n";

    response_optimization.set(&neural_network);

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


void ResponseOptimizationTest::test_calculate_inputs()
{
    cout << "test_calculate_inputs\n";

    ResponseOptimization response_optimization(&neural_network, &data_set);

    Tensor<type,2> inputs = response_optimization.calculate_inputs();

    assert_true(inputs.dimension(0) == response_optimization.get_evaluations_number(), LOG);
    assert_true(inputs.dimension(1) == neural_network.get_inputs_number(), LOG);

    assert_true(inputs(0) <= response_optimization.get_inputs_maximums()(1), LOG);
    assert_true(inputs(1) <= response_optimization.get_inputs_maximums()(1), LOG);
    assert_true(inputs(0) >= response_optimization.get_inputs_minimums()(1), LOG);
    assert_true(inputs(1) >= response_optimization.get_inputs_minimums()(1), LOG);
};


void ResponseOptimizationTest::test_perform_optimization()
{
    cout << "test_perform_optimization\n";

    ResponseOptimization response_optimization(&neural_network,&data_set);

    // Empty results

    conditions_values.resize(1);
    conditions_values.setValues({100000});
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::GreaterEqualTo,conditions_values);

    ResponseOptimizationResults* results = response_optimization.perform_optimization();
    assert_true(results->optimal_variables.size() == 0, LOG);

    // Trivial case 1

    conditions_values.resize(1);
    conditions_values.setValues({1});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_output_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::GreaterEqualTo,conditions_values);

    results = response_optimization.perform_optimization();
    assert_true(results->optimal_variables(0) = 1, LOG);
    assert_true(results->optimal_variables(1) <= 1, LOG);
    assert_true(results->optimal_variables(2) >= 1, LOG);

    // Trivial case 2

    conditions_values.resize(1);
    conditions_values.setValues({1});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_output_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);

    conditions_values.resize(2);
    conditions_values.setValues({1,2.5});
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::Between,conditions_values);

    results = response_optimization.perform_optimization();
    assert_true(results->optimal_variables(0) = 1, LOG);
    assert_true(results->optimal_variables(1) <= 1, LOG);
    assert_true(1 <= results->optimal_variables(2) <= 2.5, LOG);

    // Multiple outputs case 1

    response_optimization.set(&neural_network_2);

    conditions_values.resize(1);
    conditions_values.setValues({1});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_input_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);

    results = response_optimization.perform_optimization();
    assert_true(results->optimal_variables(0) = 1, LOG);
    assert_true(results->optimal_variables(1) <= 1, LOG);
    assert_true(1 <= results->optimal_variables(2) <= 3.0, LOG);
    assert_true(-1 <= results->optimal_variables(3) <= type(1), LOG);

    // Multiple outputs case 1

    conditions_values.resize(1);
    conditions_values.setValues({1});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_input_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);

    conditions_values.resize(2);
    conditions_values.setValues({1,2});
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::Between,conditions_values);
    conditions_values.resize(2);
    conditions_values.setValues({-1,0});
    response_optimization.set_output_condition(1,ResponseOptimization::Condition::Between,conditions_values);

    results = response_optimization.perform_optimization();
    assert_true(results->optimal_variables(0) = 1, LOG);
    assert_true(results->optimal_variables(1) <= 1, LOG);
    assert_true(1 <= results->optimal_variables(2) <= 2.0, LOG);
    assert_true(type(-1) <= results->optimal_variables(3), LOG);
    assert_true(results->optimal_variables(3) <= type(0), LOG);

    // Multiple outputs case 2

    conditions_values.resize(1);
    conditions_values.setValues({0.5});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::GreaterEqualTo,conditions_values);
    conditions_values.resize(2);
    conditions_values.setValues({0.5,1});
    response_optimization.set_input_condition(1,ResponseOptimization::Condition::Between,conditions_values);

    conditions_values.resize(0);
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::Maximum,conditions_values);
    conditions_values.resize(2);
    conditions_values.setValues({-1,0});
    response_optimization.set_output_condition(1,ResponseOptimization::Condition::Between,conditions_values);

    results = response_optimization.perform_optimization();
    assert_true(results->optimal_variables(0) >= 0.5, LOG);

    assert_true(results->optimal_variables(1) >= 0.5, LOG);
    assert_true(results->optimal_variables(2) >= 0.0, LOG);
    assert_true(results->optimal_variables(2) <= 3.0, LOG);
    assert_true(results->optimal_variables(3) >= type(-1), LOG);
    assert_true(results->optimal_variables(3) <= 0.0, LOG);


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

    test_perform_optimization();

    cout << "End of response optimization test case.\n\n";
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
