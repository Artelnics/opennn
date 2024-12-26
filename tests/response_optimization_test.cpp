#include "pch.h"

/*
void ResponseOptimizationTest::test_constructor()
{
    ResponseOptimization response_optimization_1(&neural_network);

    EXPECT_EQ(response_optimization_1.get_inputs_conditions()(0) == ResponseOptimization::Condition::None);
    EXPECT_EQ(response_optimization_1.get_outputs_conditions()(0) == ResponseOptimization::Condition::None);

    ResponseOptimization response_optimization_2(&neural_network_2);
    EXPECT_EQ(response_optimization_2.get_inputs_conditions()(0) == ResponseOptimization::Condition::None);
    EXPECT_EQ(response_optimization_2.get_inputs_conditions()(1) == ResponseOptimization::Condition::None);
    EXPECT_EQ(response_optimization_2.get_outputs_conditions()(0) == ResponseOptimization::Condition::None);
    EXPECT_EQ(response_optimization_2.get_outputs_conditions()(1) == ResponseOptimization::Condition::None);

    ResponseOptimization response_optimization_3;
}


void ResponseOptimizationTest::test_calculate_inputs()
{
    ResponseOptimization response_optimization(&neural_network, &data_set);

    Tensor<type,2> inputs = response_optimization.calculate_inputs();

    EXPECT_EQ(inputs.dimension(0) == response_optimization.get_evaluations_number());
    EXPECT_EQ(inputs.dimension(1) == neural_network.get_inputs_number());

    EXPECT_EQ(inputs(0) <= response_optimization.get_inputs_maximums()(1));
    EXPECT_EQ(inputs(1) <= response_optimization.get_inputs_maximums()(1));
    EXPECT_EQ(inputs(0) >= response_optimization.get_inputs_minimums()(1));
    EXPECT_EQ(inputs(1) >= response_optimization.get_inputs_minimums()(1));
}


void ResponseOptimizationTest::test_perform_optimization()
{
    ResponseOptimization response_optimization(&neural_network,&data_set);

    // Empty results

    conditions_values.resize(1);
    conditions_values.setValues({100000});
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::GreaterEqualTo,conditions_values);

    ResponseOptimizationResults* results = response_optimization.perform_optimization();
    EXPECT_EQ(results->optimal_variables.size() == 0);

    // Trivial case 1

    conditions_values.resize(1);
    conditions_values.setValues({1});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_output_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::GreaterEqualTo,conditions_values);

    results = response_optimization.perform_optimization();
    EXPECT_EQ(results->optimal_variables(0) = 1);
    EXPECT_EQ(results->optimal_variables(1) <= 1);
    EXPECT_EQ(results->optimal_variables(2) >= 1);

    // Trivial case 2

    conditions_values.resize(1);
    conditions_values.setValues({1});

    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_output_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);

    conditions_values.resize(2);
    conditions_values.setValues({1,2.5});
    response_optimization.set_output_condition(0,ResponseOptimization::Condition::Between,conditions_values);

    results = response_optimization.perform_optimization();
    EXPECT_EQ(results->optimal_variables(0) = 1);
    EXPECT_EQ(results->optimal_variables(1) <= 1);
    EXPECT_EQ(1 <= results->optimal_variables(2) <= 2.5);

    // Multiple outputs case 1

    response_optimization.set(&neural_network_2);

    conditions_values.resize(1);
    conditions_values.setValues({1});
    response_optimization.set_input_condition(0,ResponseOptimization::Condition::EqualTo,conditions_values);
    response_optimization.set_input_condition(1,ResponseOptimization::Condition::LessEqualTo,conditions_values);

    results = response_optimization.perform_optimization();
    EXPECT_EQ(results->optimal_variables(0) = 1);
    EXPECT_EQ(results->optimal_variables(1) <= 1);
    EXPECT_EQ(1 <= results->optimal_variables(2) <= 3.0);
    EXPECT_EQ(-1 <= results->optimal_variables(3) <= type(1));

    // Multiple outputs case 2

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
    EXPECT_EQ(results->optimal_variables(0) = 1);
    EXPECT_EQ(results->optimal_variables(1) <= 1);
    EXPECT_EQ(1 <= results->optimal_variables(2) <= 2.0);
    EXPECT_EQ(type(-1) <= results->optimal_variables(3));
    EXPECT_EQ(results->optimal_variables(3) <= type(0));

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
    EXPECT_EQ(results->optimal_variables(0) >= 0.5);

    EXPECT_EQ(results->optimal_variables(1) >= 0.5);
    EXPECT_EQ(results->optimal_variables(2) >= 0.0);
    EXPECT_EQ(results->optimal_variables(2) <= 3.0);
    EXPECT_EQ(results->optimal_variables(3) >= type(-1));
    EXPECT_EQ(results->optimal_variables(3) <= 0.0);
}

*/
