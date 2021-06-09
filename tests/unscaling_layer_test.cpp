//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   T E S T   C L A S S                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "unscaling_layer_test.h"


UnscalingLayerTest::UnscalingLayerTest() : UnitTesting()
{
}


UnscalingLayerTest::~UnscalingLayerTest()
{
}


void UnscalingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

    // Test

   UnscalingLayer unscaling_layer_1;

   assert_true(unscaling_layer_1.get_type() == Layer::Unscaling, LOG);
   assert_true(unscaling_layer_1.get_descriptives().size() == 0, LOG);

   // Test

   UnscalingLayer unscaling_layer_2(3);

   assert_true(unscaling_layer_2.get_descriptives().size() == 3, LOG);

   // Test

   descriptives.resize(2);

   UnscalingLayer unscaling_layer_3(descriptives);

   assert_true(unscaling_layer_3.get_descriptives().size() == 2, LOG);

}


void UnscalingLayerTest::test_get_dimensions()
{
   cout << "test_get_dimensions\n";

   unscaling_layer.set(1);

   // Test

   assert_true(unscaling_layer.get_neurons_number() == 1, LOG);

   // Test

   unscaling_layer.set(3);

   assert_true(unscaling_layer.get_neurons_number() == 3, LOG);
}


void UnscalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   // Test

   assert_true(unscaling_layer.get_neurons_number() == 0, LOG);
   assert_true(unscaling_layer.get_neurons_number() == unscaling_layer.get_inputs_number(), LOG);

   // Test

   unscaling_layer.set(3);

   descriptives.resize(3);
   unscaling_layer.set_descriptives(descriptives);

   assert_true(unscaling_layer.get_neurons_number() == 3, LOG);
   assert_true(unscaling_layer.get_neurons_number() == unscaling_layer.get_inputs_number(), LOG);
}


void UnscalingLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";

   // Test

   assert_true(unscaling_layer.get_inputs_number() == 0, LOG);
   assert_true(unscaling_layer.get_inputs_number() == unscaling_layer.get_neurons_number(), LOG);

   // Test

   unscaling_layer.set(3);

   descriptives.resize(3);
   unscaling_layer.set_descriptives(descriptives);

   assert_true(unscaling_layer.get_inputs_number() == 3, LOG);
   assert_true(unscaling_layer.get_inputs_number() == unscaling_layer.get_neurons_number(), LOG);
}


void UnscalingLayerTest::test_get_descriptives()
{
   cout << "test_get_descriptives\n";

    // Test 0
/*
    descriptives.resize(1);

    unscaling_layer.set_descriptives(descriptives);

    descriptives = unscaling_layer.get_descriptives();

    assert_true(descriptives.dimension(0) == 1, LOG);
    assert_true(abs(descriptives(0).minimum + 1) < numeric_limits<type>::min(), LOG);
    assert_true(abs(descriptives(0).maximum - 1) < numeric_limits<type>::min(), LOG);
    assert_true(abs(descriptives(0).mean - 0) < numeric_limits<type>::min(), LOG);
    assert_true(abs(descriptives(0).standard_deviation - 1) < numeric_limits<type>::min(), LOG);

    // Test

    Descriptives des_0(1,1,1,0);
    Descriptives des_1(2,2,2,0);

    descriptives.resize(2);
    descriptives.setValues({des_0,des_1});

    unscaling_layer.set_descriptives(descriptives);

    descriptives = unscaling_layer.get_descriptives();

    assert_true(descriptives.dimension(0) == 2, LOG);
    assert_true(abs(descriptives(1).minimum - 2) < numeric_limits<type>::min(), LOG);
    assert_true(abs(descriptives(1).maximum - 2) < numeric_limits<type>::min(), LOG);
    assert_true(abs(descriptives(1).mean - 2) < numeric_limits<type>::min(), LOG);
    assert_true(abs(descriptives(1).standard_deviation - 0) < numeric_limits<type>::min(), LOG);
*/
}


void UnscalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";

   // Test

   unscaling_layer.set(2);

   descriptives.resize(2);

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_minimums()(0) + 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(unscaling_layer.get_minimums()(1) + 1) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives(0).minimum = 1;
   descriptives(1).minimum = -1;

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_minimums()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(unscaling_layer.get_minimums()(1) + 1) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";

   // Test

   unscaling_layer.set(2);

   descriptives.resize(2);

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_maximums()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(unscaling_layer.get_maximums()(1) - 1) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives(0).maximum = 1;
   descriptives(1).maximum = -1;

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_maximums()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(unscaling_layer.get_maximums()(1) + 1) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_write_scalers()
{
    cout << "test_get_scaling_method_name\n";

    unscaling_layer.set(1);

    // Test

    Scaler minimum_maximum = Scaler::MinimumMaximum;

    Scaler mean_standard_deviation = Scaler::MeanStandardDeviation;

    Scaler logarithmic = Logarithm;

    unscaling_layer.set_scalers(NoScaling);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "NoUnscaling", LOG);

    unscaling_layer.set_scalers(minimum_maximum);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "MinimumMaximum", LOG);

    unscaling_layer.set_scalers(mean_standard_deviation);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "MeanStandardDeviation", LOG);

    unscaling_layer.set_scalers(logarithmic);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "Logarithm", LOG);

    // Test

    unscaling_layer.set_scalers(NoScaling);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "no unscaling", LOG);

    unscaling_layer.set_scalers(minimum_maximum);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);

    unscaling_layer.set_scalers(mean_standard_deviation);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "mean and standard deviation", LOG);

    unscaling_layer.set_scalers(logarithmic);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "logarithm", LOG);

}


void UnscalingLayerTest::test_set()
{
   cout << "test_set\n";
/*
   // Test

   unscaling_layer.set();

   assert_true(unscaling_layer.get_descriptives().size() == 0, LOG);

   descriptives.resize(4);
   unscaling_layer.set_descriptives(descriptives);
   unscaling_layer.set();

   assert_true(unscaling_layer.get_descriptives().size() == 0, LOG);

   // Test

   unscaling_layer.set();

   Index new_neurons_number(0);
   unscaling_layer.set(new_neurons_number);

   assert_true(unscaling_layer.get_descriptives().size()== 0, LOG);

   Index new_inputs_number_ = 4;
   unscaling_layer.set(new_inputs_number_);

   assert_true(unscaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(unscaling_layer.get_inputs_number()== unscaling_layer.get_descriptives().size(), LOG);

   // Test

   unscaling_layer.set();

   descriptives.resize(0);
   unscaling_layer.set(descriptives);

   assert_true(unscaling_layer.get_descriptives().size()== 0, LOG);

   descriptives.resize(4);
   unscaling_layer.set(descriptives);

   assert_true(unscaling_layer.get_descriptives().size()== 4, LOG);
*/
}


void UnscalingLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

//   Index new_inputs_number;
//   ul.set_inputs_number(new_inputs_number);

//   assert_true(ul.get_descriptives().size()== 0, LOG);

//   Index new_inputs_number_ = 4;
//   ul.set_inputs_number(new_inputs_number_);

//   assert_true(ul.get_descriptives().size()== 4, LOG);
}


void UnscalingLayerTest::test_set_neurons_number()
{
   cout << "test_set_inputs_number\n";

   Index new_inputs_number(0);
   unscaling_layer.set_neurons_number(new_inputs_number);

   assert_true(unscaling_layer.get_descriptives().size()== 0, LOG);

   Index new_inputs_number_ = 4;
   unscaling_layer.set_neurons_number(new_inputs_number_);

   assert_true(unscaling_layer.get_descriptives().size()== 4, LOG);
}


void UnscalingLayerTest::test_set_default()
{
   cout << "test_set_default\n";

   unscaling_layer.set(1);

   unscaling_layer.set_default();

   assert_true(unscaling_layer.get_type() == Layer::Unscaling, LOG);
   assert_true(unscaling_layer.get_type() == 7, LOG);

   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);
}


void UnscalingLayerTest::test_set_descriptives()
{
   cout << "test_set_descriptives\n";

   Descriptives item_descriptives(1,1,1,0);
   Descriptives des_1(2,2,2,0);

   // Test

   descriptives.resize(1);

   unscaling_layer.set_descriptives(descriptives);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) + 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,1) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 0) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,3) - 1) < numeric_limits<type>::min(), LOG);

   // Test

   item_descriptives.set(1,1,1,0);

   descriptives.resize(1);
   descriptives.setValues({item_descriptives});

   unscaling_layer.set_descriptives(descriptives);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 1) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_set_item_descriptives()
{
   cout << "test_set_item_descriptives\n";

   Descriptives item_descriptives;

   // Test

   unscaling_layer.set_item_descriptives(0, item_descriptives);

//   assert_true(abs(ul.get_descriptives_matrix()(0,0) + 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(ul.get_descriptives_matrix()(0,1) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(ul.get_descriptives_matrix()(0,2) - 0) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(ul.get_descriptives_matrix()(0,3) - 1) < numeric_limits<type>::min(), LOG);

   unscaling_layer.set(2);

   // Test

   Descriptives des_0(1,1,1,0);
   Descriptives des_1(2,2,2,0);

   unscaling_layer.set_item_descriptives(0,des_0);
   unscaling_layer.set_item_descriptives(1,des_1);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,1) - 2) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,3) - 0) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";

   unscaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_minimum(0, -5);
   unscaling_layer.set_minimum(1, -6);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) + 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,0) + 6) < numeric_limits<type>::min(), LOG);
}

void UnscalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";

   unscaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_maximum(0, 5);
   unscaling_layer.set_maximum(1, 6);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,1) - 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,1) - 6) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";

   unscaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_mean(0, 5);
   unscaling_layer.set_mean(1, 6);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,2) - 6) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   // Test

   unscaling_layer.set(2);

   descriptives.resize(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_standard_deviation(0, 5);
   unscaling_layer.set_standard_deviation(1, 6);

//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,3) - 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,3) - 6) < numeric_limits<type>::min(), LOG);
}


void UnscalingLayerTest::test_set_unscaling_method()
{
   cout << "test_set_unscaling_method\n";

   unscaling_layer.set(1);

   // Test

   unscaling_layer.set_scalers(Scaler::NoScaling);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "no unscaling", LOG);

   unscaling_layer.set_scalers(Scaler::MinimumMaximum);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);

   unscaling_layer.set_scalers(Scaler::MeanStandardDeviation);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "mean and standard deviation", LOG);

   unscaling_layer.set_scalers(Logarithm);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "logarithmic", LOG);

}


void UnscalingLayerTest::test_is_empty()
{
    cout << "test_is_empty\n";

    // Test

    assert_true(unscaling_layer.is_empty(), LOG);

    // Test

    unscaling_layer.set(1);

    assert_true(!unscaling_layer.is_empty(), LOG);
}


void UnscalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   Tensor<type, 2> minimums_maximums;
   Tensor<type, 2> mean_standard_deviation;

   Tensor<type, 2> standard_deviation;

   unscaling_layer.set_display(false);

   // Test 0_0

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(NoScaling);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);
   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test 0_1

   unscaling_layer.set(3);
   unscaling_layer.set_scalers(NoScaling);

   inputs.resize(1,3);
   inputs.setConstant(0);
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 3, LOG);
   assert_true(abs(outputs(0)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(2)) < numeric_limits<type>::min(), LOG);

   // Test 1_0

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(MinimumMaximum);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(outputs.dimension(1) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test 1_1

   unscaling_layer.set(2);
   unscaling_layer.set_scalers(MinimumMaximum);

   minimums_maximums.resize(2,4);
   minimums_maximums.setValues({{-1000,1000,0,0},{-100,100,0,0}});

   inputs.resize(1,2);
   inputs.setValues({{0.1f,0}});
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(outputs.dimension(1) - 2 < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(0) - static_cast<type>(100)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1)) < numeric_limits<type>::min(), LOG);

   // Test 2_0

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(MeanStandardDeviation);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(outputs.dimension(1) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test 2_1

   unscaling_layer.set(2);
   unscaling_layer.set_scalers(MeanStandardDeviation);

   mean_standard_deviation.resize(2,4);
   mean_standard_deviation.setValues({{-1,1,-1,-2},{-1,1,2,3}});

   inputs.resize(1,2);
   inputs.setValues({{-1,1}});
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(outputs.dimension(1) - 2 < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(0) - static_cast<type>(1)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(5)) < numeric_limits<type>::min(), LOG);

   // Test 3_0

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(Logarithm);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) - 1 < numeric_limits<type>::min(), LOG);
   assert_true(outputs.dimension(1) - 1 < numeric_limits<type>::min(), LOG);

   assert_true(abs(outputs(0) - 1) < numeric_limits<type>::min(), LOG);

   // Test 3_1

   unscaling_layer.set(2);
   unscaling_layer.set_scalers(Logarithm);

   standard_deviation.resize(2,4);
   standard_deviation.setValues({{-1,1,-1,2},{-1,1,1,4}});

   inputs.resize(1,2);
   inputs.setConstant(1);
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(abs(outputs.dimension(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs.dimension(1) - 2) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(0) - static_cast<type>(2.7182)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(2.7182)) < numeric_limits<type>::min(), LOG);

}


void UnscalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   Tensor<string, 1> inputs_names(1);
   Tensor<string, 1> outputs_names(1);

   string expression;

   // Test

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(NoScaling);
   inputs_names.setValues({"x"});
   outputs_names.setValues({"y"});

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = x;\n", LOG);

   // Test

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(MinimumMaximum);

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);

   assert_true(expression == "y = x*(1+1)/(1+1)-1+1*(1+1)/(1+1);\n", LOG);

   // Test

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(MeanStandardDeviation);

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = -1+0.5*(x+1)*((1)-(-1);\n", LOG);

   // Test 

   unscaling_layer.set(1);
   unscaling_layer.set_scalers(Logarithm);

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = -1+0.5*(exp(x)+1)*((1)-(-1));\n", LOG);

}


void UnscalingLayerTest::run_test_case()
{
   cout << "Running unscaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Get methods

   test_get_dimensions();

   // Unscaling layer architecture

   test_get_inputs_number();
   test_get_neurons_number();

   // Output variables descriptives

   test_get_minimums();
   test_get_maximums();

   // Variables descriptives

   test_get_descriptives();

   // Set methods

   test_set();
   test_set_inputs_number();
   test_set_neurons_number();
   test_set_default();

   // Output variables descriptives

   test_set_descriptives();
   test_set_item_descriptives();
   test_set_minimum();
   test_set_maximum();
   test_set_mean();
   test_set_standard_deviation();

   // Variables scaling and unscaling

   test_write_scalers();

   // Check methods

   test_is_empty();

   // Output methods

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   cout << "End of unscaling layer test case.\n\n";
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
