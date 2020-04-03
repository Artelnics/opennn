//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_test.h"


ScalingLayerTest::ScalingLayerTest() : UnitTesting()
{
}


ScalingLayerTest::~ScalingLayerTest()
{
}


void ScalingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   ScalingLayer sl1;

   assert_true(sl1.get_type() == Layer::Scaling, LOG);
   assert_true(sl1.get_descriptives().size() == 0, LOG);

   ScalingLayer sl2(3);

   assert_true(sl2.get_descriptives().size() == 3, LOG);
   assert_true(sl2.get_scaling_methods().size() == 3, LOG);

   Tensor<Descriptives, 1> descriptives;

   descriptives.resize(2);

   ScalingLayer sl3(descriptives);

   assert_true(sl3.get_descriptives().size() == 2, LOG);

   ScalingLayer sl4(sl1);

   assert_true(sl4.get_type() == Layer::Scaling, LOG);
}

void ScalingLayerTest::test_destructor()
{
   cout << "test_destructor\n";

}



void ScalingLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   ScalingLayer sl_1;
   ScalingLayer sl_2 = sl_1;

   assert_true(sl_2.get_inputs_number() == 0, LOG);
   assert_true(sl_2.get_neurons_number() == 0, LOG);
}

void ScalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   ScalingLayer sl1;

   // Test 0

   assert_true(sl1.get_neurons_number() == 0, LOG);
   assert_true(sl1.get_neurons_number() == sl1.get_inputs_number(), LOG);

   // Test 1

   sl1.set(3);

   Tensor<Descriptives, 1> descriptives(3);
   sl1.set_descriptives(descriptives);

   assert_true(sl1.get_neurons_number() == 3, LOG);
   assert_true(sl1.get_neurons_number() == sl1.get_inputs_number(), LOG);
}

void ScalingLayerTest::test_get_inputs_number()
{
   cout << "test_get_neurons_number\n";

   ScalingLayer sl1;

   // Test 0

   assert_true(sl1.get_inputs_number() == 0, LOG);
   assert_true(sl1.get_inputs_number() == sl1.get_neurons_number(), LOG);

   // Test 1

   sl1.set(3);

   Tensor<Descriptives, 1> descriptives(3);
   sl1.set_descriptives(descriptives);

   assert_true(sl1.get_inputs_number() == 3, LOG);
   assert_true(sl1.get_inputs_number() == sl1.get_neurons_number(), LOG);
}



void ScalingLayerTest::test_get_descriptives()
{
   cout << "test_get_descriptives\n";

   ScalingLayer sl;


}

void ScalingLayerTest::test_get_descriptives_matrix()
{
   cout << "test_get_descriptives_matrix\n";

   ScalingLayer sl;
}

void ScalingLayerTest::test_get_means()
{
   cout << "test_get_means\n";

   ScalingLayer sl;

   Tensor<type, 1> means;

   // Test

   sl.set();

//   assert_true(sl.get_means().dimension(0) == 0, LOG);

   // Test

   sl.set(1);
   sl.set_mean(0, 2.0);

   means = sl.get_means();

   assert_true(means.size() == 1, LOG);
   assert_true(means(0) == 2.0, LOG);
}

void ScalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";

   ScalingLayer sl1;

//   assert_true(sl1.get_minimums().empty(), LOG);

   sl1.set(1);

   sl1.set_minimum(0, 2.0);

   assert_true(sl1.get_minimums()(0) == 2.0, LOG);
}

void ScalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";

   ScalingLayer sl1;

//   assert_true(sl1.get_maximums().empty(), LOG);

   sl1.set(1);

   sl1.set_maximum(0, 2.0);

   assert_true(sl1.get_maximums()(0) == 2.0, LOG);
}

void ScalingLayerTest::test_get_standard_deviations()
{
    cout << "test_get_standard_deviations\n";

    ScalingLayer sl;

//    assert_true(sl.get_standard_deviations(), LOG);

    sl.set(1);

    sl.set_standard_deviation(0, 3.0);

    assert_true(sl.get_standard_deviations()(0) == 3.0, LOG);
}

void ScalingLayerTest::test_get_scaling_method()
{
   cout << "test_get_scaling_method\n";

   ScalingLayer sl(1);

   // Test

   sl.set_scaling_methods(ScalingLayer::MeanStandardDeviation);

   assert_true(sl.get_scaling_methods()[0] == ScalingLayer::MeanStandardDeviation, LOG);

   // Test

   sl.set_scaling_methods(ScalingLayer::MinimumMaximum);

   assert_true(sl.get_scaling_methods()[0] == ScalingLayer::MinimumMaximum, LOG);
}

void ScalingLayerTest::test_get_scaling_method_name()
{
   cout << "test_get_scaling_method_name\n";
}



void ScalingLayerTest::test_get_display_inputs_warning()
{
   cout << "test_get_display_inputs_warning\n";
}

void ScalingLayerTest::test_get_display()
{
   cout << "test_get_display\n";

   ScalingLayer sl;

   assert_true(sl.get_display(), LOG);
}



void ScalingLayerTest::test_set()
{
   cout << "test_set\n";

   ScalingLayer sl;

   // Test 1

   sl.set();

   assert_true(sl.get_descriptives().size() == 0, LOG);

   Tensor<Descriptives, 1> descriptives(4);
   sl.set_descriptives(descriptives);
   sl.set();

   assert_true(sl.get_descriptives().size() == 0, LOG);

   // Test 2

   sl.set();

   Index new_inputs_number;
   sl.set(new_inputs_number);

   assert_true(sl.get_descriptives().size()== 0, LOG);
   assert_true(sl.get_scaling_methods().size()== 0, LOG);

   Index new_inputs_number_ = 4;
   sl.set(new_inputs_number_);

   assert_true(sl.get_descriptives().size()== 4, LOG);
   assert_true(sl.get_scaling_methods().size()== 4, LOG);

   // Test 3

   sl.set();

   Tensor<Index, 1> new_inputs_dimensions;
   sl.set(new_inputs_dimensions);

   assert_true(sl.get_descriptives().size()== 1, LOG);
   assert_true(sl.get_scaling_methods().size()== 1, LOG);
   assert_true(sl.get_input_variables_dimensions().size() == 0, LOG);

   Tensor<Index, 1> new_inputs_dimensions_(1);
   new_inputs_dimensions_.setConstant(3);
   sl.set(new_inputs_dimensions_);

   assert_true(sl.get_descriptives().size()== 3, LOG);
   assert_true(sl.get_scaling_methods().size()== 3, LOG);
   assert_true(sl.get_input_variables_dimensions().size() == 1, LOG);

   // Test 4

   sl.set();

   Tensor<Descriptives, 1> descriptives_4;
   sl.set(descriptives_4);

   assert_true(sl.get_descriptives().size()== 0, LOG);
   assert_true(sl.get_scaling_methods().size()== 0, LOG);

   Tensor<Descriptives, 1> descriptives_4_(4);
   sl.set(descriptives_4_);

   assert_true(sl.get_descriptives().size()== 4, LOG);
   assert_true(sl.get_scaling_methods().size()== 4, LOG);

   // Test 5

   sl.set();

   ScalingLayer sl5;
   sl5.set(7);

   sl.set(sl5);

   assert_true(sl.get_descriptives().size() == 7, LOG);
   assert_true(sl.get_scaling_methods().size() == 7, LOG);
}

void ScalingLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

   ScalingLayer sl;
/*
   Index new_inputs_number;
   sl.set_inputs_number(new_inputs_number);

   assert_true(sl.get_descriptives().size()== 0, LOG);
   assert_true(sl.get_scaling_methods().size()== 0, LOG);
*/
   Index new_inputs_number_ = 4;
   sl.set_inputs_number(new_inputs_number_);

   assert_true(sl.get_descriptives().size()== 4, LOG);
   assert_true(sl.get_scaling_methods().size()== 4, LOG);
}

void ScalingLayerTest::test_set_neurons_number()
{
   cout << "test_set_neurons_number\n";

   ScalingLayer sl;
/*
   Index new_inputs_number;
   sl.set_neurons_number(new_inputs_number);

   assert_true(sl.get_descriptives().size()== 0, LOG);
   assert_true(sl.get_scaling_methods().size()== 0, LOG);
    */
   Index new_inputs_number_ = 4;
   sl.set_neurons_number(new_inputs_number_);

   assert_true(sl.get_descriptives().size()== 4, LOG);
   assert_true(sl.get_scaling_methods().size()== 4, LOG);
}

void ScalingLayerTest::test_set_default()
{
   cout << "test_set_default\n";

   ScalingLayer sl;

   assert_true( sl.get_type() == Layer::Scaling, LOG);
   assert_true( sl.get_type() == 0, LOG);
}



void ScalingLayerTest::test_set_descriptives()
{
   cout << "test_set_descriptives\n";

   ScalingLayer sl;

   // Test 0

   Tensor<Descriptives, 1> descriptives(1);

   sl.set_descriptives(descriptives);

   assert_true(sl.get_descriptives_matrix()(0,0) + 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,1) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,2) - 0 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,3) - 1 < static_cast<type>(1e-3), LOG);

   // Test 1

   Descriptives des_0(1,1,1,0);
   Descriptives des_1(2,2,2,0);

   descriptives.resize(2);
   descriptives.setValues({des_0,des_1});

   sl.set_descriptives(descriptives);

   assert_true(sl.get_descriptives_matrix()(0,0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,2) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,1) - 2 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,3) - 0 < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_set_descriptives_eigen()
{
   cout << "test_set_descriptives_eigen\n";

   ScalingLayer sl(1);

   // Test 0

   Tensor<type, 2> descriptives_eigen(1,4);

   sl.set_descriptives_eigen(descriptives_eigen);
/*
   assert_true(sl.get_descriptives_matrix()(0,0) + 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,1) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,2) - 0 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,3) - 1 < static_cast<type>(1e-3), LOG);
*/
   // Test 1

   ScalingLayer sl_(2);

   Tensor<type, 2> descriptives_eigen_(2,4);
   descriptives_eigen_.setValues({{1,1,1,0},{2,2,2,0}});

   sl_.set_descriptives_eigen(descriptives_eigen_);

   assert_true(sl_.get_descriptives_matrix()(0,0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl_.get_descriptives_matrix()(0,2) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl_.get_descriptives_matrix()(1,1) - 2 < static_cast<type>(1e-3), LOG);
   assert_true(sl_.get_descriptives_matrix()(1,3) - 0 < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_set_item_descriptives()
{
   cout << "test_set_item_descriptives\n";

   ScalingLayer sl(2);

   // Test

   Descriptives des_0(1,1,1,0);
   Descriptives des_1(2,2,2,0);

   sl.set_item_descriptives(0,des_0);
   sl.set_item_descriptives(1,des_1);

   assert_true(sl.get_descriptives_matrix()(0,0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(0,2) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,1) - 2 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,3) - 0 < static_cast<type>(1e-3), LOG);

}



void ScalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";

   ScalingLayer sl(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   sl.set_descriptives(descriptives);

   sl.set_minimum(0, -5);
   sl.set_minimum(1, -6);

   assert_true(sl.get_descriptives_matrix()(0,0) + 5 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,0) + 6 < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";

   ScalingLayer sl(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   sl.set_descriptives(descriptives);

   sl.set_maximum(0, 5);
   sl.set_maximum(1, 6);

   assert_true(sl.get_descriptives_matrix()(0,0) - 5 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,0) - 6 < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";

   ScalingLayer sl(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   sl.set_descriptives(descriptives);

   sl.set_mean(0, 5);
   sl.set_mean(1, 6);

   assert_true(sl.get_descriptives_matrix()(0,0) - 5 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,0) - 6 < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   ScalingLayer sl(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   sl.set_descriptives(descriptives);

   sl.set_standard_deviation(0, 5);
   sl.set_standard_deviation(1, 6);

   assert_true(sl.get_descriptives_matrix()(0,0) - 5 < static_cast<type>(1e-3), LOG);
   assert_true(sl.get_descriptives_matrix()(1,0) - 6 < static_cast<type>(1e-3), LOG);
}




void ScalingLayerTest::test_set_minimums()
{
   cout << "test_set_minimums\n";
}

void ScalingLayerTest::test_set_maximums()
{
   cout << "test_set_maximums\n";
}

void ScalingLayerTest::test_set_means()
{
   cout << "test_set_means\n";
}

void ScalingLayerTest::test_set_standard_deviations()
{
   cout << "test_set_standard_deviations\n";
}




void ScalingLayerTest::test_set_statistics()
{
   cout << "test_set_statistics\n";
}

void ScalingLayerTest::test_set_scaling_method()
{
   cout << "test_set_scaling_method\n";

    ScalingLayer sl(4);

    // Test 1

    Tensor<ScalingLayer::ScalingMethod, 1> method_tensor_1(4);
    method_tensor_1.setValues({ScalingLayer::ScalingMethod::NoScaling,
                                ScalingLayer::ScalingMethod::MinimumMaximum,
                                ScalingLayer::ScalingMethod::MeanStandardDeviation,
                                ScalingLayer::ScalingMethod::StandardDeviation});

    sl.set_scaling_methods(method_tensor_1);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::NoScaling, LOG);
    assert_true(sl.get_scaling_methods()(0) == 0, LOG);

    assert_true(sl.get_scaling_methods()(1) == ScalingLayer::ScalingMethod::MinimumMaximum, LOG);
    assert_true(sl.get_scaling_methods()(1) == 1, LOG);

    assert_true(sl.get_scaling_methods()(2) == ScalingLayer::ScalingMethod::MeanStandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(2) == 2, LOG);

    assert_true(sl.get_scaling_methods()(3) == ScalingLayer::ScalingMethod::StandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(3) == 3, LOG);

    // Test 2

    Tensor<string, 1> method_tensor_2(4);
    method_tensor_2.setValues({"NoScaling",
                                "MinimumMaximum",
                                "MeanStandardDeviation",
                                "StandardDeviation"});

    sl.set_scaling_methods(method_tensor_2);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::NoScaling, LOG);
    assert_true(sl.get_scaling_methods()(0) == 0, LOG);

    assert_true(sl.get_scaling_methods()(1) == ScalingLayer::ScalingMethod::MinimumMaximum, LOG);
    assert_true(sl.get_scaling_methods()(1) == 1, LOG);

    assert_true(sl.get_scaling_methods()(2) == ScalingLayer::ScalingMethod::MeanStandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(2) == 2, LOG);

    assert_true(sl.get_scaling_methods()(3) == ScalingLayer::ScalingMethod::StandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(3) == 3, LOG);

    // Test 3

    string no_scaling = "NoScaling";
    string minimum_maximum = "MinimumMaximum";
    string mean_standard_deviation = "MeanStandardDeviation";
    string standard_deviation = "StandardDeviation";

    sl.set_scaling_methods(no_scaling);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::NoScaling, LOG);
    assert_true(sl.get_scaling_methods()(0) == 0, LOG);

    sl.set_scaling_methods(minimum_maximum);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::MinimumMaximum, LOG);
    assert_true(sl.get_scaling_methods()(0) == 1, LOG);

    sl.set_scaling_methods(mean_standard_deviation);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::MeanStandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(0) == 2, LOG);

    sl.set_scaling_methods(standard_deviation);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::StandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(0) == 3, LOG);

    // Test 4

    ScalingLayer::ScalingMethod no_scaling_4 = ScalingLayer::ScalingMethod::NoScaling;

    ScalingLayer::ScalingMethod minimum_maximum_4 = ScalingLayer::ScalingMethod::MinimumMaximum;

    ScalingLayer::ScalingMethod mean_standard_deviation_4 = ScalingLayer::ScalingMethod::MeanStandardDeviation;

    ScalingLayer::ScalingMethod standard_deviation_4 = ScalingLayer::ScalingMethod::StandardDeviation;

    sl.set_scaling_methods(no_scaling_4);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::NoScaling, LOG);
    assert_true(sl.get_scaling_methods()(0) == 0, LOG);

    sl.set_scaling_methods(minimum_maximum_4);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::MinimumMaximum, LOG);
    assert_true(sl.get_scaling_methods()(0) == 1, LOG);

    sl.set_scaling_methods(mean_standard_deviation_4);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::MeanStandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(0) == 2, LOG);

    sl.set_scaling_methods(standard_deviation_4);

    assert_true(sl.get_scaling_methods()(0) == ScalingLayer::ScalingMethod::StandardDeviation, LOG);
    assert_true(sl.get_scaling_methods()(0) == 3, LOG);
}

void ScalingLayerTest::test_set_display_inputs_warning()
{
   cout << "test_set_display_inputs_warning\n";
}

void ScalingLayerTest::test_set_display()
{
   cout << "test_set_display\n";

   bool display_true = true;
   bool display_false = false;

   set_display(display_true);
   assert_true(get_display() == true, LOG);

   set_display(display_false);
   assert_true(get_display() == false, LOG);
}




void ScalingLayerTest::test_check_range()
{
   cout << "test_check_range\n";

   ScalingLayer sl;
   Tensor<type, 1> inputs;

   // Test

   sl.set(1);

   inputs.resize(1);
   inputs.setConstant(0.0);
   sl.check_range(inputs);

}

void ScalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   ScalingLayer scaling_layer;

   Tensor<type, 2> inputs;

   scaling_layer.set_display(false);

   // Test

   scaling_layer.set_scaling_methods(ScalingLayer::MinimumMaximum);

   scaling_layer.set(1);

   inputs.resize(1,1);
   inputs.setConstant(0.0);

   Tensor<type, 2> outputs = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs(0) == inputs(0), LOG);

   // Test

   scaling_layer.set_scaling_methods(ScalingLayer::MeanStandardDeviation);

   scaling_layer.set(1);

   inputs.resize(1,1);
   inputs.setConstant(0.0);

   outputs.setConstant(0.0);

   assert_true(scaling_layer.calculate_outputs(inputs)(0) == inputs(0), LOG);
}

void ScalingLayerTest::test_calculate_minimum_maximum_output()
{
   cout << "test_calculate_minimum_maximum_output\n";
}

void ScalingLayerTest::test_calculate_mean_standard_deviation_output()
{
   cout << "test_calculate_mean_standard_deviation_output\n";
}

void ScalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   ScalingLayer sl;

   Tensor<string, 1> inputs_names;
   Tensor<string, 1> outputs_names;

   string expression;

   // Test

 /*  sl.set(1);
   inputs_names.set(1, "x");
   outputs_names.set(1, "y");

   expression = sl.write_expression(inputs_names, outputs_names);

   assert_true(expression.empty() == false, LOG);
*/
}

void ScalingLayerTest::test_to_XML()
{
   cout << "test_to_XML\n";

   ScalingLayer  sl;

   tinyxml2::XMLDocument* sld;

   // Test

   sld = sl.to_XML();

   assert_true(sld != nullptr, LOG);

   delete sld;
}

void ScalingLayerTest::test_from_XML()
{
   cout << "test_from_XML\n";

   ScalingLayer  sl;

   tinyxml2::XMLDocument* sld;

   // Test

   sld = sl.to_XML();

   sl.from_XML(*sld);

   delete sld;
}


void ScalingLayerTest::run_test_case()
{
   cout << "Running scaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Scaling layer architecture

   test_get_neurons_number();

   test_get_inputs_number();

   // Input variables descriptives

   test_get_descriptives();
   test_get_descriptives_matrix();

   test_get_minimums();
   test_get_maximums();
   test_get_means();
   test_get_standard_deviations();

   // Variables scaling and unscaling

   test_get_scaling_method();
   test_get_scaling_method_name();

   // Display warning

   test_get_display_inputs_warning();

   // Display messages

   test_get_display();

   // Set methods

   test_set();

   test_set_inputs_number();
   test_set_neurons_number();

   test_set_default();

   // Input variables descriptives

   test_set_descriptives();
   test_set_descriptives_eigen();
   test_set_item_descriptives();

   test_set_minimums();
   test_set_minimum();

   test_set_maximums();
   test_set_maximum();

   test_set_means();
   test_set_mean();

   test_set_standard_deviations();
   test_set_standard_deviation();

   // Variables descriptives

   test_set_statistics();

   // Variables scaling and unscaling

   test_set_scaling_method();

   // Display inputs warning

   test_set_display_inputs_warning();

   // Display messages

   test_set_display();

   // Input range

   test_check_range();

   // Scaling and unscaling

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of scaling layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
//
// This library sl free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library sl distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
