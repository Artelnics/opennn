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
    scaling_layer.set_display(false);

}


ScalingLayerTest::~ScalingLayerTest()
{
}


void ScalingLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    ScalingLayer scaling_layer_1;

    assert_true(scaling_layer_1.get_type() == Layer::Type::Scaling, LOG);
    assert_true(scaling_layer_1.get_neurons_number() == 0, LOG);

    ScalingLayer scaling_layer_2(3);

    assert_true(scaling_layer_2.get_descriptives().size() == 3, LOG);
    assert_true(scaling_layer_2.get_scaling_methods().size() == 3, LOG);

    descriptives.resize(2);

    ScalingLayer scaling_layer_3(descriptives);

    assert_true(scaling_layer_3.get_descriptives().size() == 2, LOG);
}


void ScalingLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    ScalingLayer* scaling_layer = new ScalingLayer;
    delete scaling_layer;
}


void ScalingLayerTest::test_set()
{
    cout << "test_set\n";

    // Test

    scaling_layer.set();
    assert_true(scaling_layer.get_descriptives().size() == 0, LOG);

    descriptives.resize(4);

    scaling_layer.set(4);
    scaling_layer.set_descriptives(descriptives);

    assert_true(scaling_layer.get_descriptives().size() == 4, LOG);

    // Test

    Index new_inputs_number_ = 4;
    scaling_layer.set(new_inputs_number_);

    assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);

    // Test

    scaling_layer.set();

    Tensor<Index, 1> new_inputs_dimensions(1);
    new_inputs_dimensions.setConstant(3);
    scaling_layer.set(new_inputs_dimensions);

    assert_true(scaling_layer.get_descriptives().size()== 3, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 3, LOG);

    // Test

    scaling_layer.set();

    descriptives.resize(0);
    scaling_layer.set(descriptives);

    assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

    // Test

    descriptives.resize(4);
    scaling_layer.set(descriptives);

    assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
}


void ScalingLayerTest::test_set_inputs_number()
{
    cout << "test_set_inputs_number\n";

    Index new_inputs_number(0);
    scaling_layer.set_inputs_number(new_inputs_number);

    assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

    Index new_inputs_number_ = 4;
    scaling_layer.set_inputs_number(new_inputs_number_);

    assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
}


void ScalingLayerTest::test_set_neurons_number()
{
    cout << "test_set_neurons_number\n";

    Index new_inputs_number(0);
    scaling_layer.set_neurons_number(new_inputs_number);

    assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

    Index new_inputs_number_ = 4;
    scaling_layer.set_neurons_number(new_inputs_number_);

    assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
    assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
}


void ScalingLayerTest::test_set_default()
{
    cout << "test_set_default\n";

    scaling_layer.set_default();

    Tensor<Descriptives, 1> sl_descriptives = scaling_layer.get_descriptives();

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_display(), LOG);
    assert_true(scaling_layer.get_type() == Layer::Type::Scaling, LOG);
    assert_true(abs(sl_descriptives(0).minimum + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(sl_descriptives(0).maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(sl_descriptives(0).mean) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(sl_descriptives(0).standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ScalingLayerTest::test_set_descriptives()
{
    cout << "test_set_descriptives\n";

    Descriptives item_0(type(1), type(1), type(1), type(0));
    Descriptives item_1(type(2), type(2), type(2), type(0));

    // Test

    descriptives.resize(1);
    descriptives.setValues({item_0});

    scaling_layer.set(1);
    scaling_layer.set_descriptives(descriptives);

    assert_true(abs(scaling_layer.get_descriptives(0).minimum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(0).maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(0).mean - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(0).standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    descriptives.resize(2);
    descriptives.setValues({item_0, item_1});

    scaling_layer.set(2);
    scaling_layer.set_descriptives(descriptives);

    assert_true(abs(scaling_layer.get_descriptives(1).minimum - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(1).maximum - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(1).mean - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(1).standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ScalingLayerTest::test_set_item_descriptives()
{
    cout << "test_set_item_descriptives\n";

    Descriptives item_descriptives_1(type(1), type(1), type(1), type(0));
    Descriptives item_descriptives_2(type(2), type(2), type(2), type(0));

    // Test

    scaling_layer.set(1);

    scaling_layer.set_item_descriptives(0, item_descriptives_1);

    assert_true(abs(scaling_layer.get_descriptives(0).minimum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(0).maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(0).mean - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(0).standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    scaling_layer.set(2);

    scaling_layer.set_item_descriptives(0, item_descriptives_1);
    scaling_layer.set_item_descriptives(1, item_descriptives_1);

    assert_true(abs(scaling_layer.get_descriptives(1).minimum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(1).maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(1).mean - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaling_layer.get_descriptives(1).standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ScalingLayerTest::test_set_scaling_method()
{
    cout << "test_set_scaling_method\n";

    scaling_layer.set(4);

    // Test

    Tensor<Scaler, 1> method_tensor_1(4);
    method_tensor_1.setValues({Scaler::NoScaling,
                               Scaler::MinimumMaximum,
                               Scaler::MeanStandardDeviation,
                               Scaler::StandardDeviation});

    scaling_layer.set_scalers(method_tensor_1);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    assert_true(scaling_layer.get_scaling_methods()(1) == Scaler::MinimumMaximum, LOG);
    assert_true(scaling_layer.get_scaling_methods()(2) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(3) == Scaler::StandardDeviation, LOG);

    // Test

    Tensor<string, 1> method_tensor_2(4);
    method_tensor_2.setValues({"NoScaling",
                               "MinimumMaximum",
                               "MeanStandardDeviation",
                               "StandardDeviation"});

    scaling_layer.set_scalers(method_tensor_2);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    assert_true(scaling_layer.get_scaling_methods()(1) == Scaler::MinimumMaximum, LOG);
    assert_true(scaling_layer.get_scaling_methods()(2) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(3) == Scaler::StandardDeviation, LOG);

    // Test

    string no_scaling = "NoScaling";
    string minimum_maximum = "MinimumMaximum";
    string mean_standard_deviation = "MeanStandardDeviation";
    string standard_deviation = "StandardDeviation";

    scaling_layer.set_scalers(no_scaling);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    scaling_layer.set_scalers(minimum_maximum);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MinimumMaximum, LOG);
    scaling_layer.set_scalers(mean_standard_deviation);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MeanStandardDeviation, LOG);
    scaling_layer.set_scalers(standard_deviation);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::StandardDeviation, LOG);

    // Test

    Scaler no_scaling_4 = Scaler::NoScaling;
    Scaler minimum_maximum_4 = Scaler::MinimumMaximum;
    Scaler mean_standard_deviation_4 = Scaler::MeanStandardDeviation;
    Scaler standard_deviation_4 = Scaler::StandardDeviation;

    scaling_layer.set_scalers(no_scaling_4);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);

    scaling_layer.set_scalers(minimum_maximum_4);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MinimumMaximum, LOG);

    scaling_layer.set_scalers(mean_standard_deviation_4);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MeanStandardDeviation, LOG);

    scaling_layer.set_scalers(standard_deviation_4);
    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::StandardDeviation, LOG);
}


void ScalingLayerTest::test_is_empty()
{
    cout << "test_is_empty\n";

    scaling_layer.set();

    assert_true(scaling_layer.is_empty(), LOG);
}


void ScalingLayerTest::test_check_range()
{
    cout << "test_check_range\n";

    Tensor<type, 1> inputs;

    // Test

    scaling_layer.set(1);

    inputs.resize(1);
    inputs.setConstant(type(0));
    scaling_layer.check_range(inputs);

    // Test

    Tensor<Descriptives, 1> descriptives(1);
    Descriptives des(type(-1), type(1), type(1), type(0));
    descriptives.setValues({des});

    scaling_layer.set_descriptives(descriptives);
    scaling_layer.check_range(inputs);
}


void ScalingLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    DataSet data_set;
    Tensor<type, 2> data;
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> inputs_dimensions;
    Tensor<Index, 1> outputs_dimensions;

    Tensor<Descriptives,1> input_descriptives;

    // Test

    inputs_number = 1;
    samples_number = 1;
    bool is_training = true;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::NoScaling);

    inputs.resize(samples_number, inputs_number);
    inputs.setZero();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    Tensor<type*, 1> inputs_data(1);
    inputs_data(0) = inputs.data();

    scaling_layer.forward_propagate(inputs_data,
                                    inputs_dimensions,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = TensorMap<Tensor<type, 1>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);
/*
    assert_true(outputs.dimension(0) == samples_number, LOG);
//    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 3 + rand()%10;
    samples_number = 1;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::NoScaling);

    inputs.resize(samples_number, inputs_number);
    inputs.setZero();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);
    scaling_layer.forward_propagate(inputs_data, inputs_dimensions, &scaling_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type, 1>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);


    assert_true(outputs.dimension(0) == samples_number, LOG);
//    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1) - inputs(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(2) - inputs(2)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 1;
    samples_number = 1;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MinimumMaximum);

    inputs.resize(samples_number,inputs_number);
    inputs.setRandom();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);
    scaling_layer.forward_propagate(inputs_data, inputs_dimensions, &scaling_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type, 1>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);


    assert_true(outputs.dimension(0) == samples_number, LOG);
//    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 3;
    samples_number = 3;

    data.resize(samples_number,inputs_number + 1);
    data.setValues({{type(1),type(1),type(1),type(1)},
                    {type(2),type(2),type(2),type(2)},
                    {type(3),type(3),type(3),type(3)}});
    data_set.set_data(data);
    data_set.set_training();

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MinimumMaximum);

    input_descriptives = data_set.calculate_input_variables_descriptives();
    scaling_layer.set_descriptives(input_descriptives);

    inputs = data_set.get_input_data();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);
    scaling_layer.forward_propagate(inputs_data, inputs_dimensions, &scaling_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type,1>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0,0) - static_cast<type>(-1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1,0) - static_cast<type>(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(2,0) - static_cast<type>(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 2;
    samples_number = 2;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MeanStandardDeviation);

    data.resize(samples_number,inputs_number + 1);
    data.setValues({{type(0),type(0)},{type(2),type(2)}});
    data_set.set_data(data);

    input_descriptives = data_set.calculate_input_variables_descriptives();
    scaling_layer.set_descriptives(input_descriptives);

    inputs = data_set.get_input_data();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);
    scaling_layer.forward_propagate(inputs_data, inputs_dimensions, &scaling_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type,1>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    inputs_number = 1;
    samples_number = 1;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::StandardDeviation);

    data.resize(samples_number, inputs_number + 1);
    data.setConstant(type(1));
    data_set.set(data);

    scaling_layer.set_descriptives(data_set.calculate_input_variables_descriptives());

    inputs = data_set.get_input_data();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);
    scaling_layer.forward_propagate(inputs_data, inputs_dimensions, &scaling_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type,2>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);

    assert_true(outputs.dimension(0) == inputs_number, LOG);
    assert_true(outputs.dimension(1) == samples_number, LOG);

    assert_true(abs(outputs(0) - data(0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test


    inputs_number = 2 + rand()%10;
    samples_number = 1;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::StandardDeviation);

    data.resize(samples_number, inputs_number + 1);
    data.setConstant(type(1));
    data_set.set_data(data);

    scaling_layer.set_descriptives(data_set.calculate_input_variables_descriptives());

    inputs = data_set.get_input_data();
    inputs_dimensions = get_dimensions(inputs);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);
    scaling_layer.forward_propagate(inputs, inputs_dimensions, &scaling_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type,1>>(scaling_layer_forward_propagation.outputs_data(0),
                                         scaling_layer_forward_propagation.outputs_dimensions);

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0,0) - static_cast<type>(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1,0) - static_cast<type>(1)) < type(NUMERIC_LIMITS_MIN), LOG);
*/
}


void ScalingLayerTest::run_test_case()
{
    cout << "Running scaling layer test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set();
    test_set_inputs_number();
    test_set_neurons_number();
    test_set_default();

    // Input variables descriptives

    test_set_descriptives();
    test_set_item_descriptives();

    // Variables scaling and unscaling

    test_set_scaling_method();

    // Input range

    test_is_empty();

    test_check_range();

    // Scaling and unscaling

    test_forward_propagate();

    cout << "End of scaling layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
