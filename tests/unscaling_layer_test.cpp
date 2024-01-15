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

    assert_true(unscaling_layer_1.get_type() == Layer::Type::Unscaling, LOG);
    assert_true(unscaling_layer_1.get_descriptives().size() == 0, LOG);

    // Test

    UnscalingLayer unscaling_layer_2(3);

    assert_true(unscaling_layer_2.get_descriptives().size() == 3, LOG);

    // Test

    descriptives.resize(2);

    UnscalingLayer unscaling_layer_3(descriptives);

    assert_true(unscaling_layer_3.get_descriptives().size() == 2, LOG);
}


void UnscalingLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    UnscalingLayer* unscaling_layer = new UnscalingLayer;
    delete unscaling_layer;
}


void UnscalingLayerTest::test_set()
{
    cout << "test_set\n";

    // Test

    unscaling_layer.set();

    assert_true(unscaling_layer.get_descriptives().size() == 0, LOG);

    descriptives.resize(4);
    unscaling_layer.set(4);
    unscaling_layer.set_descriptives(descriptives);

    assert_true(unscaling_layer.get_descriptives().size() == 4, LOG);

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
}


void UnscalingLayerTest::test_set_inputs_number()
{
    cout << "test_set_inputs_number\n";

    Index new_inputs_number(0);
    unscaling_layer.set_inputs_number(new_inputs_number);

    assert_true(unscaling_layer.get_descriptives().size() == 0, LOG);

    Index new_inputs_number_ = 4;
    unscaling_layer.set_inputs_number(new_inputs_number_);

    assert_true(unscaling_layer.get_descriptives().size()== new_inputs_number_, LOG);
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

    assert_true(unscaling_layer.get_type() == Layer::Type::Unscaling, LOG);
    assert_true(unscaling_layer.get_type() == Layer::Type::Unscaling, LOG);

    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);
}


void UnscalingLayerTest::test_set_descriptives()
{
    cout << "test_set_descriptives\n";

    Descriptives item_descriptives(type(1), type(1), type(1), type(0));
    Descriptives des_1(type(2), type(2), type(2), type(0));

    // Test

    descriptives.resize(1);

    unscaling_layer.set_descriptives(descriptives);

    assert_true(abs(unscaling_layer.get_descriptives()[0].maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].minimum + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].mean) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    item_descriptives.set(type(1), type(1), type(1), type(0));

    descriptives.resize(1);
    descriptives.setValues({item_descriptives});

    unscaling_layer.set_descriptives(descriptives);

    assert_true(abs(unscaling_layer.get_descriptives()[0].minimum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].mean - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);
}


void UnscalingLayerTest::test_set_item_descriptives()
{
    cout << "test_set_item_descriptives\n";

    Descriptives item_descriptives;

    // Test

    unscaling_layer.set_item_descriptives(0, item_descriptives);

    assert_true(abs(unscaling_layer.get_descriptives()[0].maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].minimum + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].mean) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    unscaling_layer.set(2);

    // Test

    Descriptives des_0(type(1), type(1), type(1), type(0));
    Descriptives des_1(type(2), type(2), type(2), type(0));

    unscaling_layer.set_item_descriptives(0,des_0);
    unscaling_layer.set_item_descriptives(1,des_1);

    assert_true(abs(unscaling_layer.get_descriptives()[0].maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[0].mean - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[1].mean - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(unscaling_layer.get_descriptives()[1].standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);

    //   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    //   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    //   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,1) - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    //   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,3)) < type(NUMERIC_LIMITS_MIN), LOG);
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

    unscaling_layer.set_scalers(Scaler::Logarithm);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "logarithmic", LOG);

}


void UnscalingLayerTest::test_is_empty()
{
    cout << "test_is_empty\n";

    unscaling_layer.set();

    // Test

    assert_true(unscaling_layer.is_empty(), LOG);

    // Test

    unscaling_layer.set(1);

    assert_true(!unscaling_layer.is_empty(), LOG);
}


void UnscalingLayerTest::run_test_case()
{
    cout << "Running unscaling layer test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set();
    test_set_inputs_number();
    test_set_neurons_number();
    test_set_default();

    // Output variables descriptives

    test_set_descriptives();
    test_set_item_descriptives();

    // Check methods

    test_is_empty();

    cout << "End of unscaling layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
