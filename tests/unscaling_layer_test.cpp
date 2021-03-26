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

   Tensor<Descriptives, 1> descriptives;

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

void UnscalingLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void UnscalingLayerTest::test_get_dimensions()
{
   cout << "test_get_dimensions\n";

   UnscalingLayer unscaling_layer;

   unscaling_layer.set(1);

   // Test 0

   assert_true(unscaling_layer.get_neurons_number() == 1, LOG);

   // Test 1

   unscaling_layer.set(3);

   assert_true(unscaling_layer.get_neurons_number() == 3, LOG);
}


void UnscalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   UnscalingLayer unscaling_layer;

   Tensor<Descriptives, 1> descriptives;

   // Test 0

   assert_true(unscaling_layer.get_neurons_number() == 0, LOG);
   assert_true(unscaling_layer.get_neurons_number() == unscaling_layer.get_inputs_number(), LOG);

   // Test 1

   unscaling_layer.set(3);

   descriptives.resize(3);
   unscaling_layer.set_descriptives(descriptives);

   assert_true(unscaling_layer.get_neurons_number() == 3, LOG);
   assert_true(unscaling_layer.get_neurons_number() == unscaling_layer.get_inputs_number(), LOG);
}


void UnscalingLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";

   UnscalingLayer unscaling_layer;

   Tensor<Descriptives, 1> descriptives;

   // Test 0

   assert_true(unscaling_layer.get_inputs_number() == 0, LOG);
   assert_true(unscaling_layer.get_inputs_number() == unscaling_layer.get_neurons_number(), LOG);

   // Test 1

   unscaling_layer.set(3);

   descriptives.resize(3);
   unscaling_layer.set_descriptives(descriptives);

   assert_true(unscaling_layer.get_inputs_number() == 3, LOG);
   assert_true(unscaling_layer.get_inputs_number() == unscaling_layer.get_neurons_number(), LOG);
}


void UnscalingLayerTest::test_get_descriptives()
{
   cout << "test_get_descriptives\n";

   Tensor<Descriptives, 1> descriptives(1);

   UnscalingLayer unscaling_layer(descriptives);

    //    Test 0

    descriptives.resize(1);

    unscaling_layer.set_descriptives(descriptives);

    Tensor<Descriptives, 1> get_des = unscaling_layer.get_descriptives();

    assert_true(get_des.dimension(0) == 1, LOG);
    assert_true(abs(get_des(0).minimum + 1) < static_cast<type>(1e-3), LOG);
    assert_true(abs(get_des(0).maximum - 1) < static_cast<type>(1e-3), LOG);
    assert_true(abs(get_des(0).mean - 0) < static_cast<type>(1e-3), LOG);
    assert_true(abs(get_des(0).standard_deviation - 1) < static_cast<type>(1e-3), LOG);

    // Test 1

    Descriptives des_0(1,1,1,0);
    Descriptives des_1(2,2,2,0);

    descriptives.resize(2);
    descriptives.setValues({des_0,des_1});

    unscaling_layer.set_descriptives(descriptives);

    Tensor<Descriptives, 1> get_des_1 = unscaling_layer.get_descriptives();

    assert_true(get_des_1.dimension(0) == 2, LOG);
    assert_true(abs(get_des_1(1).minimum - 2) < static_cast<type>(1e-3), LOG);
    assert_true(abs(get_des_1(1).maximum - 2) < static_cast<type>(1e-3), LOG);
    assert_true(abs(get_des_1(1).mean - 2) < static_cast<type>(1e-3), LOG);
    assert_true(abs(get_des_1(1).standard_deviation - 0) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_get_descriptives_matrix()
{
   cout << "test_get_descriptives_matrix\n";

   // Test 0

   Tensor<Descriptives, 1> descriptives(1);

   UnscalingLayer unscaling_layer(descriptives);

   assert_true(unscaling_layer.get_descriptives_matrix().size() == 4, LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives.resize(2);

   descriptives(0).minimum = 1;
   descriptives(0).maximum = 1;
   descriptives(0).mean = 1;
   descriptives(0).standard_deviation = 0;

   descriptives(1).minimum = 2;
   descriptives(1).maximum = 2;
   descriptives(1).mean = 2;
   descriptives(1).standard_deviation = 0;

   unscaling_layer.set_descriptives(descriptives);

   assert_true(unscaling_layer.get_descriptives_matrix().size() == 8, LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);

}


void UnscalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";

   UnscalingLayer unscaling_layer(2);

   // Test 0

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_minimums()(0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_minimums()(1) + 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives(0).minimum = 1;
   descriptives(1).minimum = -1;

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_minimums()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_minimums()(1) + 1) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";

   UnscalingLayer unscaling_layer(2);

   // Test 0

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_maximums()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_maximums()(1) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives(0).maximum = 1;
   descriptives(1).maximum = -1;

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_maximums()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_maximums()(1) + 1) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_write_scaling_methods()
{
    cout << "test_get_scaling_method_name\n";

    UnscalingLayer unscaling_layer(1);

    // Test 1

    UnscalingLayer::UnscalingMethod no_unscaling = UnscalingLayer::UnscalingMethod::NoUnscaling;

    UnscalingLayer::UnscalingMethod minimum_maximum = UnscalingLayer::UnscalingMethod::MinimumMaximum;

    UnscalingLayer::UnscalingMethod mean_standard_deviation = UnscalingLayer::UnscalingMethod::MeanStandardDeviation;

    UnscalingLayer::UnscalingMethod logarithmic = UnscalingLayer::UnscalingMethod::Logarithmic;

    unscaling_layer.set_unscaling_methods(no_unscaling);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "NoUnscaling", LOG);

    unscaling_layer.set_unscaling_methods(minimum_maximum);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "MinimumMaximum", LOG);

    unscaling_layer.set_unscaling_methods(mean_standard_deviation);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "MeanStandardDeviation", LOG);

    unscaling_layer.set_unscaling_methods(logarithmic);
    assert_true(unscaling_layer.write_unscaling_methods()(0) == "Logarithmic", LOG);

    // Test 2

    unscaling_layer.set_unscaling_methods(no_unscaling);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "no unscaling", LOG);

    unscaling_layer.set_unscaling_methods(minimum_maximum);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);

    unscaling_layer.set_unscaling_methods(mean_standard_deviation);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "mean and standard deviation", LOG);

    unscaling_layer.set_unscaling_methods(logarithmic);
    assert_true(unscaling_layer.write_unscaling_method_text()(0) == "logarithmic", LOG);

}


void UnscalingLayerTest::test_get_display()
{
   cout << "test_get_display\n";

   UnscalingLayer unscaling_layer;

   assert_true(unscaling_layer.get_display(), LOG);
}


void UnscalingLayerTest::test_set()
{
   cout << "test_set\n";

   UnscalingLayer unscaling_layer;

   // Test 1

   unscaling_layer.set();

   assert_true(unscaling_layer.get_descriptives().size() == 0, LOG);

   Tensor<Descriptives, 1> descriptives(4);
   unscaling_layer.set_descriptives(descriptives);
   unscaling_layer.set();

   assert_true(unscaling_layer.get_descriptives().size() == 0, LOG);

   // Test 2

   unscaling_layer.set();

   Index new_neurons_number(0);
   unscaling_layer.set(new_neurons_number);

   assert_true(unscaling_layer.get_descriptives().size()== 0, LOG);

   Index new_inputs_number_ = 4;
   unscaling_layer.set(new_inputs_number_);

   assert_true(unscaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(unscaling_layer.get_inputs_number()== unscaling_layer.get_descriptives().size(), LOG);

   // Test 3

   unscaling_layer.set();

   Tensor<Descriptives, 1> descriptives_3;
   unscaling_layer.set(descriptives_3);

   assert_true(unscaling_layer.get_descriptives().size()== 0, LOG);

   Tensor<Descriptives, 1> descriptives_3_(4);
   unscaling_layer.set(descriptives_3_);

   assert_true(unscaling_layer.get_descriptives().size()== 4, LOG);

   // Test 4

   unscaling_layer.set();

   UnscalingLayer ul4;
   ul4.set(7);

   unscaling_layer.set(ul4);

   assert_true(unscaling_layer.get_descriptives().size() == 7, LOG);
}


void UnscalingLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

//   UnscalingLayer unscaling_layer;

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

   UnscalingLayer unscaling_layer;

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

   UnscalingLayer unscaling_layer(1);

   unscaling_layer.set_default();

   assert_true(unscaling_layer.get_type() == Layer::Unscaling, LOG);
   assert_true(unscaling_layer.get_type() == 7, LOG);

   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);
}


void UnscalingLayerTest::test_set_descriptives()
{
   cout << "test_set_descriptives\n";

   UnscalingLayer unscaling_layer;

   Tensor<Descriptives, 1> descriptives;

   Descriptives item_descriptives(1,1,1,0);
   Descriptives des_1(2,2,2,0);

   // Test 0

   descriptives.resize(1);

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   item_descriptives.set(1,1,1,0);

   descriptives.resize(1);
   descriptives.setValues({item_descriptives});

   unscaling_layer.set_descriptives(descriptives);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_set_descriptives_eigen()
{
   cout << "test_set_descriptives_eigen\n";


   ScalingLayer scaling_layer(2);

   UnscalingLayer unscaling_layer(1);

   // Test 0

   Tensor<type, 2> descriptives_eigen(1,4);

   unscaling_layer.set_descriptives_eigen(descriptives_eigen);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) + 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,1) + 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) + 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,3) + 0) < static_cast<type>(1e-3), LOG);

   // Test 1

   Tensor<type, 2> descriptives_eigen_(2,4);
   descriptives_eigen_.setValues({{1,1,1,0},{2,2,2,0}});

   scaling_layer.set_descriptives_eigen(descriptives_eigen_);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_set_item_descriptives()
{
   cout << "test_set_item_descriptives\n";

   ScalingLayer ul(1);

   // Test 0

   Descriptives des;

   ul.set_item_descriptives(0,des);

   assert_true(abs(ul.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(ul.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(ul.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(ul.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   UnscalingLayer ul1(2);

   // Test

   Descriptives des_0(1,1,1,0);
   Descriptives des_1(2,2,2,0);

   ul1.set_item_descriptives(0,des_0);
   ul1.set_item_descriptives(1,des_1);

   assert_true(abs(ul1.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(ul1.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(ul1.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(ul1.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);
}

void UnscalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";

   UnscalingLayer unscaling_layer(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_minimum(0, -5);
   unscaling_layer.set_minimum(1, -6);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,0) + 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,0) + 6) < static_cast<type>(1e-3), LOG);
}

void UnscalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";

   UnscalingLayer unscaling_layer(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_maximum(0, 5);
   unscaling_layer.set_maximum(1, 6);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,1) - 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,1) - 6) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";

   UnscalingLayer unscaling_layer(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_mean(0, 5);
   unscaling_layer.set_mean(1, 6);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,2) - 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,2) - 6) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   UnscalingLayer unscaling_layer;
   Tensor<Descriptives, 1> descriptives;

   // Test 1

   unscaling_layer.set(2);

   descriptives.resize(2);

   unscaling_layer.set_descriptives(descriptives);

   unscaling_layer.set_standard_deviation(0, 5);
   unscaling_layer.set_standard_deviation(1, 6);

   assert_true(abs(unscaling_layer.get_descriptives_matrix()(0,3) - 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(unscaling_layer.get_descriptives_matrix()(1,3) - 6) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_set_unscaling_method()
{
   cout << "test_set_unscaling_method\n";

   UnscalingLayer unscaling_layer(1);

   // Test 1

   unscaling_layer.set_unscaling_methods(UnscalingLayer::UnscalingMethod::NoUnscaling);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "no unscaling", LOG);

   unscaling_layer.set_unscaling_methods(UnscalingLayer::UnscalingMethod::MinimumMaximum);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "minimum and maximum", LOG);


   unscaling_layer.set_unscaling_methods(UnscalingLayer::UnscalingMethod::MeanStandardDeviation);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "mean and standard deviation", LOG);

   unscaling_layer.set_unscaling_methods(UnscalingLayer::UnscalingMethod::Logarithmic);
   assert_true(unscaling_layer.write_unscaling_method_text()(0) == "logarithmic", LOG);

}

void UnscalingLayerTest::test_set_display()
{
   cout << "test_set_display\n";

   bool display_true = true;
   bool display_false = false;

   set_display(display_true);
   assert_true(get_display(), LOG);

   set_display(display_false);
   assert_true(!get_display(), LOG);
}


void UnscalingLayerTest::test_is_empty()
{
    cout << "test_is_empty\n";

    UnscalingLayer unscaling_layer;

    // Test

    assert_true(unscaling_layer.is_empty(), LOG);

    // Test

    unscaling_layer.set(1);

    assert_true(!unscaling_layer.is_empty(), LOG);
}


void UnscalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   UnscalingLayer unscaling_layer;

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   Tensor<type, 2> minimums_maximums;
   Tensor<type, 2> mean_standard_deviation;

   Tensor<type, 2> standard_deviation;

   unscaling_layer.set_display(false);

   // Test 0_0

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::NoUnscaling);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);
   assert_true(abs(outputs(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 0_1

   unscaling_layer.set(3);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::NoUnscaling);

   inputs.resize(1,3);
   inputs.setConstant(0);
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 3, LOG);
   assert_true(abs(outputs(0) - static_cast<type>(0)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(0)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(2) - static_cast<type>(0)) < static_cast<type>(1e-3), LOG);

   // Test 1_0

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::MinimumMaximum);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(outputs.dimension(1) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 1_1

   unscaling_layer.set(2);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::MinimumMaximum);

   minimums_maximums.resize(2,4);
   minimums_maximums.setValues({{-1000,1000,0,0},{-100,100,0,0}});

   unscaling_layer.set_descriptives_eigen(minimums_maximums);
   inputs.resize(1,2);
   inputs.setValues({{0.1f,0}});
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(outputs.dimension(1) - 2 < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0) - static_cast<type>(100)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(0)) < static_cast<type>(1e-3), LOG);

   // Test 2_0

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::MeanStandardDeviation);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(outputs.dimension(1) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 2_1

   unscaling_layer.set(2);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::MeanStandardDeviation);

   mean_standard_deviation.resize(2,4);
   mean_standard_deviation.setValues({{-1,1,-1,-2},{-1,1,2,3}});

   unscaling_layer.set_descriptives_eigen(mean_standard_deviation);
   inputs.resize(1,2);
   inputs.setValues({{-1,1}});
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(outputs.dimension(1) - 2 < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(5)) < static_cast<type>(1e-3), LOG);

   // Test 3_0

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::Logarithmic);

   inputs.resize(1,1);
   outputs = unscaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) - 1 < static_cast<type>(1e-3), LOG);
   assert_true(outputs.dimension(1) - 1 < static_cast<type>(1e-3), LOG);

   assert_true(abs(outputs(0) - 1) < static_cast<type>(1e-3), LOG);

   // Test 3_1

   unscaling_layer.set(2);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::Logarithmic);

   standard_deviation.resize(2,4);
   standard_deviation.setValues({{-1,1,-1,2},{-1,1,1,4}});

   unscaling_layer.set_descriptives_eigen(standard_deviation);
   inputs.resize(1,2);
   inputs.setConstant(1);
   outputs = unscaling_layer.calculate_outputs(inputs);

   assert_true(abs(outputs.dimension(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs.dimension(1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0) - static_cast<type>(2.7182)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(2.7182)) < static_cast<type>(1e-3), LOG);
}


void UnscalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   UnscalingLayer unscaling_layer;

   Tensor<string, 1> inputs_names(1);
   Tensor<string, 1> outputs_names(1);

   string expression;

   // Test 1

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::NoUnscaling);
   inputs_names.setValues({"x"});
   outputs_names.setValues({"y"});

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = x;\n", LOG);

   // Test 2

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::MinimumMaximum);

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);

   assert_true(expression == "y = x*(1+1)/(1+1)-1+1*(1+1)/(1+1);\n", LOG);

   // Test 3

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::MeanStandardDeviation);

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = -1+0.5*(x+1)*((1)-(-1);\n", LOG);

   // Test 4

   unscaling_layer.set(1);
   unscaling_layer.set_unscaling_methods(UnscalingLayer::Logarithmic);

   expression = unscaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = -1+0.5*(exp(x)+1)*((1)-(-1));\n", LOG);
}


void UnscalingLayerTest::run_test_case()
{
   cout << "Running unscaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();


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
   test_get_descriptives_matrix();


   // Display messages

   test_get_display();


   // Set methods

   test_set();
   test_set_inputs_number();
   test_set_neurons_number();
   test_set_default();


   // Output variables descriptives

   test_set_descriptives();
   test_set_descriptives_eigen();
   test_set_item_descriptives();
   test_set_minimum();
   test_set_maximum();
   test_set_mean();
   test_set_standard_deviation();


   // Variables descriptives

   // Variables scaling and unscaling

   test_write_scaling_methods();


   // Display messages

   test_set_display();


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
